QUERY_GENERATION_PROMPT = """
Your task is to generate total-recall queries based on inputs provided.
A total-recall query is a query that requires multiple documents to be retrieved and aggregated to answer correctly. 
If any of the documents are missing, crucial information will be lost, and the query cannot be answered accurately.

The inputs provided for creating a total-recall query are:
-----------------------------------
1) Entity Set (Product Category Specification)
This section defines the set of products (entities) the query operates on.

The information provided includes:
- Product category, number of products
-----------------------------------
2) Constraint Property (Multiple Constraints Possible)
A set of constraint properties used to filter the entity set before aggregation.

Each constraint property has the following information:
- Constraint property: label, datatype
- Constraint: filter name and description (how entities are filtered by this property e.g., greater than, before, contains)
- Reference: reference value that the constraint property gets compared to 

When multiple constraints are provided, all of them must be applied together to filter the entity set before aggregation.
This means we are taking the intersection (AND) of all the constraints to filter the entity set. (all constraints must be satisfied togethe)
-----------------------------------
3) Aggregation Property (Optional)
This is the property whose values are aggregated to answer the query.

If not provided, the aggregation will be the count of entities satisfying the constraints.
If provided, information includes:
- Aggregation property: label, datatype, unit (optional, only for numerical properties)
-----------------------------------
4) Aggregation Operation
This is the operation that must be expressed in the query and applied to the aggregation property after filtering.

Provided information includes:
- Operation name and arguments and description: arguments may include a reference value 
- Final answer: this is provided only for context and must never be used or mentioned in the query.
- Answer unit: the unit of the answer the query expects, which needs to be specified if not obvious.
-----------------------------------

There are 4 property datatypes that can be used for constraint properties and aggregation properties:
- Quantity: numeric value with an optional unit (if unit is not obvious)
- Date: a year number (year-based attributes such as production or publication year)
- OrderedString: a string value that has a total order defined over it (e.g., sizes like Small < Medium < Large)
- String: categorical string value with no order defined (e.g., colors like Red, Green, Blue)


You will create a natural language query that aggregates values of the aggregation property over the set of filtered entities using the specified operation.
Here's details of how you'll use inputs to construct the query:

-- Entity Set (Product Category):
You must clearly specify the category of products that are subject of the query. 

-- Constraint Properties:
For each constraint property, you must express the constraint in natural language in the query to filter the entity set before aggregation.
Almost always, copying the format of the constraint presented will be terribly unnatural in English, so you should rephrase the constraints into a more natural form while preserving the exact meaning.
Examples are:
Product: Backpack | Constraint Property: Laptop Compartment | Constraint: IS_NOT | Reference: 'No' --> IS_NOT 'No' is negation of negation, so it means the presence of a laptop compartment. A natural way to say this in English is "with a laptop compartment" or "that has a laptop compartment".
Product: Bird Food | Constraint Property: Organic | Constraint: IS_NOT | Reference: 'Yes' --> IS_NOT 'Yes' means non-organic. A natural way to say this in English is "non-organic" or "that is not organic".
Product: Alcoholic Beverages | Constraint Property: Vintage | Constraint: IS_ANY | Reference: ['2000s', 'Non-Vintage', '2010s'] --> IS_ANY ['2000s', 'Non-Vintage', '2010s'] --> "either non-vintage or vintage from 2000s or 2010s"

When there are multiple constraint properties, you must combine them together in the query using AND logic to filter the entity set.
You must find the simplest and shortest way to express the combination of constraints in natural language, while ensuring the meaning is clear and unambiguous.
In English, there is a natural order to list adjectives of an entity; you should follow that natural order when listing multiple constraints in the query instead of following the constraints' order.
The constraints must be applied conceptually before aggregation.

Very important: for some queries, the last constraint provided will be: Constraint Property: Price | Constraint: IS | Reference: 'Listed/Mentioned'
For these constraints, the query will first apply other constraints, and then asks for the subset of the filtered entities that have the property listed or mentioned.
'Listed/Mentioned' indicates presence of a property (e.g.) price, not a specific amount. Do not imply any explicit value when using a 'Listed/Mentioned' constraint
Example:
Product: Cutting Boards
Constraint 1: Type [String] | IS ['Serving Board']
Constraint 2: Surface Finish [String] | IS_NOT ['Textured']
Constraint 3: Color [String] | IS ['Dark Brown']
Constraint 4: Price [Quantity] | IS Listed/Mentioned
--> "Dark brown serving boards that are not textured and are listed with a price / have price listed"

-- Aggregation Property + Aggregation Operation:
The query must ask for the result of applying the specified operation to the aggregation property (unless operation is count) over the filtered entity set.
The query must ask for a single-response: the answer cannot be multiple values or a list. It must be a single numerical value. 
The final answer itself must never be stated or hinted at in the query.

Notes:
* If a unit is provided for numerical values and it is not obvious, you must specify it in the query. If no unit is provided, do not mention one.
* The query must be independently readable. Do not refer to the provided inputs/properties using pronouns or placeholders. 
* Never use any form of identifiers (IDs) in the query text.

* If you can express a constraint as an adjective before the product category, that is more natural than phrasing it as a separate clause. 
Example:
Product: Alcoholic Beverages
Constraint 1: Serving Temperature [String] | IS_NOT ['Cool']
Constraint 2: Packaging Type [String] | IS_ANY ['Plastic Bottle', 'Can']
Constraint 3: Ingredients Base [String] | IS_NOT ['Rye']
--> "Non-Rye-based alcoholic beverages that are not served cool and are packaged in either plastic bottles or cans"

* If a constraint mentions a type of product that is more specific, you can name the type and skip product category to make the query more natural and shorter.
Example:
Class: Books | #59519
Constraint 1: Reference Type [String] | IS ['Dictionary']
Constraint 2: Edition [String] | IS ['Revised Edition']
Constraint 3: Language [String] | IS_ANY ['German', 'Italian', 'Chinese']
Constraint 4: Format [String] | IS_ANY ['Audiobook', 'PDF']
--> "Revised edition dictionaries in German, Italian, or Chinese that are available as audiobooks or PDFs"

* When phrasing the query, rewrite the constraints and conditions into natural, concise English rather than mirroring the structure of the input text. 
Prefer fluent rephrasings that sound natural to a human reader, while preserving the exact meaning.

*For property names, rephrase them to a natural-sounding phrase if needed. 
Example: Bowl | Special Occasion | IS | ['Thanksgiving'] --> "Bowls for Thanksgiving" or "Thanksgiving-themed bowls".
To do this, first internally form the query in your mind, then rephrase it into a more natural human-readable form (preserving the exact meaning).

Your output must be exactly one line containing only the natural language query text.
Do not include quotes, tags, brackets, or any extra characters around the query text. Do not explain your reasoning.
Your output will be directly used as a search query, so do not include any extra text.


- - - - - - - - - - - - Example 1 - - - - - - - - - - - -

---------- INPUT ----------
ENTITY SET
|  Product Category: Desktop Computers | #29540
CONSTRAINT PROPERTY
|  Constraint Property: Made In | String
|  Constraint: IS | only products (entities) that have a specific value for this property
|  Reference: [China]
CONSTRAINT PROPERTY
|  Constraint Property: Production Year | Date
|  Constraint: AFTER | only products (entities) whose time value occurs after a given reference date
|  Reference: 2025
CONSTRAINT PROPERTY
|  Constraint Property: Average Rating | Quantity
|  Constraint: GT | only products (entities) whose value is strictly greater than a given reference value
|  Reference: 3.5
AGGREGATION PROPERTY
|  Aggregation Property: Processor Speed | Quantity
|  Unit: Gigahertz
AGGREGATION OPERATION
|  Operation: MOST_COMMON | the value that appears most frequently among all products (entities)
|  Final Answer: 1.44
|  Answer Unit: ghz

---------- YOUR ANSWER (QUERY) ----------
What is the most common processor speed in GHz among desktop computers made in China that were produced after 2025 and have an average rating above 3.5?

- - - - - - - - - - - - Example 2 - - - - - - - - - - - -

---------- INPUT ----------
ENTITY SET
|  Product Category: Cutting Boards | #2947
CONSTRAINT PROPERTY
|  Constraint Property: Shape | String
|  Constraint: IS | only products (entities) that have a specific value for this property
|  Reference: [Square]
CONSTRAINT PROPERTY
|  Constraint Property: Size | String
|  Constraint: IS_ANY | only products (entities) whose value is one of a given set of values
|  Reference: [Extra Large, Large]
CONSTRAINT PROPERTY
|  Constraint Property: Type | String
|  Constraint: IS_ANY | only products (entities) whose value is one of a given set of values
|  Reference: [Cheese Board, Chopping Block, Utility Board]
CONSTRAINT PROPERTY
|  Constraint Property: Brand | String
|  Constraint: IS | only products (entities) that have a specific value for this property
|  Reference: [TeakHaus]
AGGREGATION OPERATION
|  Operation: COUNT | the total number of remaining products (entities) after all constraints are applied
|  Final Answer: 4

---------- YOUR ANSWER (QUERY) ----------
How many large or extra-large square-shaped TeakHaus cheese boards, chopping blocks, or utility boards can you find?

- - - - - - - - - - - - Example 3 - - - - - - - - - - - -

---------- INPUT ----------
ENTITY SET
|  Product Category: Dishwashers | #989
CONSTRAINT PROPERTY
|  Constraint Property: Capacity (Place Settings) | Quantity
|  Constraint: GT | only products (entities) whose value is strictly greater than a given reference value
|  Reference: 10.0
CONSTRAINT PROPERTY
|  Constraint Property: Smart Features | String
|  Constraint: IS | only products (entities) that have a specific value for this property
|  Reference: [Smart Diagnostics]
AGGREGATION PROPERTY
|  Aggregation Property: Width | Quantity
|  Unit: Inch
AGGREGATION OPERATION
|  Operation: MEDIAN | the median (middle) value when all values are ordered
|  Final Answer: 24.0
|  Answer Unit: inch

---------- YOUR ANSWER (QUERY) ----------
What is the median width, in inches, of dishwashers with a capacity of more than 10 place settings that have smart diagnostics features?

"""
