QUERY_GENERATION_PROMPT = """
Your task is to generate total-recall queries based on inputs provided.
A total-recall query is a query that requires multiple documents to be retrieved and aggregated to answer correctly. 
If any of the documents are missing, crucial information will be lost, and the query cannot be answered accurately.

The inputs provided for creating a total-recall query are:
-----------------------------------
1) Entity Set (Class Specification)
This section defines the set of entities over which aggregation is performed. The entity set can be either direct or multi-hop.

If the entity set is direct, only source class information is provided:
- Source class label, number of instances, and description

If the entity set is multi-hop, three components are provided:
- Source class (hop 0): label, number of instances, and description
- Connecting property: label, datatype, description, point-in-time (if applicable, the time at which connections are valid)
  This property connects source-class entities to hop-1 entities.
- Hop class (hop 1): label, number of instances, and description
  This is the most specific common class shared by the hop-1 entities.

Hop class information should be used only if it helps clearly specify the intended set of entities without naming them individually.
-----------------------------------
2) Constraint Property (Optional)
A constraint property is used to filter the entity set before aggregation.

If present, the following information is provided:
- Constraint property: label, datatype, and description
- Point-in-time (if applicable): the time at which constraint property values are valid
- Constraint: filter name and description of how entities are filtered (e.g., greater than, before, contains)
- Reference entity: entity name and entity's value for constraint property

This reference is an entity whose property value is used as the reference for comparison.
When stating the constraint in the query, you may choose either:
- the reference entity name, or
- the explicit reference value
For example: “greater than mean age of Texas” or “greater than 34.8”.
-----------------------------------
3) Aggregation Property
This is the property whose values are aggregated to answer the query.

Provided information includes:
- Aggregation property: label, datatype, and description
- Point-in-time (if applicable): the time at which aggregation property values are valid
- Datatype-specific metadata:
  - Quantity: unit
  - Time: calendar and precision
  - WikibaseItem: item class (the most specific common class among referenced items)
  - GlobeCoordinate: globe

Item class information should be used in wording the query only if it meaningfully improves clarity and helps specify the intended entities.

In multi-hop entity sets, the aggregation property always applies to the hop class entities, not the source class.
The query must therefore ask for aggregation over the hop class, while the source class and connecting property are used only to define which hop-class entities are included.
-----------------------------------
4) Aggregation Operation
This is the operation that must be expressed in the query and applied to the aggregation property after filtering.

Provided information includes:
- Operation name and arguments: if arguments include a reference entity with both entity name and reference value, you should select one of them to used.
- Final answer: this is provided only for context and must never be used or mentioned in the query.
- Answer unit: the unit of the answer the query expects, which needs to be specified if not obvious.
-----------------------------------


There are 4 property datatypes, each with different metadata:
- Quantity: numeric value with a unit (if unit is not obvious)
- Time: value with calendar model and precision
- WikibaseItem: reference(s) to another entity, possibly with a shared item class
- GlobeCoordinate: coordinate value 


You will create a natural language query that aggregates values of the aggregation property over the set of entities using the specified operation.
Here's details of how you'll use inputs to construct the query:
- Entity Set:
You must clearly specify the subject entity set so that the intended entities can be inferred without individually naming them. 
The entity set may be defined directly by a single class, or indirectly via a multi-hop construction using a connecting property to the hop class. 
When specifying multi-hop entity sets, use source class and connecting property, and if that doesn't clearly specify the entities, include hop class information.
- Constraint Property:
If a constraint property is provided, the query must incorporate the constraint to filter entities before aggregation. 
The constraint should be expressed naturally, using either the reference entity name or the explicit reference value, but not both. 
The constraint must be applied conceptually before aggregation.
- Aggregation Property + Aggregation Operation:
The query must ask for the result of applying the specified operation to the aggregation property over the filtered entity set.
The query must ask for a single-response: the answer cannot be multiple values or a list. It must be a single numerical value. 
The final answer itself must never be stated or hinted at in the query.


Notes:
* If a point-in-time is provided for the aggregation property or constraint property, you must explicitly specify it in the query. If no point-in-time is provided, do not mention time. 
* If a unit is provided for numerical values and it is not obvious, you must specify it in the query. If no unit is provided, do not mention one.

* You are provided with labels and descriptions for entity classes and properties. You may use descriptions only when the label alone is not sufficient to clearly identify the intended entities or attribute.
For example, if the class label is ambiguous:
  label = "independent component city"
  description = "legal class for a city in the Philippines"
then the description should be used to clarify.
The goal is to infer the entities from the query without explicitly naming them.
If the label is already specific and unambiguous, do not restate the description.

* The query must be independently readable. Do not refer to the provided inputs/entities/properties using pronouns or placeholders. 
Do not explicitly name individual entities unless absolutely necessary. 
For example, you may say “all NBA teams”, but you must not list specific teams such as the LA Lakers or Boston Celtics.

* Never use Wikidata identifiers (e.g., Q123456, P789) or any other IDs in the query text.

When phrasing the query, rewrite the constraints and conditions into natural, concise English rather than mirroring the structure of the input text. 
Prefer fluent rephrasings that sound natural to a human reader, while preserving the exact meaning.
For example, if multiple properties have similar point in times, just state it once.
For property names, rephrase them to a natural-sounding phrase if needed e.g. instead of "have a work location of" use "works in".
To do this, first internally form the query in your mind, then rephrase it into a more natural human-readable form (preserving the exact meaning).

Your output must be exactly one line containing only the natural language query text.
Do not include quotes, tags, brackets, or any extra characters around the query text. Do not explain your reasoning.
Your output will be directly used as a search query, so do not include any extra text.


- - - - - - - - - - - - Example 1 - - - - - - - - - - - -

---------- INPUT ----------
ENTITY SET
|  Source Class: federated state of Germany (Q1221156) | #16 | administrative division of the Federal Republic of Germany
|  Connecting Property: head of government (P6) | WikibaseItem | head of the executive power of this town, city, municipality, state, country, or other governmental body
|  At Time: 2025/5/20 (20 May 2025)
|  Hop Class: human (Q5) | #16 | any single member of Homo sapiens, unique extant species of the genus Homo
CONSTRAINT PROPERTY
|  Constraint Property: date of birth (P569) | Time | date on which the subject was born
|  At Time: -
|  Constraint: BEFORE | only entities whose time value occurs before a given reference date
|  Reference Entity: [Hendrik Wüst] | [1975/7/19 (19 July 1975) (calendar: proleptic Gregorian calendar)]
AGGREGATION PROPERTY
|  Aggregation Property: position held (P39) | WikibaseItem | subject currently or formerly holds the object position or public office
|  Item Class: position
|  At Time: 2025/5/24 (24 May 2025)
AGGREGATION OPERATION
|  Operation: COUNT_NOT_HAS_ANY_X, Reference=[member of the Landtag of Saxony-Anhalt, Member of the Bundesrat of Germany] | how many entities have none of the values from a given set (X) of specified entities for this property
|  Final Answer: 7

---------- YOUR ANSWER (QUERY) ----------
How many heads of government of Germany’s federated states, as of May 20, 2025, who were born before Hendrik Wüst were not members of the Landtag of Saxony-Anhalt or the Bundesrat of Germany as of May 24, 2025?

- - - - - - - - - - - - Example 2 - - - - - - - - - - - -

---------- INPUT ----------
ENTITY SET
|  Source Class: alpine main part (Q131311255) | #2 | 1st level entity in the taxonomy of the Alps according to the International Standardized Mountain Subdivision of the Alps (SOIUSA)
|  Connecting Property: country (P17) | WikibaseItem | sovereign state that this item is in (not to be used for human beings)
|  At Time: -
|  Hop Class: sovereign state (Q3624078) | #8 | state that has the highest authority over a territory
CONSTRAINT PROPERTY
|  Constraint Property: coordinates of southernmost point (P1333) | GlobeCoordinate | southernmost point of a place. For administrative entities this includes offshore islands
|  At Time: -
|  Constraint: W | only entities located west of a given reference location
|  Reference Entity: [Italy] | [(36.64674684, 15.07981896)]
AGGREGATION PROPERTY
|  Aggregation Property: urban population (P6343) | Quantity | number of people living within the territorial entity who live in its urban parts
|  At Time: 2022
AGGREGATION OPERATION
|  Operation: SUM_BOTTOM_K, K=3 | the total amount for the bottom K entities (ranked by this value)
|  Final Answer: 5400025.0

---------- YOUR ANSWER (QUERY) ----------
What was the total urban population in 2022 of the three countries with the smallest urban populations that are associated with alpine main parts and are located west of Italy?

- - - - - - - - - - - - Example 3 - - - - - - - - - - - -

---------- INPUT ----------
ENTITY SET
|  Source Class: province of Canada (Q11828004) | #10 | type of administrative division of Canada
CONSTRAINT PROPERTY
|  Constraint Property: basic form of government (P122) | WikibaseItem | subject's government
|  Item Class: form of government
|  At Time: -
|  Constraint: NOT_HAS | only entities that do not have a specific entity as a value of this property
|  Reference Entity: [parliamentary system]
AGGREGATION PROPERTY
|  Aggregation Property: total fertility rate (P4841) | Quantity | average number of children that would be born to a woman over her lifetime who lives in this territorial entity
|  At Time: 2019
AGGREGATION OPERATION
|  Operation: AVG_TOP_K, K=3 | the average value among the top K entities (ranked by this value)
|  Final Answer: 1.6469333333333334

---------- YOUR ANSWER (QUERY) ----------
What was the average total fertility rate in 2019 of the three Canadian provinces with the highest total fertility rates that did not have a parliamentary system of government?

"""
