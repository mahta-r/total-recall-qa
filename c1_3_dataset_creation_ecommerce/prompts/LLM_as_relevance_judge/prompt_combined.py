PROPERTY_CHECK_INPUT_TEMPLATE = """
PRODUCT: [{product_type}]
ENTITY: [{entity_name}]
PROPERTY: [{property_name}]
STATEMENT: 
the [{property_name}] property of [{entity_name}] is [{property_value}]

PASSAGE:
{passage}
"""


PROPERTY_CHECK_PROMPT = """
Your task is to identify if input passages contain any mention of the value of a given property for an entity (product). 
If the passage mentions the property's value, you must identify if the value mentioned matches a given value: same or different.
If the passage does not mention the property's value, you must identify if passage topic is related to the property: related or unrelated.

You are given the product type, an entity (product), a property of that product, and a statement about the value of that property for that entity (product). 
You are also given a passage about the product that may contain information about the value of the property for that entity (product). 

-------------------------------------- Steps to follow --------------------------------------

- - - - - - - - - - - - Step 1 - - - - - - - - - - - -

1) First, you will determine if the passage mentions the value of the property for the entity.
For this first task, you only care if that property is mentioned and its value is included in the text. You don't care what that value is.
Very important: the name or value of property may not be lexically mentioned in the passage, but it can be mentioned semnatically. 
For example, "Water Resistant = Yes" can be mentioned as "This product can resist water" in the passage which is a match.
The value of the property might be mentioned but referring to another product or a general standard rather than the given product, in that case the answer for task 1 is NO.
Your answer for this first task will be strictly either "YES" or "NO", without any additional words.

- - - - - - - - - - - - Step 2-YES - - - - - - - - - - - -

2-YES) If the passage mentions the value of the property for the entity/product (answer to the first task is YES), 
you will then determine if the value mentioned in the passage matches the value given in the input statement.
If value is mentioned multiple times in passage, all mentioned values need to match the statement for you to answer SAME.
If at least one mentioned value is different than the value in the statement, you should answer DIFFERENT.

{statement_explanation}

For task 2-YES, your answer should be "SAME" or "DIFFERENT", without any additional words. 
Your answer should be SAME if the value in the statement matches the value in the passage (based on the matching criteria) and DIFFERENT if the values are different.

- - - - - - - - - - - - Step 2-NO - - - - - - - - - - - -

2-NO) If the passage does not mention the value of the property for the entity (answer to the first task is NO),
You will then determine if the topic of the passage is related to the property defined in the input or not.
Your criteria for deciding if the topic is related is to consider we want to add a sentence to this passage that mentions the value of the property:
If that sentence can fit naturally in the passage without making it inconsistent or illogical --> RELATED
If the sentence would be out-of-place in that passage --> UNRELATED 
For task 2-NO, your answer should be "RELATED" or "UNRELATED", without any additional words. 

- - - - - - - - - - - - Step 3 - - - - - - - - - - - -

3) Your final answer will be the combination of the answer to task 1 and task 2, separated by a single dash.
That means your final answer will be one of the following 4 options:

"YES-SAME"
"YES-DIFFERENT"
"NO-RELATED"
"NO-UNRELATED"

You should not explain your answer in any way, just one of the above options in a single line without any additional words/whitespace.


-------------------------------------- Examples --------------------------------------

- - - - - - - - - - - - Example 1 - - - - - - - - - - - -

PRODUCT: [Hard Drives]
ENTITY: [HD-SM-FHKN]
PROPERTY: [Usage Type]
STATEMENT: 
the [Usage Type] property of [HD-SM-FHKN] is [Desktop]

PASSAGE: 
lines of the HD-SM-FHKN as a trusted friend, fitting perfectly into your laptop or external enclosure without breaking stride. Its compact design is not just a matter of space—it’s about creating a connection. Each time you open your laptop or drawer, it’s like a warm embrace, reminding you of the importance of organization and order in your digital life. **External Interface: The Unshackling of Data** Picture yourself, feeling the weight of your data lifting off your shoulders. With SATA I interface, HD-SM-FHKN ensures that your files, photos, and videos are always within reach, no matter where you go. It’s like

- - -
NO-UNRELATED


- - - - - - - - - - - - Example 2 - - - - - - - - - - - -

PRODUCT: [Alcoholic Beverages]
ENTITY: [ALB-GU-B7AQ]
PROPERTY: [Vintage]
STATEMENT: 
the [Vintage] property of [ALB-GU-B7AQ] is [2000s]

PASSAGE:
Taste of the Past** There’s something undeniably charming about the ALB-GU-B7AQ Hard Selzer’s vintage appeal. With its roots tracing back to the 2000s, this beverage encapsulates the essence of a time when flavors were bold yet balanced. The wine-like notes in this hard selzer are a testament to the careful craftsmanship and attention to detail that went into its creation. It’s perfect for those who appreciate the elegance and refinement of a bygone era, making it a must-have for any cocktail connoisseur. **Organic Indice: Purity at Its Finest** When it comes to purity and quality, the ALB-GU-B7AQ Hard Selzer stands

- - -
YES-SAME


- - - - - - - - - - - - Example 3 - - - - - - - - - - - -

PRODUCT: [Cutting Boards]
ENTITY: [CuB-TH-OFL-X]
PROPERTY: [Reversible]
STATEMENT: 
the [Reversible] property of [CuB-TH-OFL-X] is [No]


PASSAGE:
necessarily a drawback. Instead, it allows for more focused care and maintenance on one side at a time. This means you can focus on keeping the cutting surface clean and free from bacterial growth, which is crucial for food safety. ### Brand Reputation and Quality Assurance Teakhaus is a brand known for its commitment to quality and sustainability. The CuB-TH-OFL-X is crafted with the same dedication to excellence that has made Teakhaus a favorite among home cooks and professional chefs alike. Their reputation for producing high-quality, durable products is well-deserved. ### Real-World Benefits and Everyday Use Cases Imagine slicing a

- - -
NO-RELATED


- - - - - - - - - - - - Example 4 - - - - - - - - - - - -

PRODUCT: [Locks and Latches]
ENTITY: [LL-Kw-7DWU]
PROPERTY: [Finish]
STATEMENT: 
the [Finish] property of [LL-Kw-7DWU] is [satin nickel]

PASSAGE:
to corrosion. Unlike cheaper aluminum or zinc options, this material provides a long-lasting solution for your home’s entry points. The stainless steel finish ensures that the lock remains lustrous and protected against the elements, making it a perfect fit for any home environment. ### Sleek Satin Nickel Finish The satin nickel finish of the LL-Kw-7DWU Deadbolt offers a sophisticated look that complements modern and traditional decor. Unlike chrome finishes that can tarnish over time, the satin finish maintains its aesthetic appeal, providing a timeless appearance that will enhance your home’s curb appeal. This feature is particularly appreciated by homeowners who

- - -
YES-DIFFERENT


- - - - - - - - - - - - Example 5 - - - - - - - - - - - -

PRODUCT: [Backpacks]
ENTITY: [BCKP-MIL-K993]
PROPERTY: [Water Resistant]
STATEMENT: 
the [Water Resistant] property of [BCKP-MIL-K993] is [No]

PASSAGE:
### Unleash the Great Outdoors with the BCKP-MIL-K993 Trail Backpack Dive into the world of outdoor adventures with the BCKP-MIL-K993 Trail Backpack from Millican. This rugged, reliable pack is designed specifically for hikers, campers, and anyone who craves the freedom of the trails. With a capacity of 18 liters, this backpack offers ample space for your gear while maintaining a compact size for easy portability. #### Water-Resistant Design: Keeping You Dry in Any Weather One of the standout features of the BCKP-MIL-K993 is its water-resistant exterior. While not fully waterproof, this design ensures that rain or light showers won't soak

- - -
YES-DIFFERENT


"""


DATATYPE_EXPLANATION = {
    "Quantity": """
        The type of the given property is numerical. 
        The statement will contain the numerical value of the property, and potentially the unit of the numerical value.
        Values match if both the numerical value and unit (if given) are the same in the passage. If unit is not mentioned in passage, only the numerical value needs to match.
        The values don't need to be exactly the same for a match, a small difference is acceptable.
        The formatting of the numerical value in the passage can vary, for example "1,000" is equivalent to "1000" and "1e3".
    """,
    "Date": """
        The type of the given property is dates (a year). 
        The statement will contain the date value of the property, which would be a year number.
        Values match if the year in the statement the same in the passage.
    """,
    "OrderedString": """
        The type of the given property is categorical.
        The statement will contain the value of that categorical property (a string).
        Values match if the string value in statement is mentioned as value of property in the passage.
        The formatting of the string value in the passage can vary, we only care about semantic matching.
        For example "Occasion = Celebration" in statement matches "Celebratory" in passage.
    """,
    "String": """
        The type of the given property is categorical.
        The statement will contain the value of that categorical property (a string).
        Values match if the string value in statement is mentioned as value of property in the passage.
        The formatting of the string value in the passage can vary, we only care about semantic matching.
        For example "Occasion = Celebration" in statement matches "Celebratory" in passage.
    """
}