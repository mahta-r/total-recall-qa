INPUT_TEMPLATE = """
PRODUCT: [{product_type}]
ENTITY: [{entity_name}]
PROPERTY: [{property_name}]
STATEMENT: 
the [{property_name}] property of [{entity_name}] is [{property_value}]

PASSAGE:
{passage}
"""


REWRITE_PROMPT = """
Your task is to rewrite specific parts of an input passage to include information about the value of a given property for an entity (product).
Other than the part about value of property, the rest of the passage should remain unchanged, exactly same as it was given in input.
You are given the product type, topic entity (product), and a statement about the value of that property for that entity. 
The statement will be in the form: "the [property] of [entity] is [value]". This might not be a sound sentence in English.
It is very important to rephrase the statement to a more natural form in English, and not to copy the exact wording from the statement.
For example: 
"the [Vintage] property of [ALB-GU-B7AQ] is [2000s]" --> "ALB-GU-B7AQ is a vintage alcoholic beverage from 2000s"
"the [Water Resistant] property of [BCKP-MIL-K993] is [No]" --> "BCKP-MIL-K993 is not a water resistant backpack"

You are also given the text of the passage about entity, which may or may not include the value of the property mentioned in the statement.

{rewrite_explanation}

If the passage is mentioning values of other properties anywhere, including in sentences you are rewriting, you are STRICTLY FORBIDDEN to change those values. 
You can only change the value of the property mentioned in the statement and the surrounding context if necessary.
When adding the property value to the passage, do not include the brackets around property, entity, or value in the prompt.
Very important: Other than the value of the property, you must not change anything else in the passage. 
Ensure the rest of passage stays exactly the same, and output it as it was given in input, only changing/adding the necessary text about the property value.

Your output should be ONLY the rewritten passage text, without any additional explanations or comments.
You will only include the rewritten text of the passage, which is expected to be very similar to the input passage except for sentences about property value.

"""



REWRITE_EXPLANATION = {
    "REPLACE": """
        The value of the property is mentioned in the given passage one or more times, but it does not match the value given in the statement.
        You must only replace all incorrect values mentioned in the passage with the value given in the statement.
        The property in statement may be mentioned in consecutive sentences that make sense together semantically. 
        If the context sentences around the mentioned property don't make sense after replacing the value, you need to change the surrounding sentences as well to make sure the passage is coherent and fluent after rewriting.
        BUT, you still have to change the least amount of text possible, and keep the rest of passage exactly the same.
        Very important: you are FORBIDDEN from copying the statement, you must rephrase the statement to a more natural form if writing or replacing sentences.
    """,
    "ADD": """
        The value of the property is not mentioned in the passage.
        You must add a sentence/phrase to the passage that mentions the value of the property for the entity, while ensuring the rest of passage stays exactly the same.
        You will do this in 2 steps. 
        Step 1: rephrase the statement to another form (sentence or phrase), consistent with style of passage. (internal, DO NOT OUTPUT) Example ideas for rephrasing the statement:
        - "the [Finish] property of [LL-Kw-7DWU] is [satin nickel]" --> "It has a beautiful satin nickel finish" OR "LL-Kw-7DWU is covered in a satin nickel finish"
        - "the [Usage Type] property of [HD-SM-FHKN] is [Desktop]" --> "Suitable for using on desktop" OR "HD-SM-FHKN is designed for desktop use"
        Very important: you are FORBIDDEN from copying the statement, you must rephrase the statement to a more natural form if writing or replacing sentences.
        Goal of step 1 is to NEVER copy the exact wording from the statement. That is strictly forbidden.
        Step 2: add the rephrased sentence/phrase from step 1 to the passage.
        Try to place the sentence where it fits most naturally in the passage (often in the middle, but sometimes at the beginning or end).
        If it is a phrase, you can insert it within an existing sentence in the passage.
    """,
}


