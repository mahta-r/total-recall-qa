INPUT_TEMPLATE = """
ENTITY: [{entity_name}]
PROPERTY: [{property_name}] is defined as {property_description}
STATEMENT: 
the [{property_name}] of [{entity_name}] is [{property_value}]
{time_of_statement}

PASSAGE:
title: {passage_title}
section: {sections}
text: {passage}
"""


REWRITE_PROMPT = """
Your task is to rewrite a single part of an input passage to include information about the value of a given property for an entity.
Other than information about value of property, the rest of the passage should remain unchanged, exactly same as it was given in input.
You are given the topic entity, the definiton of the property, and a statement about the value of that property for that entity. 
The statement can also potentially include the time of validness of that value.

You are also given the text of a passage about entity and the title of the document from which it was extracted. 
Optionally, the passage may include the sections it appeared in (in the document) to provide additional context.

{rewrite_explanation}

When adding the property value to the passage, do not include the brackets around property, entity, or value in the prompt.
Very important: Other than the value of the property (and time if needed), you must not change anything else in the passage. 
Ensure the rest of passage stays exactly the same, and output it as it was given in input, only changing/adding the necessary sentence about the property value.

Your output should be ONLY the rewritten passage text, without any additional explanations or comments.
Don't include the title or sections in your output.
You will only include the rewritten text of the passage, which is expected to be very similar to the input passage except for the property value.

"""


REWRITE_EXPLANATION = {
    "REPLACE": """
        The value of the property is mentioned in the given passage, but it does not match the value or time of validness given in the statement.
        If statement doesn't have time or passage doesn't mention time, you must only replace the value mentioned in the passage with the value given in the statement.
        If statement has the same time as the passage, you must replace the value mentioned in the passage with the value given in the statement.
        If statement has a different time than the passage, you must add a sentence to the passage that mentions the value of the property for the time given in statement.   
    """,
    "ADD": """
        The value of the property is not mentioned in the passage.
        You must add a sentence/phrase to the passage that mentions the value of the property for the entity, while ensuring the rest of passage stays exactly the same.
        You will do this in 2 steps. 
        Step 1: rephrase the statement to another form. (internal, DO NOT OUTPUT) Example ideas for rephrasing the statement:
        - "The [coordinate location] of [A] is [x]" --> "A is located at x" OR "A has a coordinates of x"
        - "The [shares border with] of [B] is [x]" --> "B shares its border with x" OR "x is a bordering neighbour of B"
        - "The [inception] of [C] is [x]" --> "C was founded in x" OR "C came into existence in x"
        - "The [population] of [D] is [x]" --> "D, with a population of x, ..." (added as a phrase in passage)
        If the statement includes time of validness, you must also include that time in the added sentence or phrase.
        Goal of step 1 is to NEVER copy the exact wording from the statement. That is strictly forbidden.
        Step 2: add the rephrased sentence/phrase from step 1 to the passage.
        Try to place the sentence in the middle of the passage if possible, and sometimes at the beginning or end if it makes most sense there.
        If it is a phrase, you can insert it within an existing sentence in the passage.
    """,
    # "ADD": """
    #     The value of the property is not mentioned in the passage.
    #     You must add a sentence/phrase to the passage that mentions the value of the property for the entity, while ensuring the rest of passage stays exactly the same.
    #     NEVER copy the exact wording from the statement. Be creative and form the sentence/phrase in different ways. Example ideas for rephrasing the statement:
    #     - "The [coordinate location] of [A] is [x]" --> "A is located at x" OR "A has a coordinates of x"
    #     - "The [shares border with] of [B] is [x]" --> "B shares its border with x" OR "x is a bordering neighbour of B"
    #     - "The [inception] of [C] is [x]" --> "C was founded in x" OR "C came into existence in x"
    #     - "The [population] of [D] is [x]" --> "D, with a population of x, ..." (added as a phrase in passage)
    #     If the statement includes time of validness, you must also include that time in the added sentence or phrase.
    #     Try to place the sentence in the middle of the passage if possible, and sometimes at the beginning or end if it makes most sense there.
    # """,
}


