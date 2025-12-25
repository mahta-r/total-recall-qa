QUERY_GENERATION_PROMPT = """
Your task is to generate total-recall queries based on inputs provided.
A total-recall query is a query that requires multiple documents to be retrieved and aggregated to answer correctly.
If any of the documents are missing, crucial information will be lost, and the query cannot be answered accurately.

The inputs provided for creating a total-recall query are:
- original-query: the original query from which entities are derived (use this to maintain context and relevance)
- class: wikidata class type used as set of entities
- attribute: a shared attribute of all entities of the class
- attribute-type: the type of the attribute (quantity, time, or entity-list)
- unit [optional]: the unit of the numerical value (e.g. foot, kelvin, dollar, .. only for numerical attributes)
- point-in-time [optional]: the time at which the values of the attribute are known to be true
- aggregation-operation: aggregation operation to be applied over the attribute values to answer the query
- entity-values: entities and the value of the attribute for each entity
- descriptions: descriptions for the entity class and attribute, explaining what they mean


You will create a natural language query that aggregates the given attribute values over all entities using given operation.
IMPORTANT: Your generated query MUST preserve ALL conditions, constraints, and contextual elements from the original query.
- If the original query contains temporal constraints (e.g., "before X", "after Y"), you MUST include them in your generated query.
- If the original query contains spatial constraints (e.g., "in X country"), you MUST include them.
- If the original query contains any other filters or conditions, you MUST include them.
- The entities in your query should be those referenced or derived from the original query.
- Simply add the new attribute condition to the existing constraints in the original query.

You should clearly specify the subject entity class such that entities can be inferred without individually naming them.
The query must be single-response: the answer cannot be multiple values or a list, it must be a single value.

## Attribute Type Handling:

### For QUANTITY attributes (numerical values):
- If there is a unit provided, you MUST specify it in your query
- The answer will be a numerical value
- Example: "what was the total wealth of members of X family in 2021 in million dollars"

### For TIME attributes (dates):
- ALWAYS ask about the YEAR, not the full date
- Questions should use terms like "year", "earliest year", "latest year", "average year"
- The answer will be a year (number)

### For ENTITY-LIST attributes (list of entities):
- The operation is COUNT - counting occurrences of a specific entity value in the list
- You should ask "how many" or "what is the count of" specific entities
- The answer will be a COUNT (number) representing how many entities have a specific value for this attribute
- IMPORTANT: When adding the attribute condition, preserve ALL conditions from the original query
- Example: Original: "How many X were there before Y?" → Generated: "How many X were there before Y with Z as their W?"
- Example: "What is the count of members in X family who were educated at University of Cambridge?"

If there is a point-in-time provided, you MUST specify it in your query. If no unit or point-in-time provided, don't mention them.

You are provided with descriptions for the entity class and attribute. You may use them to clarify the query only if the label alone is not clear enough;
For example when the name of the entity class alone is not descriptive enough to specify the entities.
Example: name="city" description="city in japan" -> use description to clarify
Example: attribute="water as percent of area" is clear, does not need description

Remember the query needs to be independently readable, so don't reference the given entities/attributes using pronouns in your query.
Do not explicitly name the entities unless absolutely necessary;
For example, you can say all nba teams, but don't state them one by one like LA Lakers, Boston Celtics, etc.
Never use any wikidata id (e.g. Q123456) in the query text.

The ANSWER must be an AGGREGATED VALUE (a number, year, or statistic), NOT an entity name or identifier.

Your output should be in a strictly specific format. Here is the format:
[Query] <Your generated query here>
[Aggregation] <The aggregation operation used>
[Answer] <Answer to your query. This is the result of the aggregation on entities.>
Do not add any extra text, or change the starting [Query] or [Aggregation] or [Answer] tags, or modify the format in any way.

{inputs}
"""
