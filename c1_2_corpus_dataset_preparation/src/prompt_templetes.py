
SYSTEM_PROMPT_SPARQL_LIST = """You are an expert in SPARQL and semantic web data. 
You are tasked to rewrite the SPARQL query so that instead of returning a boolean or an aggregate value, 
it returns the list of items that contribute to that result.

The input SPARQL query may:
- Contain multiple PREFIX declarations.
- Use SELECT, SELECT DISTINCT, or ASK forms.
- Contain subqueries like SELECT (COUNT(...)) inside other SELECT or ASK blocks.
- Use aggregates like COUNT(?x), COUNT(DISTINCT ?x), AVG(...), etc.
- Have arbitrary spacing or capitalization.

Your goal:
1. Identify the variable being counted or aggregated (e.g., ?writer in COUNT(DISTINCT ?writer)).
2. Produce a clean query that **lists those distinct items** instead of counting them.
   - Use `SELECT DISTINCT ?var ?varLabel`.
3. Keep all relevant PREFIX declarations intact.
4. Keep the WHERE patterns that determine which entities are counted.
5. Add the following service if not already present:
   `SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }`
6. Ignore outer aggregations like AVG or ASK and focus on the innermost counted variable.
7. If the query cannot be recognized as a count-type query, return exactly:
   `the pattern is not recognized`.

Format the result as valid SPARQL.

Example:
Input:
SELECT (COUNT(DISTINCT ?writer) AS ?result) WHERE { ?writer wdt:P27 wd:Q17; wdt:P106 wd:Q36180. }

Output:
SELECT DISTINCT ?writer ?writerLabel
WHERE {
  ?writer wdt:P27 wd:Q17;
          wdt:P106 wd:Q36180.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
} 

Return your answer as a JSON object with a single field "converted_query".

"""


SUBQUERY_GENERATOR_PROMPT = """You are an expert in query generation.
Your task is to generate a verification-style question based on: A main user query, and One of its answer entities.
The goal is to produce a natural yes/no-style question that checks whether the given entity is a valid member of the ground-truth answer set implied by the main query.
In other words, rewrite the main query to focus on the given entity, preserving the meaning while turning it into a specific verification question.

Instructions: The generated question must be grammatically correct and natural.
It should follow the semantic intent of the main user question.
It must be answerable with “yes” or “no.”
Avoid adding extra details not implied by the main query.

Examples:
Main User Question: How many businesses did Steve Jobs found?
One of the Answers: NeXT
Generated Question: Is NeXT a business founded by Steve Jobs?

Main User Question: Do all of Batman’s partners speak English as their native language?
One of the Answers: Talia al Ghul
Generated Question: Does Talia al Ghul, one of Batman’s partners, speak English as her native language?

Now, generate the question in the same style for the given input, without providing any reasoning:
Main User Question: {main_query}
One of the Answers: {entity}
Generated Question: 
"""


RELEVANCE_JUDGMENT_PROMPT = """Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query,
1 = represents that the passage seems related to the query but does not answer it,
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely,
category 2 if passage presents something very important related to the entire topic but also has some extra information and
category 3 if the passage only and entirely refers to the topic.
If none of the above satisfies give it category 0.

Query: {query}
Passage: {passage}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. 
Provide each score in the format of: ##final score: score without providing any reasoning."""


