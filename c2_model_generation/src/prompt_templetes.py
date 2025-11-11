

SYSTEM_PROMPT_NO_RETRIEVAL = 'Answer the given question. You must conduct reasoning inside <think> and </think>. Then you must provide the answer inside <answer> and </answer>. Provide only SHORT form answers, NOT complete sentence, without any additional text or explanation.'
SYSTEM_PROMPT_SINGLE_RETRIEVAL = 'Answer the given question. The retrieved information is inserted into <information> </information>. You must conduct reasoning inside <think> and </think>. Then you must provide the answer inside <answer> and </answer>. Provide only SHORT form answers, NOT complete sentence, without any additional text or explanation.'


SYSTEM_PROMPT_RESEARCH_BASE = """A conversation between User and Assistant.
The user asks a question, and the assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
During thinking, the assistant can invoke the wikipedia search tool to search for fact information about specific topics if needed.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively,
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively.
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result>
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>.
In the last part of the answer, the final exact answer is enclosed within \\boxed{{}} with latex format.
User: {prompt}. Assistant:"""

SYSTEM_PROMPT_RESEARCH_INST = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool.
Given a question, you need to first think about the reasoning process in the mind and then provide the answer.
During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively,
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively.
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result>
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>.
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""

PROMPT_SEARCHR1 = """Answer the given question.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

PROMPT_STEPSEARCH = """## Background
You are a deep AI research assistant. I will give you a single-hop or multi-hop question.
You don't have to answer the question now, but you should first think about your research plan or what to search for next.
You can use search to fill in knowledge gaps.
## Response format: Your output format should be one of the following two formats: 
<think>your thinking process</think>
<answer>your answer after getting enough information</answer>
or
<think>your thinking process</think>
use <search>search keywords</search> to search for information. For example, <think> plan to search: (Q1) (Q2) (Q3) ... </think> <search> (Q1) question </search> <think> reasoning ... </think> <answer> Beijing </answer>.
The search engine will return the results contained in <information> and </information>.
Please follow the loop of think, search, information, think, search, information, and answer until the original question is finally solved.
Note: The retrieval results may not contain the answer or contain noise.
You need to tell whether there is a golden answer. If not, you need to correct the search query and search again. Question:{question}
"""

REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.\n"""

SELF_ASK_PROMPT_MULTI_HOP = """Given the following question, answer it by providing follow up questions and intermediate answers. If intermediate questions are not necessarry, answer the question directly. You are provided with evidence that can help you arrive at the answer before the question.
After "So the final answer is:", please provide the final answer in short form only — not a full sentence — and do not include any extra text or explanation.
#
Context1: Xawery Żuławski: Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska. So the answer is Xawery Żuławski.
Context2: Xawery Żuławski: Xawery Żuławski ; National Film School in Łódź · 1995–present · Maria Strzelecka · 2.
Question: Who is the mother of the director of film Polish-Russian War (Film)?
Are follow up questions needed here: Yes.
Follow up: Who is the director of the film Polish-Russian War (Film)?
Intermediate answer: The director of the film Polish-Russian War is Xawery Żuławski.
Follow up: Who is the mother of Xawery Żuławski?
Intermediate answer: The mother of Xawery Żuławski is Małgorzata Braunek.
So the final answer is: Rick Scott Małgorzata Braunek.
#
Context1: 2003: Blind Shaft (Chinese: 盲井; pinyin: Mángjǐng) is a 2003 film about a pair of brutal con artists operating in the illegal coal mines of present-day northern China. So the answer is 2003.
Context2: December 2, 1932: Release and reception. The Mask of Fu Manchu opened in New York on December 2, 1932. The film cost a total of $338,000 and had worldwide rentals of $625,000. It had a profit of $62,000. So the answer is December 2, 1932.
Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Are follow up questions needed here: Yes.
Follow up: When did Blind Shaft come out?
Intermediate answer: Blind Shaft came out in 2003.
Follow up: When did The Mask Of Fu Manchu come out?
Intermediate answer: The Mask Of Fu Manchu came out in 1932.
So the final answer is: The Mask Of Fu Manchu.
#
Context1: John V, Prince of Anhalt-Zerbst: John was the second (but eldest surviving) son of Ernest I, Prince of Anhalt-Dessau, by his wife Margarete, daughter of Henry I, Duke of Münsterberg-Oels, and granddaughter of George of Poděbrady, King of Bohemia.
Context2: 12 June 1516: Ernest I, Prince of Anhalt-Dessau (died Dessau, 12 June 1516), was a German prince of the House of Ascania and ruler of the principality of Anhalt-Dessau. So the answer is 12 June 1516.
Question: When did John V, Prince Of Anhalt-Zerbst's father die?
Are follow up questions needed here: Yes.
Follow up: Who is the father of John V, Prince Of Anhalt-Zerbst?
Intermediate answer: The father of John V, Prince Of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau.
Follow up: When did Ernest I, Prince of Anhalt-Dessau die?
Intermediate answer: Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.
So the final answer is: 12 June 1516
#
Context1: El extraño viaje: El extraño viaje (English: The Strange Voyage) is a 1964 Spanish black drama film directed by Fernando Fernán Gómez.
Context2: Love in Pawn: Love in Pawn is a 1953 British comedy film directed by Charles Saunders and starring Bernard Braden, Barbara Kelly and Jeannie Carson.
Context3: 28 August 1921: Fernando Fernández Gómez (28 August 1921 – 21 November 2007) better known as Fernando Fernán Gómez was a Spanish actor, screenwriter, film director, theater director and member of the Royal Spanish Academy for seven years. So the answer is 28 August 1921.
Context4: Charles Saunders (director): Charles Joel Saunders (8 April 1904 – 20 April 1997) was an English film director and screenwriter who began in the industry as a film editor, and who also contributed to television.
Question: Which film has the director who was born later, El Extraño Viaje or Love In Pawn?
Are follow up questions needed here: Yes.
Follow up: Who is the director of El Extraño Viaje?
Intermediate answer: The director of El Extraño Viaje is Fernando Fernán Gómez.
Follow up: Who is the director of Love in Pawn?
Intermediate answer: The director of Love in Pawn is Charles Saunders.
Follow up: When was Fernando Fernán Gómez born?
Intermediate answer: Fernando Fernán Gómez was born on 28 August 1921.
Follow up: When was Charles Saunders (director) born?
Intermediate answer: Charles Saunders was born on 8 April 1904.
So the final answer is: El Extraño Viaje.
#
Context1: John, Count Palatine of Neumarkt: John (Johann von Pfalz-Neumarkt; 1383 – 14 March 1443) was the Count Palatine of Neumarkt from 1410 to his death. The son of Rupert III of the Palatinate, he married Catherine of Pomerania in 1407.
Context2: John, Count Palatine of Neumarkt: John (Johann von Pfalz-Neumarkt; 1383 – 14 March 1443) was the Count Palatine of Neumarkt from 1410 to his death. The son of Rupert III of the Palatinate, he married Catherine of Pomerania in 1407.
Question: Who is Catherine Of Pomerania, Countess Palatine Of Neumarkt's father-in-law?
Are follow up questions needed here: Yes.
Follow up: Who is the husband of Catherine of Pomerania, Countess Palatine of Neumarkt?
Intermediate answer: The husband of Catherine of Pomerania, Countess Palatine of Neumarkt is John, Count Palatine of Neumarkt.
Follow up: Who is the father of John, Count Palatine of Neumarkt?
Intermediate answer: The father of John, Count Palatine of Neumarkt is Rupert III of the Palatinate.
So the final answer is: Rupert III of the Palatinate.
#
Context1: Crimen a las tres: Crimen a las tres is a 1935 Argentine crime film directed and written by Luis Saslavsky. Crimen a las tres. Directed by, Luis Saslavsky.
Context2: Elio Petri: The Working Class Goes to Heaven (Italian: La classe operaia va in paradiso), released in the US as Lulu the Tool, is a 1971 political drama film directed by Elio Petri. So the answer is Elio Petri.
Context3: March 20, 1995: Luis Saslavsky (April 21, 1903 – March 20, 1995) was an Argentine film director, screenwriter and film producer, and one of the influential directors in the Cinema of Argentina of the classic era. So the answer is March 20, 1995.
Context4: Elio Petri: Final years. In 1981, Petri visited Geneva to direct Arthur Miller\'s new play The American Clock, with Marcello Mastroianni playing the lead role. Petri died of cancer on 10 November 1982. He was 53 years old.
Question: Which film has the director died first, Crimen A Las Tres or The Working Class Goes To Heaven?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Crimen a las tres?
Intermediate answer: The director of Crimen a las tres is Luis Saslavsky.
Follow up: Who is the director of The Working Class Goes to Heaven?
Intermediate answer: The director of The Working Class Goes to Heaven is Elio Petri.
Follow up: When did Luis Saslavsky die?
Intermediate answer: Luis Saslavsky died on March 20, 1995.
Follow up: When did Elio Petri die?
Intermediate answer: Elio Petri died on 10 November 1982.
So the final answer is: The Working Class Goes to Heaven
#"""

def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n'
        )
    return user_prompt

def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""

