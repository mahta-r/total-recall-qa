import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import re
import torch
import requests
import transformers
from bs4 import BeautifulSoup

from utils.general_utils import passages2string
from c2_model_generation.src.llm_generator import LLMGenerator_api, LLMGenerator_hf_local, StopOnSequence
from c2_model_generation.src.retrievers_local import BM25Retriever, RerankRetriever, DenseRetriever
from c2_model_generation.src.prompt_templetes import (
    SYSTEM_PROMPT_NO_RETRIEVAL,
    SYSTEM_PROMPT_SINGLE_RETRIEVAL,
    SYSTEM_PROMPT_RESEARCH_INST,
    PROMPT_SEARCHR1,
    PROMPT_STEPSEARCH,
    REACT_INSTRUCTION,
    SELF_ASK_PROMPT_MULTI_HOP,
    get_multiqa_search_o1_instruction,
    get_task_instruction_openqa,
    get_webpage_to_reasonchain_instruction
)

class BasicRAG:
    def __init__(self, device, args):
        self.args = args
        
        # --- Generators
        if args.model_source == 'api':
            self.generator = LLMGenerator_api(args.model_name_or_path)
        elif args.model_source == 'hf_local':
            backbone_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16).to(device) # attn_implementation="eager"
            backbone_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
            self.generator = LLMGenerator_hf_local(backbone_model, backbone_tokenizer, device, args)
        else:
            raise NotImplementedError
        
        # --- Retrievers 
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)  
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)

    # --- Information Extraction Functions
    def get_unique_docs(self, docs_lst:list):
        return list({doc['id']: doc for doc in docs_lst}.values()) 
    
    def get_think(self, text):
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

    def get_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

    def get_answer(self, text):
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

class NoRetrieval(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.user_prompt_template = "Question: {user_query}"
    
    def inference(self, question, generation_temp=0.7):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_RETRIEVAL},
            {"role": "user", "content": self.user_prompt_template.format(user_query=question)}
        ]
        output_text, _ = self.generator.generate(messages, temperature=generation_temp)
        reasoning = self.get_think(output_text)
        prediction = self.get_answer(output_text)
        reasoning_path = [{'think': reasoning, 'prediction': prediction}]
        
        return reasoning_path, prediction

class SingleRetrieval(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.user_prompt_template = "<information>{documents}</information>\n\nQuestion: {user_query}"
    
    def inference(self, question, generation_temp=0.7):
        search_docs = self.retriever.search(question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SINGLE_RETRIEVAL},
            {"role": "user", "content": self.user_prompt_template.format(
                documents=passages2string(search_docs),
                user_query=question
            )}
        ]
        output_text, _ = self.generator.generate(messages, temperature=generation_temp)
        reasoning = self.get_think(output_text)
        prediction = self.get_answer(output_text)
        reasoning_path = [
            {'think': '', 'search_query': '', 'docs': search_docs},
            {'think': reasoning, 'prediction': prediction}
        ]
        
        return reasoning_path, prediction

class ReSearch_Model(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.curr_step_template = '\n{output_text}<result>{search_results}</result>\n'
        self.answer_template = '<answer> \boxed{answer} </answer>'
    
    def get_query(self, text):
        pattern = re.compile(r"<search>\s*search query:\s*(.*?)\s*</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None
    
    def get_boxed_answer(self, text: str) -> str:
        match = re.search(r"\\boxed\{(.*?)\}", text)
        return match.group(1).strip() if match else None   
    
    def inference(self, question, generation_temp=0.7):
        input_prompt = question
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT_RESEARCH_INST},
            {'role': 'user', 'content': input_prompt}
        ]
        
        reasoning_path = []
        while True:
            output_text, output_ = self.generator.generate(
                messages,
                self.generator.rar_stopping_criteria,
                temperature=generation_temp
            )
            if output_[-1].item() in self.generator.curr_eos:
                break

            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                search_results = passages2string(search_docs)
            else:
                search_docs, search_results = [], ''
                
            reasoning_path.append({
                'think': self.get_think(output_text),
                'search_query': tmp_query,
                'docs': search_docs
            })
            search_text = self.curr_step_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT_RESEARCH_INST},
                {'role': 'user', 'content': input_prompt}
            ]
        
        one_step_think = self.get_think(output_text)
        prediction = self.get_boxed_answer(output_text)
        reasoning_path.append({'think': one_step_think, 'prediction': prediction})
            
        return reasoning_path, prediction

class SearchR1_Model(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.curr_step_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
        
    def inference(self, question, generation_temp=0.7):
        input_prompt = PROMPT_SEARCHR1.format(question=question)
        messages = [{"role": "user", "content": input_prompt}]
        
        reasoning_path = []
        while True:
            output_text, output_ = self.generator.generate(
                messages,
                self.generator.rar_stopping_criteria,
                temperature=generation_temp
            )
            if output_[-1].item() in self.generator.curr_eos:
                break
        
            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                search_results = passages2string(search_docs)
            else:
                search_docs, search_results = [], ''

            reasoning_path.append({
                'think': self.get_think(output_text),
                'search_query': tmp_query,
                'docs': search_docs
            })
            search_text = self.curr_step_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = [{"role": "user", "content": input_prompt}]

        one_step_think = self.get_think(output_text)
        prediction = self.get_answer(output_text)
        reasoning_path.append({'think': one_step_think, 'prediction': prediction})
            
        return reasoning_path, prediction

class StepSearch_Model(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.curr_step_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    
    def inference(self, question, generation_temp=0.7):
        input_prompt = PROMPT_STEPSEARCH.format(question=question)
        messages = [{"role": "user", "content": input_prompt}]
        
        reasoning_path = []
        while True:
            output_text, output_ = self.generator.generate(
                messages,
                self.generator.rar_stopping_criteria,
                temperature=generation_temp
            )
            if output_[-1].item() in self.generator.curr_eos:
                break
        
            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                search_results = passages2string(search_docs)
            else:
                search_docs, search_results = [], ''

            reasoning_path.append({
                'think': self.get_think(output_text),
                'search_query': tmp_query,
                'docs': search_docs
            })
            search_text = self.curr_step_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = [{"role": "user", "content": input_prompt}]

        one_step_think = self.get_think(output_text)
        prediction = self.get_answer(output_text)
        reasoning_path.append({'think': one_step_think, 'prediction': prediction})
            
        return reasoning_path, prediction

class ReAct_Model(BasicRAG):
    # Ref: https://github.com/ysymyth/ReAct/blob/master/hotpotqa.ipynb
    def __init__(self, device, args):
        super().__init__(device, args)
        self.instruction = REACT_INSTRUCTION
        self.answer_template = '{answer}'
        self.pattern_action = r"^(\w+)\[(.+?)\]$"
        self.current_step_template = 'Thought {iter_num}: {think}\nAction {iter_num}: {action_text}\nObservation {iter_num}: {observation}\n'
        
        self.examples = {
            "webthink_simple6": [
                {
                    "question": 'What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?',
                    "steps": [
                        {
                            "Thought": 'I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.',
                            "Action": 'Search[Colorado orogeny]',
                            "Observation": 'The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.'
                        },
                        {
                            "Thought": 'It does not mention the eastern sector. So I need to look up eastern sector.',
                            "Action": 'Lookup[eastern sector]',
                            "Observation": '(Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.',
                        },
                        {
                            "Thought": 'The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.',
                            "Action": 'Search[High Plains]',
                            "Observation": 'High Plains refers to one of two distinct land regions:',
                        },
                        {
                            "Thought": 'I need to instead search High Plains (United States).',
                            "Action": 'Search[High Plains (United States)]',
                            "Observation": 'The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).',
                        },
                        {
                            "Thought": 'High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.',
                            "Action": 'Finish[1,800 to 7,000 ft]',
                        }
                    ]
                },
                {
                    "question": 'Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?',
                    "steps": [
                        {
                            "Thought": 'The question simplifies to \"The Simpsons\" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.',
                            "Action": 'Search[Milhouse]',
                            "Observation": 'Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.',
                        },
                        {
                            "Thought": 'The paragraph does not tell who Milhouse is named after, maybe I can look up \"named after\".',
                            "Action": 'Lookup[named after]',
                            "Observation": '(Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.',
                        },
                        {
                            "Thought": 'Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.',
                            "Action": 'Finish[Richard Nixon]',
                        }
                    ]
                },
                {
                    "question": 'Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?',
                    "steps": [
                        {
                            "Thought": 'I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.',
                            "Action": 'Search[Adam Clayton Powell]',
                            "Observation": "Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].",
                        },
                        {
                            "Thought": 'To find the documentary, I can search Adam Clayton Powell (film).',
                            "Action": 'Search[Adam Clayton Powell (film)]',
                            "Observation": 'Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.',
                        },
                        {
                            "Thought": 'Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.',
                            "Action": 'Finish[The Saimaa Gesture]',
                        },
                    ]
                    
                },
                {
                    "question": 'What profession does Nicholas Ray and Elia Kazan have in common?',
                    "steps": [
                        {
                            "Thought": 'I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.',
                            "Action": 'Search[Nicholas Ray]',
                            "Observation": 'Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 \u2013 June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.',
                        },
                        {
                            "Thought": 'Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.',
                            "Action": 'Search[Elia Kazan]',
                            "Observation": 'Elia Kazan was an American film and theatre director, producer, screenwriter and actor.',
                        },
                        {
                            "Thought": 'Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.',
                            "Action": 'Finish[director, screenwriter, actor]',
                        },
                    ]
                },
                {
                    "question": "Which magazine was started first Arthur's Magazine or First for Women?",
                    "steps": [
                        {
                            "Thought": "I need to search Arthur's Magazine and First for Women, and find which was started first.",
                            "Action": "Search[Arthur's Magazine]",
                            "Observation": "Arthur's Magazine (1844-\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century.",
                        },
                        {
                            "Thought": "Arthur's Magazine was started in 1844. I need to search First for Women next.",
                            "Action": 'Search[First for Women]',
                            "Observation": "First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.",
                        },
                        {
                            "Thought": "First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.",
                            "Action": "Finish[Arthur's Magazine]",
                        },
                    ]
                },
                {
                    "question": 'Were Pavel Urysohn and Leonid Levin known for the same type of work?',
                    "steps": [
                        {
                            "Thought": 'I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.',
                            "Action": 'Search[Pavel Urysohn]',
                            "Observation": 'Pavel Samuilovich Urysohn (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.',
                        },
                        {
                            "Thought": 'Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.',
                            "Action": 'Search[Leonid Levin]',
                            "Observation": 'Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.',
                        },
                        {
                            "Thought": 'Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.',
                            "Action": 'Finish[yes]',
                        },
                    ]
                },
            ]
        }
        self.examples_text = ''
        for example in self.examples['webthink_simple6']:
            self.examples_text += f"Question: {example['question']}\n"
            for step_i, think_step in enumerate(example['steps']):
                for step_key, step_val in think_step.items():
                    self.examples_text += f"{step_key} {step_i+1}: {step_val}\n"
            self.examples_text += "\n"

    def generate_stopping_criteria(self, sequences):
        return transformers.StoppingCriteriaList([StopOnSequence(sequences, self.generator.tokenizer)])

    @staticmethod
    def clean_str(p):
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

    @staticmethod
    def get_page_obs(page):
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

    @staticmethod
    def clean_action(text):
        pattern = r'Observation \d+:\s*'
        match = re.search(pattern, text)
        return text[:match.start()].strip() if match else text.strip()

    @staticmethod
    def extract_action_text(text):
        pattern = r"\w+\[[^\]]*?\]"
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def extract_action_type_entity(self, action_text):
        match = re.match(self.pattern_action, action_text)
        if match:
            action_type = match.group(1).lower()
            action_entity = match.group(2)
            # print(f"Action Type: {action_type}, Action Entity: {action_entity}")
        else:
            # print("Action text does not match the expected template.")
            action_type, action_entity = '', ''
        
        return action_type, action_entity

    # For search action
    def wikipedia_search(self, entity):
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url).text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            result_titles = [self.clean_str(div.get_text().strip()) for div in result_divs]
            obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.search_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += self.clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        return obs

    def retriever_search(self, search_query):
        search_docs = self.retriever.search(search_query)
        self.page = ""
        for doc in search_docs:
            content = doc['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            self.page += text
        
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        
        return search_docs

    # For lookup action
    def construct_lookup_list(self, keyword):
        # find all paragraphs
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

    def get_observation(self, action_text):
        done, docs, obs = False, None, None
        
        action_type, action_entity = self.extract_action_type_entity(action_text)
        if action_type == 'search':
            docs = self.retriever_search(action_entity)
            obs = passages2string(docs)
        elif action_type =='lookup':
            keyword = action_entity
            if self.lookup_keyword != keyword:  # reset lookup
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                obs = "No more results.\n"
            else:
                obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        elif action_type =='finish':
            done = True
        
        elif action_type == 'think':
            obs = "Nice thought."
        else:
            obs = "Invalid action: {}".format(action_text)

        return action_type, action_entity, docs, obs, done

    def inference(self, question, generation_temp=0.7):
        self.page = None  # current Wikipedia page
        self.lookup_keyword = None  # current lookup keyword
        self.lookup_list = None  # list of paragraphs containing current lookup keyword
        self.lookup_cnt = None  # current lookup index
        search_query = question
        instruct_exps = self.instruction + self.examples_text 
        input_prompt = instruct_exps + f"Question: {search_query}\n"
        
        reasoning_path = []
        for iter_num in range(1, self.args.max_iter+1):
            messages = [
                {"role": "system", "content": ''},
                {"role": "user", "content": input_prompt + f"Thought {iter_num}:"}
            ]
            thought_action, _ = self.generator.generate(
                messages,
                self.generate_stopping_criteria([f"\n\nObservation {iter_num}:", f"\nObservation {iter_num}:", f"Observation {iter_num}:", f"Observation {iter_num}: ", f"\nObservation {iter_num}: "]),
                temperature=generation_temp
            )
            try:
                thought, action = thought_action.strip().split(f"Action {iter_num}:")
            except:
                # print('ohh...', thought_action)
                thought = thought_action.strip().split('\n')[0]
                messages = [
                    {"role": "system", "content": ''},
                    {"role": "user", "content": input_prompt + f"Thought {iter_num}: {thought}\nAction {iter_num}:"}
                ]
                action, _ = self.generator.generate(
                    messages,
                    self.generate_stopping_criteria(["]\n", "]\n\n", " ]\n", " ]\n\n"]),
                    temperature=generation_temp
                )
            
            thought = thought.replace('\n', ' ').strip()
            action_text = self.clean_action(action).replace('\n', ' ').strip()
            action_text = self.extract_action_text(action_text)
            
            if action_text: # if the action text is valid
                action_type, action_entity, docs, obs, done = self.get_observation(action_text)
                obs = obs.replace('\\n', ' ').strip() if obs else None
                
                if action_type == 'search':
                    reasoning_path.append({'think': thought, 'action_type': action_type, 'search_query': action_entity, 'docs': docs})
                elif action_type == 'lookup':
                    reasoning_path.append({'think': thought, 'action_type': action_type, 'entity': action_entity, 'observation': obs})
                elif action_type == 'finish':
                    reasoning_path.append({'think': thought, 'prediction': action_entity})
                    pred_answer = action_entity
                
                if done or (iter_num == self.args.max_iter):
                    break
                
                current_step_text = self.current_step_template.format(
                    iter_num=iter_num,
                    think=thought,
                    action_text=action_text,
                    observation=obs
                )
                input_prompt += current_step_text

        # Regenerate the last sentence if it is needed
        if action_type != 'finish':
            messages = [
                {"role": "system", "content": ''},
                {"role": "user", "content": input_prompt + f"Action {iter_num+1}: Finish["}
            ]
            output_text, _ = self.generator.generate(
                messages,
                self.generate_stopping_criteria(["]", "] ", ']\n', ']\n ', ']\n\n', ']\n\n ']),
                max_new_tokens=32
            )
            pred_answer = output_text[:-1]
            reasoning_path.append({'think': '', 'prediction': pred_answer})
        
        return reasoning_path, pred_answer

class SearchO1_Model(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        # Define special tokens
        self.BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
        self.END_SEARCH_QUERY = "<|end_search_query|>"
        self.BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
        self.END_SEARCH_RESULT = "<|end_search_result|>"
        self.MAX_SEARCH_LIMIT = 10
        self.with_reason_in_documents = True
        # 
        self.instruction = get_multiqa_search_o1_instruction(self.MAX_SEARCH_LIMIT)
        # 
        self.answer_template = '\\boxed{answer}'
        self.current_step_template = '\n{think}\n<|begin_search_query|>{search_query}<|end_search_query|>\n<|begin_search_result|>{search_result}<|end_search_result|>\n'

    def get_reasoning_think(self, text: str) -> str:
        pattern = re.compile(rf"(.*?){re.escape(self.BEGIN_SEARCH_QUERY)}", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

    def get_search_query(self, text: str) -> str:
        pattern = re.compile(rf"{re.escape(self.BEGIN_SEARCH_QUERY)}(.*?){re.escape(self.END_SEARCH_QUERY)}", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None
    
    def get_search_results(self, text: str) -> str:
        match = re.search(r"\*\*Final Information\*\*\s*(.*)", text, re.DOTALL)
        return match.group(1).replace("\n", " ").strip() if match else None
    
    def get_last_think(self, text: str) -> str:
        match = re.search(r"^(.*?)\\boxed\{.*?\}", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def get_boxed_answer(self, text: str) -> str:
        match = re.search(r"\\boxed\{(.*?)\}", text)
        return match.group(1).strip() if match else None

    def reason_in_documents(self, path, search_query, docs_text):
        prev_reasoning = ' '.join([step['think'] for step in path])
        rid_input_prompt = get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, docs_text)
        rid_messages = [{"role": "user", "content": rid_input_prompt}]
        rid_output_text, _ = self.generator.generate(rid_messages)
        rid_output_text_ = self.get_search_results(rid_output_text)
        return rid_output_text_

    def inference(self, question, generation_temp=0.7):
        input_prompt = self.instruction + get_task_instruction_openqa(question)
        messages = [{"role": "user", "content": input_prompt}]
        
        reasoning_path = []
        for idx in range(self.MAX_SEARCH_LIMIT):
            # -- One step generation
            output_text, output = self.generator.generate(
                messages,
                self.generator.searcho1_stopping_criteria,
                temperature=generation_temp
            )
            
            if (output[-1].item() in self.generator.curr_eos) or (idx+1 == self.MAX_SEARCH_LIMIT):
                break # Don't perform another retrieval or prompt construction
            
            # -- Extract info
            tmp_think = self.get_reasoning_think(output_text).replace("\n", ' ').replace("\n\n", ' ') if self.get_reasoning_think(output_text) else ''
            tmp_query = self.get_search_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                docs_text = passages2string(search_docs)
            else:
                search_docs, docs_text = [], ''

            # -- Save in the path
            reasoning_path.append({'think': tmp_think, 'search_query': tmp_query, 'docs': search_docs})
            
            # -- Reason-in-docs
            if self.with_reason_in_documents:
                rid_output_text = self.reason_in_documents(reasoning_path, tmp_query, docs_text)
                reasoning_path[-1]['reason_in_docs'] = rid_output_text
                search_result_txt = rid_output_text
            else:
                search_result_txt = docs_text
            
            # -- Create new input prompt
            current_step_text = self.current_step_template.format(
                think=tmp_think,
                search_query=tmp_query,
                search_result=search_result_txt
            )
            input_prompt += current_step_text
            messages = [{"role": "user", "content": input_prompt}]

        last_think = self.get_last_think(output_text) if self.get_last_think(output_text) else output_text
        pred_answer = self.get_boxed_answer(output_text)
        reasoning_path.append({'think': last_think, 'prediction': pred_answer})

        return reasoning_path, pred_answer

class SelfAsk_Model(BasicRAG):
    def __init__(self, device, args):
        super().__init__(device, args)
        self.single_hop = False
        self.system_prompt = SELF_ASK_PROMPT_MULTI_HOP
        self.user_prompt = "{documents}Quesiton: {question}\nAre follow up questions needed here: Yes.\n"
        self.FOLLOW_UP_PATTERN = r"Follow up:.*\n"
    
    def documents2string(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Context{idx+1}: {text}\n"

        return format_reference
    
    def extract_follow_up(self, text: str) -> str:
        match = re.search(r'Follow up:\s*(.*?)\nIntermediate answer:', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
        
    def extract_intermediate(self, text: str) -> str:
        match = re.search(r'(.*?)(?:Follow up:|So the final answer is:)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_final_answer(self, text: str) -> str:
        parts = text.split("So the final answer is: ", 1)  # Split at the first occurrence
        if len(parts) <= 1:
            return None
        pred = parts[1].strip() if len(parts) > 1 else ""
        pattern = r"\.?</s>"
        pred = re.sub(pattern, "", pred)
        pred = pred.rstrip(".?!")
        return pred

    def inference(self, question, generation_temp=0.7):
        reasoning_path, text = [], ""
        
        # Initial retrieval
        search_query = question
        cur_search_docs = self.retriever.search(search_query)
        user_input_prompt = self.user_prompt.format(
            documents = self.documents2string(cur_search_docs),
            question=question
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input_prompt}
        ]
        reasoning_path.append({'think': '', 'search_query': search_query, 'docs': cur_search_docs})

        for idx in range(self.args.max_iter):
            output_text, output = self.generator.generate(
                messages,
                self.generator.selfask_stopping_criteria,
                temperature=generation_temp
            )
            
            if ("So the final answer is:" in output_text):
                text += output_text
                break
            if (output[-1].item() in self.generator.curr_eos) or (idx+1 == self.args.max_iter):
                break # Don't perform another retrieval or prompt construction
        
            intermediate_ans = self.extract_intermediate(output_text)
            search_query = self.extract_follow_up(output_text)
            cur_search_docs = self.retriever.search(search_query) if search_query else []
            tmp_docs = [doc for step in reasoning_path for doc in step['docs']] + cur_search_docs
            unq_tmp_doc = self.get_unique_docs(tmp_docs)
            
            reasoning_path.append({
                'think': intermediate_ans,
                'search_query': search_query,
                'docs': cur_search_docs
            })
            
            if idx == 0:
                text += f"Follow up: {search_query}\nIntermediate answer: "
            else:
                text += f"{intermediate_ans}\nFollow up: {search_query}\nIntermediate answer: "
            
            user_input_prompt = self.user_prompt.format(
                documents = self.documents2string(unq_tmp_doc),
                question=question
            ) + text
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input_prompt}
            ]
        
        # Regenerate the last sentence if it is needed
        if "So the final answer is:" not in output_text:
            text += f"{output_text}.\nSo the final answer is: "
            
            tmp_docs = [doc for step in reasoning_path for doc in step['docs']]
            unq_tmp_doc = self.get_unique_docs(tmp_docs)
            user_input_prompt = self.user_prompt.format(
                documents = self.documents2string(unq_tmp_doc),
                question=question
            ) + text
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input_prompt}
            ]
            output_text, _ = self.generator.generate(messages)
            pred_answer = self.extract_final_answer(output_text) if self.extract_final_answer(output_text) else output_text
            reasoning_path.append({'think': output_text, 'answer': pred_answer})
            
        else:
            intermediate_ans = self.extract_intermediate(output_text)
            pred_answer = self.extract_final_answer(output_text)
            reasoning_path.append({'think': intermediate_ans, 'prediction': pred_answer})

        return reasoning_path, pred_answer

