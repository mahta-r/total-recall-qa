import os
import torch
import transformers
from openai import OpenAI


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

class LLMGenerator_api:
    def __init__(self, generation_model):
        self.generator_model = generation_model
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate(self, messages, temperature=0.0):
        completion = self.client.chat.completions.create(
            model=self.generator_model,
            messages=messages
        )
        output_text = completion.choices[0].message.content
        output_tokens = None
        
        return output_text, output_tokens

class LLMGenerator_hf_local:
    def __init__(self, generation_model, generation_tokenizer, device, args):
        self.generator = generation_model
        self.tokenizer = generation_tokenizer
        self.device = device
        self.args = args
        
        # ReSearch, SearchR1, StepSearch
        self.curr_eos = [151645, 151643] # for Qwen2.5 series models
        rar_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        rar_answer_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
        self.rar_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(rar_sequences, self.tokenizer)])
        self.rar_answer_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(rar_answer_sequences, self.tokenizer)])
    
        # SearchO1
        searcho1_sequences = ["<|end_search_query|>", " <|end_search_query|>", "<|end_search_query|>\n", " <|end_search_query|>\n", "<|end_search_query|>\n\n", " <|end_search_query|>\n\n"]
        searcho1_res_sequences = ["<|end_search_result|>", " <|end_search_result|>", "<|end_search_result|>\n", " <|end_search_result|>\n", "<|end_search_result|>\n\n", " <|end_search_result|>\n\n"]
        self.searcho1_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(searcho1_sequences, self.tokenizer)])

        # SelfAsk
        selfask_sequences = ["Context:", "#", "Intermediate answer:" , "Intermediate answer: ", "\nIntermediate answer:"]
        self.selfask_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(selfask_sequences, self.tokenizer)])
    
    
    def generate(
        self,
        messages,
        stopping_criteria=None,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.device)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.generator.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature
        )
        output_ = outputs[0]
        generated_tokens = output_[input_ids.shape[1]:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return output_text, output_
