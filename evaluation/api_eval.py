import os
import argparse
import time
import re
from datetime import datetime
from openai import OpenAI
import tqdm

from io_utils import read_jsonl_from_file, write_jsonl_to_file


SYSTEM_PROMPT_NO_RETRIEVAL = 'Answer the given question. You must conduct reasoning inside <think> and </think>. Then you must provide the answer inside <answer> and </answer>. Provide only a SINGLE numerical answer, NOT complete sentence, without any additional text or explanation.'
USER_PROMPT_TEMPLATE = 'Question: {user_query}'


def parse_numeric_value(text):
    """
    Extract and normalize a single numeric value from text.
    Handles commas or dots as decimal/thousand separators.
    Returns float or None.
    """
    if not text:
        return None
    # Remove spaces, non-breaking spaces, and units like '%' or 'km'
    cleaned = re.sub(r"[^\d,.\-]", "", text.replace(" ", "").replace(" ", ""))
    # Case 1: European-style numbers ("1.234,56")
    if re.match(r"^\d{1,3}(\.\d{3})*,\d+$", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    # Case 2: US-style numbers ("1,234.56")
    elif re.match(r"^\d{1,3}(,\d{3})*\.\d+$", cleaned):
        cleaned = cleaned.replace(",", "")
    # Case 3: Ambiguous comma decimal ("2,7")
    elif cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def get_think(text):
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

def get_answer(text):
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[0] if matches else None


def run_eval(args):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    queries = read_jsonl_from_file(args.queries_path)

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = os.path.join(args.out_dir, f"{args.model}_answers_{timestamp}.jsonl")
    logfile = os.path.join(args.log_dir, f"{args.model}_eval_{timestamp}.log")

    with open(logfile, 'w', encoding='utf-8') as log:

        results = []
        for query in tqdm.tqdm(queries):

            qid = query['id']
            question = query['question']
            gold_answer = query['answer']['value']

            try:
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_NO_RETRIEVAL},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(user_query=question)}
                    ],
                    temperature=args.temperature,
                )

                output_text = completion.choices[0].message.content
                reasoning = get_think(output_text)
                prediction = get_answer(output_text)
                predicted_answer = parse_numeric_value(prediction)

                print(f"Query: {qid} | {question}", file=log)
                print(f"Gold Answer: {gold_answer}", file=log)
                print(f"Prediction Raw: {prediction}", file=log)
                print(f"Predicted Answer: {predicted_answer}", file=log)
                print(f"Reasoning: {reasoning}", file=log)
                # print(f"Completion: {output_text}", file=log)
                print("\n---------------------------------------------------------------------------------------------------\n", file=log)

                results.append({
                    "qid": qid,
                    "query": question,
                    "generation": output_text,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                })

            except Exception as e:
                print(f"Error processing query ID {qid}: {e}")
                time.sleep(0.2)  # light throttle for rate limits
    
    write_jsonl_to_file(outfile, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_path", type=str, required=True, help="Path to generated queries, same as --queries_file in generate_queries.py")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save evaluation outputs")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to save evaluation logs")
    parser.add_argument("--model", type=str, required=True, help="Model to use for evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for answer generation")

    args = parser.parse_args()
    run_eval(args)
