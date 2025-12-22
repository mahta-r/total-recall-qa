import os
import json
import tqdm
import argparse
from datasets import load_dataset
from openai import OpenAI

from io_utils import read_jsonl_from_file, read_json_from_file, read_text_from_file, write_json_to_file
from prompts.LLM_as_relevance_judge.prompt_property_check import PROPERTY_PROMPT, PROPERTY_INPUT_TEMPLATE



def main(args):

    print("Loading generations...")
    generations = read_jsonl_from_file(args.generations_path)
    print("Loading classes...")
    classes = read_json_from_file(args.query_candidates_path)
    print("Loading page2passage2idx...")
    page2passage2idx = read_json_from_file(args.page2passage2idx_path)
    print("Loading corpus...")
    corpus = load_dataset(args.corpus_path, split="train")


    api_key = os.getenv("OPENAI_API_KEY")
    property_client = OpenAI(api_key=api_key)


    with open(args.log_file, 'w', encoding='utf-8') as log:

        for idx, gen in tqdm.tqdm(enumerate(reversed(generations))):

            subclass = classes[gen['subclass_id']]
            class_property_record = gen['property']
            instances = subclass['instances']
            assert all("wiki_page_id" in inst for inst in instances)

            prop = class_property_record["property_info"]
            entity_values = class_property_record["entities_values"]
            point_in_time = class_property_record.get("point_in_time", None)    
            unit_id, unit_label = class_property_record.get("unit", (None, None))

            for inst, record in zip(
                sorted(instances, key=lambda x: x["id"]), 
                sorted(entity_values, key=lambda x: x["entity_id"])
            ):
                assert inst["id"] == record["entity_id"]
                entity_label = record["entity_label"]
                property_value = record["value"]

                wiki_page_id = inst["wiki_page_id"]
                if wiki_page_id not in page2passage2idx:
                    # TODO: fix mismatch in page_ids
                    continue
                
                passage2idx = page2passage2idx[wiki_page_id]
                
                print(f"---------- {entity_label}:[{prop['label']}] ----------\n", file=log)

                # YES/NO answers from LLM
                is_property_in_passage = {
                    "YES": [],
                    "NO": []
                }
                for chunk_id, passage_idx in sorted(passage2idx.items(), key=lambda x: x[0]):
                    passage = corpus[passage_idx]
                    assert passage['id'] == f"{wiki_page_id}-{chunk_id}"
                    passage_content = passage['contents']

                    input_instance = PROPERTY_INPUT_TEMPLATE.format(
                        entity_name=entity_label,
                        property_name=prop['label'],
                        property_description=prop['description'],
                        passage=passage_content
                    )

                    resp = property_client.chat.completions.create(
                        model=args.model, 
                        messages=[
                            {"role": "system", "content": PROPERTY_PROMPT},
                            {"role": "user", "content": input_instance}
                        ],
                        temperature=args.temperature,
                    )

                    response = resp.choices[0].message.content.strip()
                    assert response in ["YES", "NO"]
                    is_property_in_passage[response].append(passage['id'])

                    print("PROMPT:\n" + input_instance, file=log)
                    print("RESPONSE:\n" + response, file=log)
                    print("- - - - - -", file=log)
                    
                relevant_passages = is_property_in_passage["YES"]
                if len(relevant_passages) > 0:
                    # TODO: rewrite passages with wikidata property value if mismatch
                    pass # waiting on final corpus version
                else:
                    # TODO: add passage containing wikidata property value
                    pass # waiting on final corpus version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_candidates_path", 
        type=str, 
        required=True, 
        help="Path to wikidata query candidates used for generations"
    )
    parser.add_argument(
        "--generations_path", 
        type=str, 
        required=True, 
        help="Path to JSON generations file, same as --output_file in generate_queries.py"
    )
    parser.add_argument(
        "--log_file", 
        type=str, 
        required=True, 
        help="Log file location"
    )
    parser.add_argument(
        "--corpus_path", 
        type=str, 
        required=True, 
        help="Path to corpus on huggingface hub or local directory"
    )
    parser.add_argument(
        "--page2passage2idx_path", 
        type=str, 
        required=True, 
        help="Path to JSON file containing wikipageid->chunkid->corpusidx mapping" \
        "This needs a pre-iteration over corpus to allow look-ups by wiki_page_id"
    )
    parser.add_argument(
        "--model",
        type=str, 
        required=True, 
        help="Model to use for generation e.g gpt-4o, o3-mini, etc."
    )    
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")


    args = parser.parse_args()
    main(args) 

