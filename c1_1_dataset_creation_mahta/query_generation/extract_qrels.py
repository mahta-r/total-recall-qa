import os
import json
import re
import copy
import logging
import sys
import tqdm
import argparse

from collections import defaultdict
from datetime import date
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI

from io_utils import read_jsonl_from_file, read_json_from_file, write_json_to_file, encode_datetime
from wikidata.data_utils import format_value, format_time_for_prompt

from query_generation.prompts.LLM_as_relevance_judge.prompt_rewrite import (
    REWRITE_PROMPT,
    REWRITE_EXPLANATION
)
from query_generation.prompts.LLM_as_relevance_judge.prompt_combined import (
    PROPERTY_CHECK_PROMPT, 
    PROPERTY_CHECK_INPUT_TEMPLATE,
    DATATYPE_EXPLANATION
)

main_logger = logging.getLogger(__name__)


def load_qrel_results_if_exists(results_path):
    results = {}
    
    if Path(results_path).exists():
        main_logger.info(f"Loading existing qrel results from {results_path}...")
        
        result_list = read_jsonl_from_file(results_path)
        for result_line in result_list:
            candidate_class_id = result_line['candidate_class_id']
            property_id = result_line['property_id']
            instance_id = result_line['instance_id']
            if candidate_class_id not in results:
                results[candidate_class_id] = {}
            if property_id not in results[candidate_class_id]:
                results[candidate_class_id][property_id] = {}
            results[candidate_class_id][property_id][instance_id] = result_line['qrel_result']
        
        main_logger.info(f"Loaded qrel results for {len(results)} candidates.")
    
    return results


def append_record_to_qrel_results_file(results_path, candidate_class_id, property_id, instance_id, qrel_result):
    with open(results_path, 'a', encoding='utf-8') as f:
        record = {
            "candidate_class_id": candidate_class_id,
            "property_id": property_id,
            "instance_id": instance_id,
            "qrel_result": qrel_result
        }
        f.write(json.dumps(record, ensure_ascii=False, default=encode_datetime) + '\n')


def load_existing_rewrites_if_exists(rewrites_path):
    rewrites = defaultdict(list)
    
    if Path(rewrites_path).exists():
        main_logger.info(f"Loading existing passage rewrites from {rewrites_path}...")
        
        rewrite_list = read_jsonl_from_file(rewrites_path)
        for rewrite_record in rewrite_list:
            passage_id = rewrite_record["passage"]['id']
            rewrites[passage_id].append(rewrite_record)
        
        main_logger.info(f"Loaded rewrites for {len(rewrites)} passages.")
        
    
    return rewrites


def append_passages_to_rewrites_file(rewrites_path, rewrites):
    with open(rewrites_path, 'a', encoding='utf-8') as f:
        for rewrite_record in rewrites:
            f.write(json.dumps(rewrite_record, ensure_ascii=False, default=encode_datetime) + '\n')


class QrelGenerator:
    def __init__(self, args):

        main_logger.info("Loading corpus...") 

        self.corpus = load_dataset(args.corpus_path, split="train")
        self.page2chunk2idx = create_random_access_mapping(self.corpus, args.page2chunk2idx_path)
        
        self.property_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.property_model = args.judge_model
        self.property_temperature = args.judge_temperature

        self.rewrite_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rewrite_model = args.rewrite_model
        self.rewrite_temperature = args.rewrite_temperature

        self.results_path = args.qrel_results_path
        self.results = load_qrel_results_if_exists(args.qrel_results_path)

        self.passage_rewrites_path = args.passage_rewrites_path
        self.rewrites = load_existing_rewrites_if_exists(args.passage_rewrites_path)


        self.total_llm_calls = 0
        self.num_queries_missing_wikipages = 0
        self.missing_wikipage_ids = set()
        self.total_passages_processed = 0
        self.num_yes_same = 0
        self.num_yes_different = 0
        self.num_entities_yes_same = 0
        self.num_entities_yes_different = 0
        self.num_entities_no = 0



    def get_query_relevance_judgements(self, instances, property, candidate_id, candidate_logger):
        
        missing_wikipage_ids = []
        assert all("wikipage_id" in inst for inst in instances)
        for inst in instances:
            wikipage_id = str(inst["wikipage_id"])
            if wikipage_id not in self.page2chunk2idx:
                missing_wikipage_ids.append(wikipage_id)
                main_logger.warning(f"{wikipage_id} ({inst['label']}, {inst['wikipedia']}) not found in corpus mapping.")
                candidate_logger.warning(f"{wikipage_id} ({inst['label']}, {inst['wikipedia']}) not found in corpus mapping.")
        if missing_wikipage_ids:
            self.missing_wikipage_ids.update(missing_wikipage_ids)
            self.num_queries_missing_wikipages += 1
            return
            # return 0, missing_wikipage_ids 

        candidate_num, aggregation_class_id, aggregation_prop_id = candidate_id.split('_')

        if aggregation_class_id not in self.results:
            self.results[aggregation_class_id] = {}
        if property['id'] not in self.results[aggregation_class_id]:
            self.results[aggregation_class_id][property['id']] = {}

        total_llm_calls_for_this_property = 0

        for idx, instance in enumerate(instances):
            
            main_logger.info(f"---> Instance {idx+1}/{len(instances)}: {instance['label']} | Property: {property['label']}")
            
            candidate_logger.info(f"--------------------------------------------------------------------------------")
            candidate_logger.info(f"---> Instance {idx+1}/{len(instances)}: {instance['label']} | Property: {property['label']}")
            candidate_logger.info(f"--------------------------------------------------------------------------------")

            if instance["id"] in self.results[aggregation_class_id][property['id']]:
                main_logger.info(f"Already processed instance {idx+1}, skipping.")
                continue # already processed

            for ev in property["list_of_entity_values"]:
                if instance["id"] == ev["entity_id"]:
                    entity_value = ev
                    break
            assert instance["label"] == entity_value["entity_label"]

            wikipage_id = str(instance["wikipage_id"])
            chunk2idx = self.page2chunk2idx[wikipage_id]
            
            # total_judgments_for_this_property += len(chunk2idx)

            label_to_passage = defaultdict(list) # YES/NO answers from LLM
            for chunk_id, passage_idx in sorted(chunk2idx.items(), key=lambda x: x[0]):
                
                passage_id = f"{wikipage_id}-{chunk_id}"
                passage = self.get_latest_passage_version(passage_id, passage_idx)
                # passage = self.corpus[passage_idx]
                # assert passage['id'] == passage_id
                
                total_llm_calls_for_this_property += 1
                response = self.run_llm_as_judge(
                    entity_value, 
                    property, 
                    passage,
                    candidate_logger
                )

                label_to_passage[response].append((passage['id'], passage_idx))

            for label, passage_ids in label_to_passage.items():
                main_logger.info(f"  {label}: {len(passage_ids)} passages")

            self.total_passages_processed += len(chunk2idx)
            self.num_yes_same += len(label_to_passage['YES-SAME'])
            self.num_yes_different += len(label_to_passage['YES-DIFFERENT'])
            self.num_entities_yes_same += 1 if len(label_to_passage['YES-SAME']) > 0 else 0
            self.num_entities_yes_different += 1 if len(label_to_passage['YES-DIFFERENT']) > 0 else 0
            self.num_entities_no += 1 if (len(label_to_passage['YES-SAME']) == 0 and len(label_to_passage['YES-DIFFERENT']) == 0) else 0
            
            relevant_passages = []
            rewrites = []
            
            for passage_id, passage_idx in label_to_passage['YES-SAME']:
                relevant_passages.append({
                    "passage_id": passage_id,
                    "rewrite": False
                })
            
            for passage_id, passage_idx in label_to_passage['YES-DIFFERENT']:
                passage = self.get_latest_passage_version(passage_id, passage_idx)
                # passage = self.corpus[passage_idx]
                # assert passage['id'] == passage_id
                total_llm_calls_for_this_property += 1
                rewrite_response = self.run_llm_as_rewriter(
                    entity_value,
                    property,
                    passage,
                    candidate_logger,
                    rewrite_type="REPLACE",
                )
                rewritten_passage = copy.deepcopy(passage)
                rewritten_passage['contents'] = rewrite_response
                rewrites.append({
                    "property_label": property['label'],
                    "shared_time": property["shared_time"],
                    "entity_value": entity_value,
                    "passage_idx": passage_idx,
                    "passage": passage,
                    "rewritten_passage": rewritten_passage,
                })
                relevant_passages.append({
                    "passage_id": passage_id,
                    "rewrite": True
                })
            
            if len(relevant_passages) == 0:
                rewrite_options = label_to_passage['NO-RELATED'] + label_to_passage['NO-UNRELATED']
                if len(rewrite_options) > 0:
                    passage_id, passage_idx = rewrite_options[0] # choose a related passage, or the first passage in document
                    passage = self.get_latest_passage_version(passage_id, passage_idx)
                    # passage = self.corpus[passage_idx]
                    # assert passage['id'] == passage_id
                    
                    total_llm_calls_for_this_property += 1
                    rewrite_response = self.run_llm_as_rewriter(
                        entity_value,
                        property,
                        passage,
                        candidate_logger,
                        rewrite_type="ADD"
                    )
                    rewritten_passage = copy.deepcopy(passage)
                    rewritten_passage['contents'] = rewrite_response
                    rewrites.append({
                        "property_label": property['label'],
                        "shared_time": property["shared_time"],
                        "entity_value": entity_value,
                        "passage_idx": passage_idx,
                        "passage": passage,
                        "rewritten_passage": rewritten_passage,
                    })
                    relevant_passages.append({
                        "passage_id": passage_id,
                        "rewrite": True
                    })
            
            main_logger.info(f"** Found {len(relevant_passages)} relevant passages, including {len(rewrites)} rewrites for instance {idx+1} **")
            candidate_logger.info(f"** Found {len(relevant_passages)} relevant passages, including {len(rewrites)} rewrites for instance {idx+1} **\n")
            

            for rewrite_record in rewrites:
                passage_id = rewrite_record["passage"]['id']
                self.rewrites[passage_id].append(rewrite_record)
            append_passages_to_rewrites_file(
                rewrites_path=self.passage_rewrites_path,
                rewrites=rewrites
            )
            self.results[aggregation_class_id][property['id']][instance["id"]] = {
                "property_label": property['label'],
                "shared_time": property["shared_time"],
                "entity_value": entity_value,
                "llm_judgments": label_to_passage,
                "relevant_passages": relevant_passages,
            }
            append_record_to_qrel_results_file(
                results_path=self.results_path,
                candidate_class_id=aggregation_class_id,
                property_id=property['id'],
                instance_id=instance["id"],
                qrel_result=self.results[aggregation_class_id][property['id']][instance["id"]]
            )

        
        main_logger.info(F"{total_llm_calls_for_this_property} LLM calls for Property {property['label']}")
        self.total_llm_calls += total_llm_calls_for_this_property
        
        # return total_llm_calls_for_this_property, missing_wikipage_ids
        # return True


    def run_llm_as_judge(self, entity_value, property, passage, candidate_logger):

        input_instance = self.prepare_input_instance(
            entity_value, 
            property, 
            passage
        )

        prompt = PROPERTY_CHECK_PROMPT.format(
            statement_explanation = DATATYPE_EXPLANATION[property['datatype']]
        )

        completion = self.property_client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_instance}
            ],
            model=self.property_model, 
            temperature=self.property_temperature,
        )

        response = completion.choices[0].message.content.strip()

        candidate_logger.info("PROMPT:\n" + input_instance)
        candidate_logger.info("RESPONSE:\n" + response + "\n")
        candidate_logger.info("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        if response not in ["YES-SAME", "YES-DIFFERENT", "NO-RELATED", "NO-UNRELATED"]:
            if "YES-SAME".lower() in response.lower():
                response = "YES-SAME"
            elif "YES-DIFFERENT".lower() in response.lower():
                response = "YES-DIFFERENT"
            elif "NO-RELATED".lower() in response.lower():
                response = "NO-RELATED"
            elif "NO-UNRELATED".lower() in response.lower():
                response = "NO-UNRELATED"
            else:
                response = "NO-UNRELATED"

        return response


    def run_llm_as_rewriter(self, entity_value, property, passage, candidate_logger, rewrite_type):

        input_instance = self.prepare_input_instance(
            entity_value, 
            property, 
            passage
        )

        prompt = REWRITE_PROMPT.format(
            rewrite_explanation = REWRITE_EXPLANATION[rewrite_type]
        )

        completion = self.rewrite_client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_instance}
            ],
            model=self.rewrite_model, 
            temperature=self.rewrite_temperature,
        )

        response = completion.choices[0].message.content.strip()
        response = re.sub(r'\s+', ' ', response).strip()

        
        candidate_logger.info(f"------------------------- REWRITE: {rewrite_type} -------------------------")
        candidate_logger.info("PROMPT:\n" + input_instance)
        candidate_logger.info("RESPONSE:\n" + response + "\n")
        
        return response
    

    def prepare_input_instance(self, entity_value, property, passage):
        shared_time = property.get("shared_time", None)
        if shared_time is not None:
            shared_time_str = format_time_for_prompt(shared_time=shared_time)
            statement_time = f"this statement is valid as of {shared_time_str}."  
        else:
            statement_time = ""

        input_instance = PROPERTY_CHECK_INPUT_TEMPLATE.format(
            entity_name = entity_value["entity_label"],
            property_name = property['label'],
            property_description = property['description'],
            property_value = format_value(property['datatype'], entity_value['value_node']),
            time_of_statement = statement_time,
            passage_title = passage['title'],
            sections = " - ".join(passage['section']),
            passage = passage['contents']
        )

        return input_instance
    

    def get_latest_passage_version(self, passage_id, passage_idx):
        if passage_id in self.rewrites:
            latest_rewrite = self.rewrites[passage_id][-1]
            assert latest_rewrite['passage_idx'] == passage_idx
            passage = latest_rewrite['rewritten_passage']
            assert passage['id'] == passage_id
            return passage
        else:
            passage = self.corpus[passage_idx]
            assert passage['id'] == passage_id
            return passage



def create_random_access_mapping(corpus, page2chunk2idx_path):
    
    if Path(page2chunk2idx_path).exists():
        main_logger.info("Loading page2chunk2idx mapping...") 
        page2chunk2idx = read_json_from_file(page2chunk2idx_path)
        main_logger.info(f"Loaded mapping for {len(page2chunk2idx)} wikipages.") 
        return page2chunk2idx
    
    main_logger.info("Creating page2chunk2idx mapping...")
    
    id2idx = {row["id"]: idx for idx, row in tqdm.tqdm(enumerate(corpus), total=len(corpus))}
    page2chunk2idx = {}
    for passage_id, idx in tqdm.tqdm(id2idx.items()):
        page_id, chunk_idx = passage_id.split("-")
        if page_id not in page2chunk2idx:
            page2chunk2idx[page_id] = {}
        page2chunk2idx[page_id][chunk_idx] = idx

    write_json_to_file(page2chunk2idx_path, page2chunk2idx)
    main_logger.info(f"Created mapping for {len(page2chunk2idx)} wikipages.")

    return page2chunk2idx


def create_candidate_logger(candidate_idx, log_dir):
    
    logger = logging.getLogger(f"candidate_{candidate_idx}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # IMPORTANT: don't leak into main log
    
    log_path = Path(log_dir) / f"candidate_{candidate_idx}.log"

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s")) 
    logger.addHandler(handler)

    return logger, handler
    


def main(args):

    # print("Loading generations...")
    # generations = read_jsonl_from_file(args.generations_path)
    # with open(args.log_file_path, 'w', encoding='utf-8') as log_file:
        
        
        main_logger.info("Loading candidates...")
        candidates = read_json_from_file(args.query_candidates_path)
        main_logger.info(f"Total candidates loaded: {len(candidates)}")
        
        # total_llm_calls = 0
        qrel_generator = QrelGenerator(args)

        candidates_processed = 0
        for idx, candidate in tqdm.tqdm(enumerate(candidates)):
            
            # if candidate['id'] not in [
            #     '8_Q1221156-P6-Q5_P39-P569',
            #     '40_Q83057-P37-Q1288568_P5109-P3823',
            #     '145_Q56059-P36-Q82794_P571', 
            #     '159_Q250811_P1082', 
            #     '162_Q1052743_P47', 
            # ]:
            #     continue
            
            candidate_logger, candidate_handler = create_candidate_logger(candidate_idx=idx, log_dir=args.log_file_dir)

            main_logger.info(f"---------- Processing candidate {idx+1}/{len(candidates)}: {candidate['id']} ----------") 
            candidate_logger.info(f"---------- Processing candidate {idx+1}/{len(candidates)}: {candidate['id']} ----------")

            if 'connecting_prop' in candidate:
                pass # multihop, for now skip checking
            
            if 'hop_class' in candidate:
                aggregation_subclass = candidate['hop_class']
            else:
                aggregation_subclass = candidate['src_class']

            aggregation_prop = candidate['aggregation_prop']
            instances = aggregation_subclass['instances']


            # num_llm_calls_aggregation, missing_wikipage_ids = qrel_generator.get_query_relevance_judgements(
            qrel_generator.get_query_relevance_judgements(
                instances, 
                aggregation_prop,
                candidate_id=candidate['id'],
                candidate_logger=candidate_logger
            )
            # TODO: remove queries with missing wikipages from candidates in next step
            
            # total_llm_calls += num_llm_calls_aggregation
            # print(f"+{num_llm_calls_aggregation}", end=' ')

            if 'constraint_prop' in candidate:
                constraint_prop = candidate['constraint_prop']

                # num_llm_calls_constraint, missing_wikipage_ids = qrel_generator.get_query_relevance_judgements(
                qrel_generator.get_query_relevance_judgements(
                    instances, 
                    constraint_prop,
                    candidate_id=candidate['id'],
                    candidate_logger=candidate_logger
                )
                # TODO: remove queries with missing wikipages from candidates in next step
                
                # total_llm_calls += num_llm_calls_constraint
                # print(f"+{num_llm_calls_constraint}", end=' ')

            candidate_logger.removeHandler(candidate_handler)
            candidate_handler.close()

            candidates_processed += 1
            
            # print(f"= {total_llm_calls}")

        main_logger.info("\n========== SUMMARY ==========")
        main_logger.info(f"Total candidates processed: {candidates_processed}/{len(candidates)}")
        main_logger.info(f"Total wikipages missing from corpus: {len(qrel_generator.missing_wikipage_ids)}")
        main_logger.info(f"Total queries removed: {qrel_generator.num_queries_missing_wikipages}")
        main_logger.info(f"Total LLM calls needed: {qrel_generator.total_llm_calls}") 
        main_logger.info(f"Total passages processed: {qrel_generator.total_passages_processed}")
        main_logger.info(f"Total YES-SAME passages: {qrel_generator.num_yes_same}")
        main_logger.info(f"Total YES-DIFFERENT passages: {qrel_generator.num_yes_different}")
        main_logger.info(f"Total entities with at least one YES-SAME passage: {qrel_generator.num_entities_yes_same}")
        main_logger.info(f"Total entities with at least one YES-DIFFERENT passage: {qrel_generator.num_entities_yes_different}")
        main_logger.info(f"Total entities with NO relevant passages: {qrel_generator.num_entities_no}")
        
        # print(f"Total wikipages missing from corpus: {len(missing_wikipages)}")
        # print(f"Total queries removed: {total_queries_removed}")
        # print(f"Total LLM calls needed: {total_llm_calls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-candidates-path", 
        type=str, 
        required=True, 
        help="Path to wikidata query candidates used for generations"
    )
    parser.add_argument(
        "--qrel-results-path",
        type=str,
        required=True,
        help="Path to JSONL file to write results to (and read existing results to resume progress)."
    )
    parser.add_argument(
        "--passage-rewrites-path",
        type=str,
        required=True,
        help="Path to JSONL file to write passage rewrites to."
    )
    # parser.add_argument(
    #     "--generations_path", 
    #     type=str, 
    #     required=True, 
    #     help="Path to JSON generations file, same as --output_file in generate_queries.py"
    # )
    parser.add_argument(
        "--log-file-dir", 
        type=str, 
        required=True, 
        help="Log file location (directory that will contain logs)"
    )
    parser.add_argument(
        "--corpus-path", 
        type=str, 
        required=True, 
        help="Path to corpus on huggingface hub or local directory"
    )
    parser.add_argument(
        "--page2chunk2idx-path", 
        type=str, 
        required=True, 
        help="Path to JSON file containing wikipageid->chunkid->corpusidx mapping" \
        "This needs a pre-iteration over corpus to allow look-ups by wiki_page_id"
    )
    parser.add_argument(
        "--judge-model",
        type=str, 
        required=True, 
        help="Model to use for generation e.g gpt-4o, o3-mini, etc."
    )    
    parser.add_argument(
        "--judge-temperature", 
        type=float, 
        default=0.0, 
        help="Temperature for generation"
    )
    parser.add_argument(
        "--rewrite-model",
        type=str, 
        required=True, 
        help="Model to use for generation e.g gpt-4o, o3-mini, etc."
    )    
    parser.add_argument(
        "--rewrite-temperature", 
        type=float, 
        default=0.7, 
        help="Temperature for generation"
    )
    args = parser.parse_args()
    
    
    # base_log_dir = Path(args.log_file_dir)
    # log_dir = base_log_dir / date.today().isoformat()
    log_dir = Path(args.log_file_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = log_dir / "extract_qrels.log"
    
    logging.basicConfig(
        filename=main_log_path,
        level=logging.INFO,
        # format="%(asctime)s | %(levelname)s | %(message)s",
        format="%(message)s",
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        # logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        logging.Formatter("%(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    main(args) 


    """
    python -m query_generation.extract_qrels \
        --query-candidates-path data/data_creation_runs/V4/query_generation_candidates.json \
        --qrel-results-path data/data_creation_runs/V4/qrels/qrel_results.jsonl \
        --passage-rewrites-path data/data_creation_runs/V4/qrels/passage_rewrites.jsonl \
        --log-file-dir data/data_creation_runs/V4/qrels/logs \
        --corpus-path HeydarS/enwiki_20251001_infoboxconv \
        --page2chunk2idx-path data/data_creation_runs/V4/enwiki_20251001_infoboxconv_page2chunk2idx.json \
        --judge-model gpt-4o-mini \
        --judge-temperature 0.0 \
        --rewrite-model gpt-4o-mini \
        --rewrite-temperature 0.7
    """

