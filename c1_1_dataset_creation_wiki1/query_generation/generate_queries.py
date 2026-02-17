from collections import defaultdict
import os
import io
import json
import argparse
from pathlib import Path
import random
import tqdm
from openai import OpenAI
from openai.types.chat import ChatCompletion

from io_utils import encode_datetime, read_json_from_file, write_json_to_file, write_jsonl_to_file, read_jsonl_from_file
from wikidata.data_utils import format_time_for_prompt, format_value
from query_generation.parse_generations import extract_query_text
from query_generation.prompts.query_generation.prompt_v3 import QUERY_GENERATION_PROMPT  
from wikidata.prop_operation_mapping import OPERATION_DESCRIPTIONS, CONSTRAINT_DESCRIPTIONS




def append_record_to_file(results_path, record):
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=encode_datetime) + '\n')


def load_qrel_results(qrel_results_path):
    qrel_results = {}
    
    if Path(qrel_results_path).exists():
        
        qrel_list = read_jsonl_from_file(qrel_results_path)
        for qrel_line in qrel_list:
            candidate_class_id = qrel_line['candidate_class_id']
            property_id = qrel_line['property_id']
            instance_id = qrel_line['instance_id']
            if candidate_class_id not in qrel_results:
                qrel_results[candidate_class_id] = {}
            if property_id not in qrel_results[candidate_class_id]:
                qrel_results[candidate_class_id][property_id] = {}
            qrel_results[candidate_class_id][property_id][instance_id] = qrel_line['qrel_result']
        
    else:
        raise FileNotFoundError(f"Qrel results file not found: {qrel_results_path}")
    
    return qrel_results


def build_prompt_inputs(candidate: dict) -> str:
    inputs_buffer = io.StringIO()
    entity_values_buffer = io.StringIO()

    # -------- Entity Set --------
    print("ENTITY SET", file=inputs_buffer)
    src_class = candidate["src_class"]
    print(f"|  Source Class: {src_class['label']} ({src_class['id']}) | #{src_class['count']} | {src_class['description']}", file=inputs_buffer)
    if "connecting_prop" in candidate and "hop_class" in candidate:
        connecting_prop = candidate["connecting_prop"]
        print(f"|  Connecting Property: {connecting_prop['label']} ({connecting_prop['id']}) | {connecting_prop['datatype']} | {connecting_prop['description']}", file=inputs_buffer)
        point_in_time = connecting_prop['shared_time']
        print(f"|  At Time: {'-' if point_in_time is None else format_time_for_prompt(shared_time=point_in_time)}", file=inputs_buffer)

        hop_class = candidate["hop_class"]
        print(f"|  Hop Class: {hop_class['label']} ({hop_class['id']}) | #{hop_class['count']} | {hop_class['description']}", file=inputs_buffer)


    # -------- Constraint Property --------
    if "constraint_prop" in candidate:
        print("CONSTRAINT PROPERTY", file=inputs_buffer)
        constraint_prop = candidate["constraint_prop"]
        print(f"|  Constraint Property: {constraint_prop['label']} ({constraint_prop['id']}) | {constraint_prop['datatype']} | {constraint_prop['description']}", file=inputs_buffer)
        if constraint_prop['datatype'] == 'WikibaseItem':
            print(f"|  Item Class: {constraint_prop['item_class'][1]}", file=inputs_buffer)
        point_in_time = constraint_prop['shared_time']
        print(f"|  At Time: {'-' if point_in_time is None else format_time_for_prompt(shared_time=point_in_time)}", file=inputs_buffer)
        constraint_description = CONSTRAINT_DESCRIPTIONS[constraint_prop['constraint']]
        print(f"|  Constraint: {constraint_prop['constraint']} | {constraint_description}", file=inputs_buffer)
        if constraint_prop['datatype'] == 'WikibaseItem':
            value_string = format_value(constraint_prop['datatype'], constraint_prop['reference_entity'])
            print(f"|  Reference Entity: [{value_string}]", file=inputs_buffer)
        else: 
            value_string = format_value(constraint_prop['datatype'], constraint_prop['reference_entity']['value_node'])
            print(f"|  Reference Entity: [{constraint_prop['reference_entity']['entity_label']}] | [{value_string}]", file=inputs_buffer)


    # -------- Aggregation Property --------
    print("AGGREGATION PROPERTY", file=inputs_buffer)
    aggregation_prop = candidate["aggregation_prop"]
    print(f"|  Aggregation Property: {aggregation_prop['label']} ({aggregation_prop['id']}) | {aggregation_prop['datatype']} | {aggregation_prop['description']}", file=inputs_buffer)
    if aggregation_prop['datatype'] == 'WikibaseItem':
        print(f"|  Item Class: {aggregation_prop['item_class'][1]}", file=inputs_buffer)
    elif aggregation_prop['datatype'] == 'Quantity':
        unit_id, unit_label = aggregation_prop['unit']
        if unit_id != "Q199": # excluding unit=1 (Q199)
            print(f"|  Unit: {unit_label}", file=inputs_buffer)
    elif aggregation_prop['datatype'] == 'Time':
        calendar_id, calendar_label = aggregation_prop['calendar']
        print(f"|  Calendar: {calendar_label}", file=inputs_buffer)
    point_in_time = aggregation_prop['shared_time']
    print(f"|  At Time: {'-' if point_in_time is None else format_time_for_prompt(shared_time=point_in_time)}", file=inputs_buffer)


    # -------- Aggregation Operation --------
    print("AGGREGATION OPERATION", file=inputs_buffer)
    operation_str = candidate['operation']
    if candidate['operation_args']:
        operation_str += f", {candidate['operation_args']}"
    operation_description = OPERATION_DESCRIPTIONS[candidate['operation']]
    print(f"|  Operation: {operation_str} | {operation_description}", file=inputs_buffer)
    print(f"|  Final Answer: {candidate['final_answer']}", file=inputs_buffer)


    # -------- Entity Values --------
    print("ENTITY VALUES", file=entity_values_buffer)
    
    for idx, entity_value in enumerate(sorted(
        aggregation_prop['list_of_entity_values'], 
        key=lambda x: x['entity_id']
    )):
        bullet = "[YES]" if entity_value['entity_id'] in candidate['filtered_entity_ids'] else "[NO]"
        print(f"  {bullet} {entity_value['entity_label']} ({entity_value['entity_id']}) = {format_value(aggregation_prop['datatype'], entity_value['value_node'])}", file=entity_values_buffer)
        if 'constraint_prop' in candidate:
            constraint_prop = candidate['constraint_prop']
            constraint_entity_value = sorted(
                constraint_prop['list_of_entity_values'], 
                key=lambda x: x['entity_id']
            )[idx]
            assert constraint_entity_value['entity_id'] == entity_value['entity_id']
            print(f"    |__ Constraint value: {format_value(constraint_prop['datatype'], constraint_entity_value['value_node'])}", file=entity_values_buffer)


    return inputs_buffer.getvalue(), entity_values_buffer.getvalue()



def run_llm_as_query_generator(client: OpenAI, model: str, temperature: float, prompt_inputs: str) -> ChatCompletion:
    
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": QUERY_GENERATION_PROMPT},
            {"role": "user", "content": prompt_inputs},
        ],
        model=model,
        temperature=temperature,
    )

    return completion



# TODO: add resume capability
def generate_queries(args):

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    candidates = read_json_from_file(args.candidates_path)
    qrel_results = load_qrel_results(args.qrel_results_path)

    generations, queries = [], []
    trec_qrels = []
    qrels = defaultdict(dict)
    with open(args.log_file, "w", encoding="utf-8") as log:
        for candidate in tqdm.tqdm(candidates, total=len(candidates)):

            candidate_num, aggregation_class_id, prop_combo_id = candidate['id'].split('_')
            if (
                aggregation_class_id not in qrel_results or 
                candidate['aggregation_prop']['id'] not in qrel_results[aggregation_class_id]
            ):
                print(f"Skipping candidate {candidate['id']} due to missing qrel results.", file=log)
                continue


            prompt_inputs, entity_values_log = build_prompt_inputs(candidate)
            
            completion = run_llm_as_query_generator(
                client=client,
                model=args.model,
                temperature=args.temperature,
                prompt_inputs=prompt_inputs,
            )

            query_text = completion.choices[0].message.content.strip()
            query_text = " ".join(query_text.splitlines())
            query_text = " ".join(query_text.split())
            query_text = query_text.strip()
            
            generation = {
                "id": candidate["id"],
                "candidate": candidate,
                "completion": completion.to_dict(),
            }
            generations.append(generation)
            append_record_to_file(args.generations_out, generation)

            query = {
                "id": candidate["id"],
                "question": query_text,
                "answer": {
                    "type": candidate["aggregation_prop"]["datatype"],
                    "value": candidate["final_answer"],
                }
            }
            queries.append(query)
            append_record_to_file(args.queries_out, query)

            aggregation_prop_id = candidate['aggregation_prop']['id']
            filtered_entity_ids = candidate['filtered_entity_ids']
            aggregation_qrels = qrel_results[aggregation_class_id][aggregation_prop_id]
            for instance_id in filtered_entity_ids:
                instance_qrels = aggregation_qrels[instance_id]
                assert len(instance_qrels['relevant_passages']) > 0
                for relevant_passage in instance_qrels['relevant_passages']:
                    passage_id = relevant_passage['passage_id']
                    qrels[candidate['id']][passage_id] = 1 
                    # TREC format: query_id 0 doc_id relevance
                    trec_qrels.append(f"{candidate['id']} 0 {passage_id} 1")


            print(f"\n-------------------------- Candidate ID: {candidate['id']} --------------------------\n", file=log)
            print("******** PROMPT INPUTS ********\n" + prompt_inputs, file=log)
            print(entity_values_log + "\n", file=log)
            print("******** GENERATED QUERY ********\n" + query_text, file=log)


    write_json_to_file(args.qrels_out, qrels)
    with open(args.trec_qrels_out, "w", encoding="utf-8") as f:
        for line in trec_qrels:
            f.write(line + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidates-path", type=str, required=True, help="JSON file: list of pre-selected candidates")
    parser.add_argument("--qrel-results-path", type=str, required=True, help="JSONL file: qrel results for candidates")

    parser.add_argument("--generations-out", type=str, required=True, help="Output JSONL for generations (raw completions)")
    parser.add_argument("--queries-out", type=str, required=True, help="Output JSONL for parsed queries")
    parser.add_argument("--qrels-out", type=str, required=True, help="Output JSON for qrels")
    parser.add_argument("--trec-qrels-out", type=str, required=True, help="Output file for TREC style qrels")

    parser.add_argument("--log-file", type=str, required=True)

    parser.add_argument("--model", type=str, required=True, help="e.g. gpt-4o-mini, gpt-5-mini, etc.")
    parser.add_argument("--temperature", type=float, default=0.2)

    args = parser.parse_args()
    
    out_folder = Path(args.generations_out).parent
    out_folder.mkdir(parents=True, exist_ok=True)
    log_folder = Path(args.log_file).parent
    log_folder.mkdir(parents=True, exist_ok=True)

    generate_queries(args)


    """
    python -m query_generation.generate_queries \
        --candidates-path data/data_creation_runs/V4/query_generation_candidates_operations.json \
        --qrel-results-path data/data_creation_runs/V4/qrels/qrel_results.jsonl \
        --generations-out data/data_creation_runs/V4/queries/generations.jsonl \
        --queries-out data/data_creation_runs/V4/queries/queries.jsonl \
        --qrels-out data/data_creation_runs/V4/queries/qrels.json \
        --trec-qrels-out data/data_creation_runs/V4/queries/trec_qrels.txt \
        --log-file data/data_creation_runs/V4/queries/logs/query_generation.log \
        --model gpt-5.2 \
        --temperature 0.5



    python -m query_generation.generate_queries \
        --candidates-path data/data_creation_runs/V5_validation/query_generation_candidates_validation_combined.json \
        --qrel-results-path data/data_creation_runs/V5_validation/qrels/qrel_results.jsonl \
        --generations-out data/data_creation_runs/V5_validation/queries/generations.jsonl \
        --queries-out data/data_creation_runs/V5_validation/queries/queries.jsonl \
        --qrels-out data/data_creation_runs/V5_validation/queries/qrels.json \
        --trec-qrels-out data/data_creation_runs/V5_validation/queries/trec_qrels.txt \
        --log-file data/data_creation_runs/V5_validation/queries/logs/query_generation.log \
        --model gpt-5.2 \
        --temperature 0.5
    """
