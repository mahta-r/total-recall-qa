from collections import defaultdict
import os
import io
import json
import argparse
from pathlib import Path
import tqdm
from openai import OpenAI

from io_utils import read_json_from_file, write_json_to_file, read_jsonl_from_file
from prompts.query_generation.prompt_v3 import QUERY_GENERATION_PROMPT
from prop_operation_mapping import OPERATION_DESCRIPTIONS, CONSTRAINT_DESCRIPTIONS



def append_record_to_file(results_path, record):
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_qrel_results(qrel_results_path):
    qrel_results = {}

    if not Path(qrel_results_path).exists():
        raise FileNotFoundError(f"Qrel results file not found: {qrel_results_path}")

    qrel_list = read_jsonl_from_file(qrel_results_path)
    for qrel_line in qrel_list:
        candidate_class_id = qrel_line['candidate_class_id']
        property_label = qrel_line['property_label']
        entity_id = qrel_line['entity_id']
        if candidate_class_id not in qrel_results:
            qrel_results[candidate_class_id] = {}
        if property_label not in qrel_results[candidate_class_id]:
            qrel_results[candidate_class_id][property_label] = {}
        qrel_results[candidate_class_id][property_label][entity_id] = qrel_line['qrel_result']

    return qrel_results


def format_reference(reference, unit=None):
    """Format constraint reference value for the prompt."""
    if isinstance(reference, list):
        formatted = ", ".join(str(v) for v in reference)
        return f"[{formatted}]"
    ref_str = str(reference)
    if unit is not None:
        short_unit, long_unit = unit
        ref_str = f"{ref_str} {short_unit} ({long_unit})"
    return ref_str


def get_qrel_prop_label(candidate):
    """Determine which property's qrels to use for this candidate.

    If aggregation_prop exists, use its label.
    Otherwise (COUNT), use the last non-Listed/Mentioned constraint's label.
    """
    if candidate.get('aggregation_prop') is not None:
        return candidate['aggregation_prop']['label']
    else:
        valid_constraints = [
            c for c in candidate['constraint_props']
            if c.get('reference') != "Listed/Mentioned"
        ]
        return valid_constraints[-1]['label']


def build_prompt_inputs(candidate: dict):
    inputs_buffer = io.StringIO()
    entity_values_buffer = io.StringIO()

    # -------- Entity Set --------
    src_class = candidate["src_class"]
    print("ENTITY SET", file=inputs_buffer)
    print(f"|  Product Category: {src_class['label']} | #{src_class['count']}", file=inputs_buffer)

    # -------- Constraint Properties (ALL, including Listed/Mentioned) --------
    for constraint_prop in candidate.get('constraint_props', []):
        print("CONSTRAINT PROPERTY", file=inputs_buffer)
        print(f"|  Constraint Property: {constraint_prop['label']} | {constraint_prop['datatype']}", file=inputs_buffer)
        if constraint_prop['datatype'] == 'Quantity' and constraint_prop.get('unit'):
            _, unit_label = constraint_prop['unit']
            print(f"|  Unit: {unit_label}", file=inputs_buffer)
        constraint_desc = CONSTRAINT_DESCRIPTIONS.get(constraint_prop['constraint'], constraint_prop['constraint'])
        print(f"|  Constraint: {constraint_prop['constraint']} | {constraint_desc}", file=inputs_buffer)
        ref_str = format_reference(constraint_prop['reference'], constraint_prop.get('unit') if constraint_prop['datatype'] == 'Quantity' else None)
        print(f"|  Reference: {ref_str}", file=inputs_buffer)

    # -------- Aggregation Property --------
    aggregation_prop = candidate.get("aggregation_prop")
    if aggregation_prop is not None:
        print("AGGREGATION PROPERTY", file=inputs_buffer)
        print(f"|  Aggregation Property: {aggregation_prop['label']} | {aggregation_prop['datatype']}", file=inputs_buffer)
        if aggregation_prop['datatype'] == 'Quantity' and aggregation_prop.get('unit'):
            _, unit_label = aggregation_prop['unit']
            print(f"|  Unit: {unit_label}", file=inputs_buffer)

    # -------- Aggregation Operation --------
    print("AGGREGATION OPERATION", file=inputs_buffer)
    operation_str = candidate['operation']
    if candidate['operation_args']:
        operation_str += f", {candidate['operation_args']}"
    operation_description = OPERATION_DESCRIPTIONS[candidate['operation']]
    print(f"|  Operation: {operation_str} | {operation_description}", file=inputs_buffer)
    print(f"|  Final Answer: {candidate['final_answer']}", file=inputs_buffer)
    if aggregation_prop is not None and aggregation_prop['datatype'] == 'Quantity' and aggregation_prop.get('unit'):
        short_unit, _ = aggregation_prop['unit']
        print(f"|  Answer Unit: {short_unit}", file=inputs_buffer)

    # -------- Entity Values (for logging only) --------
    # Build set of filtered entity IDs for [YES]/[NO] marking
    filtered_set = {(eid[0], eid[1]) for eid in candidate['filtered_entity_ids']}

    # Use aggregation_prop entity values if available, otherwise last valid constraint
    if aggregation_prop is not None:
        value_entities = aggregation_prop['list_of_entity_values']
        value_prop_label = aggregation_prop['label']
    else:
        valid_constraints = [
            c for c in candidate['constraint_props']
            if c.get('reference') != "Listed/Mentioned"
        ]
        value_entities = valid_constraints[-1]['list_of_entity_values']
        value_prop_label = valid_constraints[-1]['label']

    print("ENTITY VALUES", file=entity_values_buffer)
    for entity_value in sorted(value_entities, key=lambda x: x['entity_id']):
        bullet = "[YES]" if (entity_value['entity_id'], entity_value['entity_label']) in filtered_set else "[NO]"
        value = entity_value['value_node']['value']
        print(f"  {bullet} {entity_value['entity_label']} = {value}", file=entity_values_buffer)

    return inputs_buffer.getvalue(), entity_values_buffer.getvalue()


def run_llm_as_query_generator(client, model, temperature, prompt_inputs):

    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": QUERY_GENERATION_PROMPT},
            {"role": "user", "content": prompt_inputs},
        ],
        model=model,
        temperature=temperature,
    )

    return completion


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

            candidate_class_id = candidate['id'].split('_', 1)[1]
            qrel_prop_label = get_qrel_prop_label(candidate)

            # Check qrels exist for this candidate
            if (
                candidate_class_id not in qrel_results or
                qrel_prop_label not in qrel_results[candidate_class_id]
                or candidate['id'] == "108_BCK"
            ):
                print(f"Skipping candidate {candidate['id']} due to missing qrel results for property '{qrel_prop_label}'.", file=log)
                continue

            prompt_inputs, entity_values_log = build_prompt_inputs(candidate)

            # Print prompt inputs only mode
            if args.print_inputs:
                print(f"\n{'='*80}", file=log)
                print(f"Candidate ID: {candidate['id']} | Candidate class ID: {candidate_class_id}", file=log)
                print(f"{'='*80}", file=log)
                print(prompt_inputs, file=log)
                print(entity_values_log, file=log)
                continue

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
            # generations.append(generation)
            append_record_to_file(args.generations_out, generation)

            answer_type = candidate['aggregation_prop']['datatype'] if candidate.get('aggregation_prop') else "Count"
            query = {
                "id": candidate["id"],
                "question": query_text,
                "answer": candidate["final_answer"],
                "datatype": answer_type,
            }
            # queries.append(query)
            append_record_to_file(args.queries_out, query)

            # Build qrels from qrel_results for filtered entities
            prop_qrels = qrel_results[candidate_class_id][qrel_prop_label]
            for eid in candidate['filtered_entity_ids']:
                entity_id = eid[0]  # (entity_id, entity_label) — JSON round-trip from tuple
                if entity_id not in prop_qrels:
                    print(f"WARNING: entity {entity_id} not found in qrel results for {candidate['id']} / {qrel_prop_label}", file=log)
                    continue
                instance_qrels = prop_qrels[entity_id]
                for relevant_passage in instance_qrels['relevant_passages']:
                    passage_id = relevant_passage['passage_id']
                    qrels[candidate['id']][passage_id] = 1
                    trec_qrels.append(f"{candidate['id']} 0 {passage_id} 1")

            print(f"\n-------------------------- Candidate ID: {candidate['id']} | Candidate class ID: {candidate_class_id} --------------------------\n", file=log)
            print("******** PROMPT INPUTS ********\n" + prompt_inputs, file=log)
            print(entity_values_log + "\n", file=log)
            print("******** GENERATED QUERY ********\n" + query_text, file=log)

            if args.max_queries is not None and len(queries) >= args.max_queries:
                print(f"\nReached max queries limit ({args.max_queries}), stopping.", file=log)
                break

    if not args.print_inputs:
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

    parser.add_argument("--max-queries", type=int, default=None,
                        help="Maximum number of queries to generate (for testing). Stops after N successful queries.")
    parser.add_argument("--print-inputs", action="store_true",
                        help="Print prompt inputs for all candidates and exit (no LLM calls). Use to inspect prompts and pick examples.")

    args = parser.parse_args()

    out_folder = Path(args.generations_out).parent
    out_folder.mkdir(parents=True, exist_ok=True)
    log_folder = Path(args.log_file).parent
    log_folder.mkdir(parents=True, exist_ok=True)

    generate_queries(args)


    """
    # Print prompt inputs only (no LLM calls):
    python generate_queries.py \
        --candidates-path data/final/candidates/query_generation_candidates.json \
        --qrel-results-path data/final/qrels/qrel_results.jsonl \
        --generations-out data/final/queries/e_commerce_generations.jsonl \
        --queries-out data/final/queries/e_commerce_queries.jsonl \
        --qrels-out data/final/queries/e_commerce_qrels.json \
        --trec-qrels-out data/final/queries/e_commerce_trec_qrels.txt \
        --log-file data/final/queries/logs/query_generation.log \
        --model gpt-5.2 \
        --print-inputs

    # Full run:
    python generate_queries.py \
        --candidates-path data/final/candidates/query_generation_candidates.json \
        --qrel-results-path data/final/qrels/qrel_results.jsonl \
        --generations-out data/final/queries/e_commerce_generations.jsonl \
        --queries-out data/final/queries/e_commerce_queries.jsonl \
        --qrels-out data/final/queries/e_commerce_qrels.json \
        --trec-qrels-out data/final/queries/e_commerce_trec_qrels.txt \
        --log-file data/final/queries/logs/query_generation.log \
        --model gpt-5.2 \
        --temperature 0.2
    """
