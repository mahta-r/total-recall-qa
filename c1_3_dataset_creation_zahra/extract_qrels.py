import os
import json
import re
import copy
import logging
import sys
import random
import tqdm
import argparse
import asyncio

from collections import defaultdict
from pathlib import Path
from openai import OpenAI, AsyncOpenAI

from io_utils import read_jsonl_from_file, read_json_from_file, write_json_to_file

from prompts.LLM_as_relevance_judge.prompt_rewrite import (
    REWRITE_PROMPT,
    REWRITE_EXPLANATION,
    INPUT_TEMPLATE as REWRITE_INPUT_TEMPLATE,
)

from prompts.LLM_as_relevance_judge.prompt_combined import (
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
            property_label = result_line['property_label']
            entity_id = result_line['entity_id']
            if candidate_class_id not in results:
                results[candidate_class_id] = {}
            if property_label not in results[candidate_class_id]:
                results[candidate_class_id][property_label] = {}
            results[candidate_class_id][property_label][entity_id] = result_line['qrel_result']

        main_logger.info(f"Loaded qrel results for {len(results)} candidate classes.")

    return results


def append_record_to_qrel_results_file(results_path, candidate_class_id, property_label, entity_id, qrel_result):
    with open(results_path, 'a', encoding='utf-8') as f:
        record = {
            "candidate_class_id": candidate_class_id,
            "property_label": property_label,
            "entity_id": entity_id,
            "qrel_result": qrel_result
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


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
            f.write(json.dumps(rewrite_record, ensure_ascii=False) + '\n')


class QrelGenerator:
    def __init__(self, args):

        main_logger.info("Loading corpus...")

        self.corpus = read_jsonl_from_file(args.corpus_path)
        self.title2idx = create_random_access_mapping(self.corpus, args.title2idx_path)

        self.property_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.property_model = args.judge_model
        self.property_temperature = args.judge_temperature
        self.max_concurrent_calls = args.max_concurrent_calls

        self.rewrite_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rewrite_model = args.rewrite_model
        self.rewrite_temperature = args.rewrite_temperature

        self.results_path = args.qrel_results_path
        self.results = load_qrel_results_if_exists(args.qrel_results_path)

        self.passage_rewrites_path = args.passage_rewrites_path
        self.rewrites = load_existing_rewrites_if_exists(args.passage_rewrites_path)

        self.max_llm_calls = getattr(args, 'max_llm_calls', None)
        self.total_llm_calls = 0
        self.entities_missing_passages = []
        self.missing_entity_labels = set()
        self.total_passages_processed = 0
        self.num_yes_same = 0
        self.num_yes_different = 0
        self.num_entities_yes_same = 0
        self.num_entities_yes_different = 0
        self.num_entities_no = 0


    async def get_query_relevance_judgements(
        self,
        entity_values,
        property_label,
        property_datatype,
        product_type,
        candidate_class_id,
        candidate_logger,
        entity_progress_bar=None
    ):

        # Check all entity_labels have passages in corpus
        missing_labels = []
        for ev in entity_values:
            entity_label = ev["entity_label"]
            if entity_label not in self.title2idx:
                missing_labels.append(entity_label)
                main_logger.warning(f"{entity_label} not found in corpus mapping.")
                candidate_logger.warning(f"{entity_label} not found in corpus mapping.")
        if missing_labels:
            self.missing_entity_labels.update(missing_labels)
            return

        if candidate_class_id not in self.results:
            self.results[candidate_class_id] = {}
        if property_label not in self.results[candidate_class_id]:
            self.results[candidate_class_id][property_label] = {}

        total_llm_calls_for_this_property = 0

        for idx, entity_value in enumerate(entity_values):
            entity_id = entity_value["entity_id"]
            entity_label = entity_value["entity_label"]

            main_logger.info(f"---> Entity {idx+1}/{len(entity_values)}: {entity_label} | Property: {property_label}")

            candidate_logger.info(f"--------------------------------------------------------------------------------")
            candidate_logger.info(f"---> Entity {idx+1}/{len(entity_values)}: {entity_label} | Property: {property_label}")
            candidate_logger.info(f"--------------------------------------------------------------------------------")

            if entity_id in self.results[candidate_class_id][property_label]:
                main_logger.info(f"Already processed entity {idx+1}, skipping.")
                if entity_progress_bar is not None:
                    entity_progress_bar.update(1)
                continue # already processed

            passage_indices = self.title2idx[entity_label]

            # Heuristic: if value appears lexically in any passage, treat as YES-SAME (skip LLM)
            prop_value = str(entity_value['value_node']['value']).lower()
            if prop_value not in ('yes', 'no'):
                heuristic_matched = []
                for pidx in passage_indices:
                    pid = self.corpus[pidx]['id']
                    passage = self.get_latest_passage_version(pidx, pid)
                    if prop_value in passage['contents'].lower():
                        heuristic_matched.append((pid, pidx))

                if len(heuristic_matched) > 0:
                    main_logger.info(f"  HEURISTIC: lexical match in {len(heuristic_matched)}/{len(passage_indices)} passages, skipping LLM")
                    candidate_logger.info(f"  HEURISTIC: lexical match in {len(heuristic_matched)}/{len(passage_indices)} passages, skipping LLM")

                    relevant_passages = [{"passage_id": pid, "rewrite": False} for pid, _ in heuristic_matched]

                    self.total_passages_processed += len(passage_indices)
                    self.num_yes_same += len(heuristic_matched)
                    self.num_entities_yes_same += 1

                    self.results[candidate_class_id][property_label][entity_id] = {
                        "property_label": property_label,
                        "entity_value": entity_value,
                        "llm_judgments": {"YES-SAME": heuristic_matched},
                        "relevant_passages": relevant_passages,
                        "heuristic_resolved": True,
                    }
                    append_record_to_qrel_results_file(
                        results_path=self.results_path,
                        candidate_class_id=candidate_class_id,
                        property_label=property_label,
                        entity_id=entity_id,
                        qrel_result=self.results[candidate_class_id][property_label][entity_id]
                    )
                    if entity_progress_bar is not None:
                        entity_progress_bar.update(1)
                    continue

            # Check LLM call limit
            if self.max_llm_calls is not None and self.total_llm_calls >= self.max_llm_calls:
                main_logger.info(f"Reached max LLM calls limit ({self.max_llm_calls}), stopping.")
                return

            label_to_passage = defaultdict(list) # YES/NO answers from LLM

            passage_labeling_tasks, task_meta = [], []
            semaphore = asyncio.Semaphore(self.max_concurrent_calls)
            for passage_idx in passage_indices:
                passage_id = self.corpus[passage_idx]["id"]
                passage = self.get_latest_passage_version(passage_idx, passage_id)

                passage_labeling_tasks.append(
                    self._judge_with_semaphore(
                        semaphore,
                        entity_value,
                        property_label,
                        property_datatype,
                        product_type,
                        passage,
                        candidate_logger,
                    )
                )
                task_meta.append((passage_id, passage_idx))

            total_llm_calls_for_this_property += len(passage_labeling_tasks)

            responses = await asyncio.gather(*passage_labeling_tasks)

            for (passage_id, passage_idx), response in zip(task_meta, responses):
                label_to_passage[response].append((passage_id, passage_idx))


            for label, passage_ids in label_to_passage.items():
                main_logger.info(f"  {label}: {len(passage_ids)} passages")

            self.total_passages_processed += len(passage_indices)
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
                passage = self.get_latest_passage_version(passage_idx, passage_id)
                total_llm_calls_for_this_property += 1
                rewrite_response = self.run_llm_as_rewriter(
                    entity_value,
                    property_label,
                    property_datatype,
                    product_type,
                    passage,
                    candidate_logger,
                    rewrite_type="REPLACE",
                )
                rewritten_passage = copy.deepcopy(passage)
                rewritten_passage['contents'] = rewrite_response
                rewrites.append({
                    "property_label": property_label,
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
                    passage = self.get_latest_passage_version(passage_idx, passage_id)

                    total_llm_calls_for_this_property += 1
                    rewrite_response = self.run_llm_as_rewriter(
                        entity_value,
                        property_label,
                        property_datatype,
                        product_type,
                        passage,
                        candidate_logger,
                        rewrite_type="ADD"
                    )
                    rewritten_passage = copy.deepcopy(passage)
                    rewritten_passage['contents'] = rewrite_response
                    rewrites.append({
                        "property_label": property_label,
                        "entity_value": entity_value,
                        "passage_idx": passage_idx,
                        "passage": passage,
                        "rewritten_passage": rewritten_passage,
                    })
                    relevant_passages.append({
                        "passage_id": passage_id,
                        "rewrite": True
                    })

            main_logger.info(f"** Found {len(relevant_passages)} relevant passages, including {len(rewrites)} rewrites for entity {idx+1} **")
            candidate_logger.info(f"** Found {len(relevant_passages)} relevant passages, including {len(rewrites)} rewrites for entity {idx+1} **\n")


            for rewrite_record in rewrites:
                passage_id = rewrite_record["passage"]['id']
                self.rewrites[passage_id].append(rewrite_record)
            append_passages_to_rewrites_file(
                rewrites_path=self.passage_rewrites_path,
                rewrites=rewrites
            )
            self.results[candidate_class_id][property_label][entity_id] = {
                "property_label": property_label,
                "entity_value": entity_value,
                "llm_judgments": label_to_passage,
                "relevant_passages": relevant_passages,
            }
            append_record_to_qrel_results_file(
                results_path=self.results_path,
                candidate_class_id=candidate_class_id,
                property_label=property_label,
                entity_id=entity_id,
                qrel_result=self.results[candidate_class_id][property_label][entity_id]
            )

            if entity_progress_bar is not None:
                entity_progress_bar.update(1)


        main_logger.info(f"{total_llm_calls_for_this_property} LLM calls for Property {property_label}")
        self.total_llm_calls += total_llm_calls_for_this_property


    async def _judge_with_semaphore(
        self,
        semaphore,
        entity_value,
        property_label,
        property_datatype,
        product_type,
        passage,
        candidate_logger,
    ):
        async with semaphore:
            return await self.run_llm_as_judge(
                entity_value,
                property_label,
                property_datatype,
                product_type,
                passage,
                candidate_logger,
            )


    async def run_llm_as_judge(self, entity_value, property_label, property_datatype, product_type, passage, candidate_logger):

        # TODO: add heuristic check before calling LLM


        input_instance = self.prepare_input_instance(
            entity_value,
            property_label,
            property_datatype,
            product_type,
            passage
        )

        prompt = PROPERTY_CHECK_PROMPT.format(
            statement_explanation = DATATYPE_EXPLANATION[property_datatype]
        )

        completion = await self.property_client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_instance}
            ],
            model=self.property_model,
            temperature=self.property_temperature,
        )

        response = completion.choices[0].message.content.strip()

        passage_log = (
            f"PASSAGE ID: {passage['id']}\n"
            f"PROMPT:\n{input_instance}\n\n"
            f"RESPONSE:\n{response}\n"
            f"{'- '*40}"
        )
        candidate_logger.info(passage_log)

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


    def run_llm_as_rewriter(self, entity_value, property_label, property_datatype, product_type, passage, candidate_logger, rewrite_type):

        input_instance = self.prepare_input_instance(
            entity_value,
            property_label,
            property_datatype,
            product_type,
            passage,
            use_rewrite_template=True,
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


    def prepare_input_instance(self, entity_value, property_label, property_datatype, product_type, passage, use_rewrite_template=False):
        property_value = entity_value['value_node']['value']

        template = REWRITE_INPUT_TEMPLATE if use_rewrite_template else PROPERTY_CHECK_INPUT_TEMPLATE

        input_instance = template.format(
            product_type=product_type,
            entity_name=entity_value["entity_label"],
            property_name=property_label,
            property_value=property_value,
            passage=passage['contents'],
        )

        return input_instance


    def get_latest_passage_version(self, passage_idx, passage_id):
        passage = self.corpus[passage_idx]
        assert passage_id == passage['id']
        if passage_id in self.rewrites:
            latest_rewrite = self.rewrites[passage_id][-1]
            return latest_rewrite['rewritten_passage']
        else:
            passage = self.corpus[passage_idx]
            assert passage_id == passage['id']
            return passage


    def count_expected_llm_calls(self, candidates):
        """Dry-run: count expected LLM calls without making any API calls."""
        total_judge = 0
        total_entities = 0
        heuristic_resolved_entities = 0
        heuristic_resolved_passages = 0
        processed = set()  # (candidate_class_id, property_label, entity_id)

        heuristic_samples = []  # collect (entity, property, value, passage snippet) for inspection
        entities_missing = set()
        queries_with_missing_entities = set()

        for candidate in candidates:
            candidate_class_id = candidate['id'].split('_', 1)[1]

            # Only process entities in the final filtered set
            final_entity_set = {
                (eid[0], eid[1]) for eid in candidate['filtered_entity_ids']
            }

            props_to_process = []

            if candidate.get('aggregation_prop') is not None:
                props_to_process.append(candidate['aggregation_prop'])
            for cp in candidate.get('constraint_props', []):
                if cp.get('reference') == "Listed/Mentioned":
                    continue  # no value to check — entities verified via aggregation_prop
                props_to_process.append(cp)

            for prop in props_to_process:
                for ev in prop['list_of_entity_values']:
                    # Only process entities in the final filtered set
                    if (ev['entity_id'], ev['entity_label']) not in final_entity_set:
                        continue

                    key = (candidate_class_id, prop['label'], ev['entity_id'])
                    if key in processed:
                        continue
                    # Skip if already in results (resume)
                    if (candidate_class_id in self.results
                        and prop['label'] in self.results[candidate_class_id]
                        and ev['entity_id'] in self.results[candidate_class_id][prop['label']]):
                        continue
                    processed.add(key)
                    entity_label = ev['entity_label']
                    if entity_label not in self.title2idx:
                        entities_missing.add(entity_label)
                        queries_with_missing_entities.add(candidate['id'])
                        main_logger.warning(f"[dry-run] {entity_label} not found in corpus mapping.")
                        continue

                    passage_indices = self.title2idx[entity_label]
                    num_passages = len(passage_indices)
                    total_entities += 1

                    # Heuristic: check if property value appears lexically in any passage
                    prop_value = str(ev['value_node']['value']).lower()
                    if prop_value in ('yes', 'no'):
                        total_judge += num_passages
                        continue
                    matched_passages_for_entity = []
                    for pidx in passage_indices:
                        passage_text = self.corpus[pidx]['contents']
                        if prop_value in passage_text.lower():
                            matched_passages_for_entity.append(pidx)

                    if len(matched_passages_for_entity) > 0:
                        heuristic_resolved_entities += 1
                        heuristic_resolved_passages += len(matched_passages_for_entity)
                        # Collect one sample per entity for inspection
                        sample_pidx = matched_passages_for_entity[0]
                        heuristic_samples.append({
                            'entity_label': entity_label,
                            'property_label': prop['label'],
                            'property_value': ev['value_node']['value'],
                            'passage_id': self.corpus[sample_pidx]['id'],
                            'passage_snippet': self.corpus[sample_pidx]['contents'],
                        })
                    else:
                        total_judge += num_passages  # all passages need LLM

        llm_entities = total_entities - heuristic_resolved_entities
        main_logger.info(f"========== DRY RUN RESULTS ==========")
        main_logger.info(f"Total entities to process (final filtered only): {total_entities}")
        main_logger.info(f"Entities resolved by heuristic (lexical match): {heuristic_resolved_entities}")
        main_logger.info(f"  -> passages with lexical match: {heuristic_resolved_passages}")
        main_logger.info(f"Entities needing LLM: {llm_entities}")
        main_logger.info(f"Judge LLM calls needed: {total_judge}")
        main_logger.info(f"Worst-case rewrite calls: {total_judge}")
        main_logger.info(f"Total LLM calls (upper bound): {total_judge * 2}")
        main_logger.info(f"Entities missing from corpus: {len(entities_missing)}")
        main_logger.info(f"Queries with missing entities: {len(queries_with_missing_entities)}")

        # Print 100 random heuristic-matched samples for manual inspection
        # if heuristic_samples:
        #     sample_size = min(100, len(heuristic_samples))
        #     sampled = random.sample(heuristic_samples, sample_size)
        #     main_logger.info(f"\n========== HEURISTIC MATCH SAMPLES ({sample_size}/{len(heuristic_samples)}) ==========")
        #     for i, s in enumerate(sampled):
        #         main_logger.info(
        #             f"\n--- Sample {i+1} ---\n"
        #             f"Entity: {s['entity_label']} | Property: {s['property_label']} | Value: {s['property_value']}\n"
        #             f"Passage ({s['passage_id']}): {s['passage_snippet']}..."
        #         )

def create_random_access_mapping(corpus, mapping_path):

    if Path(mapping_path).exists():
        main_logger.info("Loading title2idx mapping...")
        title2idx = read_json_from_file(mapping_path)
        main_logger.info(f"Loaded mapping for {len(title2idx)} entity titles.")
        return title2idx

    main_logger.info("Creating title2idx mapping...")

    title2idx = defaultdict(list)
    for idx, row in tqdm.tqdm(enumerate(corpus), total=len(corpus)):
        title2idx[row["title"]].append(idx)

    # Sort indices within each title for deterministic order
    title2idx = {title: sorted(indices) for title, indices in title2idx.items()}

    write_json_to_file(mapping_path, title2idx)
    main_logger.info(f"Created mapping for {len(title2idx)} entity titles.")

    return title2idx


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



async def main(args):

        # print argparse to log
        main_logger.info("========== STARTING QREL GENERATION ==========")
        main_logger.info("Arguments:")
        for arg, value in vars(args).items():
            main_logger.info(f"  {arg}: {value}")
        main_logger.info("")

        main_logger.info("Loading candidates...")
        candidates = read_json_from_file(args.query_candidates_path)
        main_logger.info(f"Total candidates loaded: {len(candidates)}")

        qrel_generator = QrelGenerator(args)

        # Dry-run mode: count expected LLM calls and exit
        if args.dry_run:
            qrel_generator.count_expected_llm_calls(candidates)
            return

        candidates_processed = 0
        pbar = tqdm.tqdm(desc="Entities processed")
        for idx, candidate in enumerate(candidates):

            candidate_logger, candidate_handler = create_candidate_logger(candidate_idx=idx, log_dir=args.log_file_dir)

            main_logger.info(f"---------- Processing candidate {idx+1}/{len(candidates)}: {candidate['id']} ----------")
            candidate_logger.info(f"---------- Processing candidate {idx+1}/{len(candidates)}: {candidate['id']} ----------")

            product_type = candidate['src_class']['label']
            candidate_class_id = candidate['id'].split('_', 1)[1]

            # Only process entities in the final filtered set
            final_entity_set = {
                (eid[0], eid[1]) for eid in candidate['filtered_entity_ids']
            }

            # Process aggregation_prop
            if candidate.get('aggregation_prop') is not None:
                agg_prop = candidate['aggregation_prop']
                filtered_evs = [ev for ev in agg_prop['list_of_entity_values']
                                if (ev['entity_id'], ev['entity_label']) in final_entity_set]
                await qrel_generator.get_query_relevance_judgements(
                    entity_values=filtered_evs,
                    property_label=agg_prop['label'],
                    property_datatype=agg_prop['datatype'],
                    product_type=product_type,
                    candidate_class_id=candidate_class_id,
                    candidate_logger=candidate_logger,
                    entity_progress_bar=pbar,
                )

            # Process ALL constraint_props (skip "Listed/Mentioned" — no value to check)
            for constraint_prop in candidate.get('constraint_props', []):
                if constraint_prop.get('reference') == "Listed/Mentioned":
                    continue
                filtered_evs = [ev for ev in constraint_prop['list_of_entity_values']
                                if (ev['entity_id'], ev['entity_label']) in final_entity_set]
                await qrel_generator.get_query_relevance_judgements(
                    entity_values=filtered_evs,
                    property_label=constraint_prop['label'],
                    property_datatype=constraint_prop['datatype'],
                    product_type=product_type,
                    candidate_class_id=candidate_class_id,
                    candidate_logger=candidate_logger,
                    entity_progress_bar=pbar,
                )

            candidate_logger.removeHandler(candidate_handler)
            candidate_handler.close()

            candidates_processed += 1

        main_logger.info("\n========== SUMMARY ==========")
        main_logger.info(f"Total candidates processed: {candidates_processed}/{len(candidates)}")
        main_logger.info(f"Total entity labels missing from corpus: {len(qrel_generator.missing_entity_labels)}")
        main_logger.info(f"Total LLM calls made: {qrel_generator.total_llm_calls}")
        main_logger.info(f"Total passages processed: {qrel_generator.total_passages_processed}")
        main_logger.info(f"Total YES-SAME passages: {qrel_generator.num_yes_same}")
        main_logger.info(f"Total YES-DIFFERENT passages: {qrel_generator.num_yes_different}")
        main_logger.info(f"Total entities with at least one YES-SAME passage: {qrel_generator.num_entities_yes_same}")
        main_logger.info(f"Total entities with at least one YES-DIFFERENT passage: {qrel_generator.num_entities_yes_different}")
        main_logger.info(f"Total entities with NO relevant passages: {qrel_generator.num_entities_no}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-candidates-path",
        type=str,
        required=True,
        help="Path to query candidates JSON file"
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
        help="Path to corpus JSONL file (e.g. corpus_e_commerce.jsonl)"
    )
    parser.add_argument(
        "--title2idx-path",
        type=str,
        required=True,
        help="Path to JSON file containing title->corpus_indices mapping. "
             "Created automatically on first run from the corpus."
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        required=True,
        help="Model to use for passage judging e.g gpt-4o-mini"
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for judge model"
    )
    parser.add_argument(
        "--rewrite-model",
        type=str,
        required=True,
        help="Model to use for passage rewriting e.g gpt-4o"
    )
    parser.add_argument(
        "--rewrite-temperature",
        type=float,
        default=0.7,
        help="Temperature for rewrite model"
    )
    parser.add_argument(
        "--max-concurrent-calls",
        type=int,
        default=20,
        help="Maximum number of concurrent LLM calls"
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=None,
        help="Maximum number of LLM judge calls to make before stopping (for testing)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count expected LLM calls without making any API calls, then exit."
    )
    args = parser.parse_args()


    log_dir = Path(args.log_file_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = log_dir / "extract_qrels.log"

    logging.basicConfig(
        filename=main_log_path,
        level=logging.INFO,
        format="%(message)s",
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    asyncio.run(main(args))


    """
    python extract_qrels.py \
        --query-candidates-path data/final/candidates/query_generation_candidates.json \
        --qrel-results-path data/final/qrels/qrel_results.jsonl \
        --passage-rewrites-path data/final/qrels/passage_rewrites.jsonl \
        --log-file-dir data/final/qrels/logs \
        --corpus-path data/final/corpus/corpus_e_commerce.jsonl \
        --title2idx-path data/final/corpus/title2idx.json \
        --judge-model gpt-4o-mini \
        --judge-temperature 0.0 \
        --rewrite-model gpt-5-mini \
        --rewrite-temperature 1

    # Test run (with limited LLM calls):
    python extract_qrels.py \
    --query-candidates-path data/final/candidates/query_generation_candidates.json \
    --qrel-results-path data/final/qrels/qrel_results.jsonl \
    --passage-rewrites-path data/final/qrels/passage_rewrites.jsonl \
    --log-file-dir data/final/qrels/logs \
    --corpus-path data/final/corpus/corpus_e_commerce.jsonl \
    --title2idx-path data/final/corpus/title2idx.json \
    --judge-model gpt-4o-mini \
    --judge-temperature 0.0 \
    --rewrite-model gpt-5-mini \
    --rewrite-temperature 1 \
    --max-llm-calls 100


    # Dry run (count LLM calls only):
    python extract_qrels.py \
        --query-candidates-path data/final/candidates/query_generation_candidates.json \
        --qrel-results-path data/final/qrels/qrel_results.jsonl \
        --passage-rewrites-path data/final/qrels/passage_rewrites.jsonl \
        --log-file-dir data/final/qrels/logs \
        --corpus-path data/final/corpus/corpus_e_commerce.jsonl \
        --title2idx-path data/final/corpus/title2idx.json \
        --judge-model gpt-4o-mini \
        --rewrite-model gpt-5-mini \
        --dry-run
    """