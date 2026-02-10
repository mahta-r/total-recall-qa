import json
import random
import copy
from collections import defaultdict


from operation_utils import (
    numerical_filter,
    temporal_filter,
    ordered_type_filter,
    type_filter,
    numerical_aggregation,
    temporal_aggregation,
    string_aggregation,
    count_aggregation,
    is_valid_for_values,
)
from prop_operation_mapping import PROP_OP_MAPPING


class CandidateSelector:

    def __init__(
        self,
        feature_summary,
        candidate_categories,
        max_queries_per_category = None,
        max_queries_per_prop = None,
        min_queries_per_prop = None,
        min_entities_per_query = None,
        max_entities_per_query = None,
        max_constraints_per_query = None,
        max_prop_type_per_category = None,
        seed = 42,
    ):

        random.seed(seed)
        self.operation_sampler = random.Random(seed)
        self.constraint_sampler = random.Random(seed + 1)

        self.max_queries_per_category = max_queries_per_category
        self.max_queries_per_prop = max_queries_per_prop
        self.min_queries_per_prop = min_queries_per_prop
        self.min_entities_per_query = min_entities_per_query
        self.max_entities_per_query = max_entities_per_query
        self.max_constraints_per_query = max_constraints_per_query
        self.max_prop_type_per_category = max_prop_type_per_category

        self.property_stats = defaultdict(dict)
        self.category_stats = defaultdict(list)
        self.datatype_stats = defaultdict(int)
        self.selected_category2props = defaultdict(lambda: defaultdict(list))
        self.selected_props2category = defaultdict(lambda: defaultdict(list))
        self.used_property_operation_combos = defaultdict(lambda: defaultdict(int))

        self.feature_summary = feature_summary
        self.candidate_categories = {category['label']: category for category in candidate_categories}

        self.log = open("candidate_selector_log.log", "w")
        
        self.get_all_available_stats()


    def __del__(self):
        self.log.close()


    def filter_based_on_constraint(self, constraint_label, constraint_datatype, constraint_entity_values, sort_order_key=None):
        if constraint_datatype == 'Quantity':
            values = [
                (entity_value['value_node']['operation_value'], idx)
                for idx, entity_value in enumerate(constraint_entity_values)
            ]
            filtered_indices, reference, operator = numerical_filter(values, rand=self.constraint_sampler)
            # filtered_indices = [idx for idx, _ in enumerate(constraint_entity_values)]
            # reference = constraint_label
            # operator = "MENTION"


        if constraint_datatype == 'Date':
            datetimes = [
                (entity_value['value_node']['operation_value'], idx)
                for idx, entity_value in enumerate(constraint_entity_values)
            ]
            filtered_indices, reference, operator = temporal_filter(datetimes, rand=self.constraint_sampler)
            # filtered_indices = [idx for idx, _ in enumerate(constraint_entity_values)]
            # reference = constraint_label
            # operator = "MENTION"

        
        if constraint_datatype == 'OrderedString':
            types = [
                (entity_value['value_node']['operation_value'], idx)
                for idx, entity_value in enumerate(constraint_entity_values)
            ]
            filtered_indices, reference, operator = ordered_type_filter(types, rand=self.constraint_sampler, sort_order_key=sort_order_key)

        
        if constraint_datatype == 'String':
            types = [
                (entity_value['value_node']['operation_value'], idx)
                for idx, entity_value in enumerate(constraint_entity_values)
            ]
            filtered_indices, reference, operator = type_filter(types, rand=self.constraint_sampler)


        if not filtered_indices:
            return None, None, None
        
        filtered_ids = [
            (constraint_entity_values[idx]['entity_id'], constraint_entity_values[idx]['entity_label']) 
            for idx in filtered_indices
        ]
        
        return filtered_ids, reference, operator


    def get_all_available_stats(self):
        for category in self.candidate_categories.values():
            category_key = category['label']
            if category_key not in self.category_stats:
                self.category_stats[category_key] = list(category['candidate_properties'].keys())

            if 'candidate_properties' in category:
                for prop_label, prop in category['candidate_properties'].items():
                    
                    if prop_label not in self.property_stats:
                        self.property_stats[prop_label] = {
                            "label": prop['label'],
                            "datatype": prop['datatype'],
                            "num_categories": 0,
                            "categories": []
                        }
                    
                    self.property_stats[prop_label]['num_categories'] += 1
                    self.property_stats[prop_label]['categories'].append(category_key)

                    datatype = prop['datatype']
                    self.datatype_stats[datatype] += 1

        print(len(self.property_stats))
        print(len(self.category_stats))
        print(json.dumps(self.datatype_stats, indent=2))

        for prop_label, prop_info in self.property_stats.items():
            if prop_info['datatype'] == 'Quantity':
                print(f"{prop_label}: {prop_info['num_categories']} categories")
        # print(json.dumps(self.property_stats, indent=2))


    def select_operation(self, agg_prop_label, agg_prop_datatype, input_entity_values):
        """Select and execute an aggregation operation for a property.

        Returns (operation, operation_args, final_answer) or None if no valid operation.
        """
        NUMERICAL_OPS = {"AVG","MEDIAN", "MIN", "MAX", "DIFFERENCE(MAX-MIN)","MOST_COMMON"}
        TEMPORAL_OPS = {"EARLIEST", "LATEST", "TIME_BETWEEN_FIRST_LAST"}
        STRING_OPS = {"PERCENTAGE"}
        DATATYPE_COMPATIBLE_OPS = {
            'Quantity': NUMERICAL_OPS,
            'Date': TEMPORAL_OPS,
            'String': STRING_OPS,
            'OrderedString': STRING_OPS,
        }
        compatible_ops = DATATYPE_COMPATIBLE_OPS.get(agg_prop_datatype, set())

        all_ops = PROP_OP_MAPPING[agg_prop_datatype][agg_prop_label]
        valid_ops = [op for op in all_ops if op in compatible_ops and is_valid_for_values(op, input_entity_values)]

        if not valid_ops:
            return None

        # Sort by diversity: least used for this property, then least used globally
        self.operation_sampler.shuffle(valid_ops)
        op_counts = [
            (
                op,
                self.used_property_operation_combos[agg_prop_label][op],
                sum(self.used_property_operation_combos[pid][op]
                    for pid in self.used_property_operation_combos),
            )
            for op in valid_ops
        ]
        op_counts_sorted = sorted(op_counts, key=lambda x: (x[1], x[2]))
        min_prop_count = op_counts_sorted[0][1]
        min_global_count = op_counts_sorted[0][2]
        least_used = [op for op, pc, gc in op_counts_sorted
                      if pc == min_prop_count and gc == min_global_count]
        selected_op = self.operation_sampler.choice(least_used)

        self.used_property_operation_combos[agg_prop_label][selected_op] += 1

        if agg_prop_datatype == 'Quantity':
            return numerical_aggregation(selected_op, input_entity_values, self.operation_sampler)
        elif agg_prop_datatype == 'Date':
            return temporal_aggregation(selected_op, input_entity_values, self.operation_sampler)
        elif agg_prop_datatype in ('String', 'OrderedString'):
            return string_aggregation(selected_op, input_entity_values, self.operation_sampler)
        else:
            raise ValueError(f"Unsupported aggregation datatype: {agg_prop_datatype}")


    def select_class_property_pairs(self):
        all_query_records = []
        candidate_num = 0

        for category_label, category in self.candidate_categories.items():
            # input("Press Enter for next category...")

            if category_label in [
                'Candy & Chocolate'
            ]:
                continue

            candidate_properties = category['candidate_properties']

            print(f"\n================== '{category_label}': {len(candidate_properties)} properties ==================")

            # analysis of property datatype distribution
            datatype_distribution = defaultdict(dict)
            for prop_label, prop_record in candidate_properties.items():
                datatype = prop_record['datatype']
                datatype_distribution[datatype][prop_label] = ', '.join(map(str, prop_record['possible_values']))
            
            print({datatype: len(props) for datatype, props in sorted(datatype_distribution.items())})

            num_queries = 0
            const_property_usage_counts = defaultdict(int)
            agg_property_usage_counts = defaultdict(int)
            # property_labels = list(candidate_properties.items())
            property_labels = list(
                (prop_label, prop_record['datatype']) 
                for prop_label, prop_record in candidate_properties.items()
            )

            max_attempts = self.max_queries_per_category * 5
            attempts = 0

            while num_queries < self.max_queries_per_category and attempts < max_attempts:
                attempts += 1
                # input("Press Enter for next attempt...")
                # print(f"-----> Attempt {attempts} for category '{category_label}' (current {num_queries} queries)")

                # create one new query candidate
                input_entities = [(product['index'],product['ID']) for product in category['instances']]
                query_constraints = []

                # ----------------- select constraints for query candidate -----------------

                while len(input_entities) >= self.min_entities_per_query:
                    if len(query_constraints) >= self.max_constraints_per_query:
                        break
                    
                    input_entity_set = set(input_entities)

                    # print(f"Input entity set size: {len(input_entity_set)}")

                    # filter to only string properties that are sufficiently shared across remaining input_entities
                    shared_property_labels = [
                        label for label, datatype in property_labels
                        if label not in {c['label'] for c in query_constraints}
                        and (
                            datatype in ['String', 'OrderedString']
                            or len(datatype_distribution['Quantity']) + len(datatype_distribution['Date']) > 2
                        )
                        and len({
                            (ev["entity_id"], ev["entity_label"]) for ev in candidate_properties[label]["entities_values"]
                            }.intersection(input_entity_set)
                        ) >= max(0.3 * len(input_entity_set), self.min_entities_per_query)
                    ]

                    # print(f"Shared properties: {', '.join(shared_property_labels)}")

                    if not shared_property_labels:
                        break

                    # select least used properties across category as constraints
                    self.constraint_sampler.shuffle(shared_property_labels)
                    shared_property_labels = sorted(shared_property_labels, key=lambda x: const_property_usage_counts[x])
                    constraint_prop_label = shared_property_labels[0]
                    constraint_prop = candidate_properties[constraint_prop_label]

                    # print(f"\n Selected constraint property: {constraint_prop_label} (datatype: {constraint_prop['datatype']})")

                    input_entity_values = [
                        entity_value for entity_value in constraint_prop['entities_values']
                        if (entity_value['entity_id'], entity_value['entity_label']) in input_entity_set
                    ]

                    # print(f"  -> {len(input_entity_values)} entity values for constraint property '{constraint_prop_label}' before applying constraint")
                    # assert len(input_entity_values) == len(input_entities)

                    filtered_entity_ids, reference, direction = self.filter_based_on_constraint(
                        constraint_prop_label,
                        constraint_prop['datatype'],
                        input_entity_values,
                        sort_order_key=self.feature_summary[constraint_prop_label][category_label].get('order_key', None)
                    )

                    if filtered_entity_ids is not None:
                        # print(f"  -> Constraint on '{constraint_prop_label}' with reference value '{reference}' and direction '{direction}' reduces entities from {len(input_entities)} to {len(filtered_entity_ids)}")

                        assert len(filtered_entity_ids) < len(input_entities)
                        # if len(filtered_entity_ids) < len(input_entities):
                        query_constraints.append({
                            'label': constraint_prop_label,
                            'datatype': constraint_prop['datatype'],
                            'reference': reference,
                            'constraint': direction,
                            'input_entity_values': copy.deepcopy(input_entity_values),
                            'filtered_entity_ids': copy.deepcopy(filtered_entity_ids),
                        })
                        input_entities = filtered_entity_ids
                        # else:
                            # assert len(filtered_entity_ids) == len(input_entity_values)
                    else:
                        # print(f"Could not apply constraint on property {constraint_prop_label} for category {category_label}")
                        input_entities = []

                # ----------------- select stopping point for constraints -----------------

                valid_final_constraints = [
                    idx for idx, constraint in enumerate(query_constraints)
                    if self.min_entities_per_query <= len(constraint['filtered_entity_ids']) <= self.max_entities_per_query
                ]

                if not valid_final_constraints:
                    # print(f"Discarding candidate because no valid constraint stopping point: {len(query_constraints)} constraints, but none reduce to between {self.min_entities_per_query} and {self.max_entities_per_query} entities")
                    continue

                
                # selected_final_constraint_idx = self.constraint_sampler.choice(valid_final_constraints)
                selected_final_constraint_idx = min(valid_final_constraints)
                selected_constraints = query_constraints[:selected_final_constraint_idx + 1]
                final_filtered_entity_ids = selected_constraints[-1]['filtered_entity_ids']

                # ----------------- select aggregation property for query candidate -----------------

                constraint_prop_labels = {c['label'] for c in selected_constraints}
                # constraint_prop_labels = {c['label'] for c in selected_constraints if c['datatype'] in ['String', 'OrderedString']}

                shared_property_labels = [
                    label for label,datatype in property_labels
                    if label not in constraint_prop_labels
                    # and set(final_filtered_entity_ids).issubset(
                    #     {ev["entity_id"] for ev in candidate_properties[label]["entities_values"]}
                    # )
                    and len({
                            (ev["entity_id"], ev["entity_label"]) for ev in candidate_properties[label]["entities_values"]
                            }.intersection(final_filtered_entity_ids)
                        ) >= max(0.7 * len(final_filtered_entity_ids), self.min_entities_per_query)
                ]

                aggregation_prop_label = None
                aggregation_prop = None
                use_count = False

                # Priority 1: Quantity or Date properties
                self.operation_sampler.shuffle(shared_property_labels)
                quantity_date_shared = sorted(
                    [
                        label for label in shared_property_labels
                        if candidate_properties[label]['datatype'] in ['Quantity', 'Date']
                        and agg_property_usage_counts[label] < self.max_prop_type_per_category
                    ],
                    key=lambda x: agg_property_usage_counts[x]
                )
                if quantity_date_shared:
                    aggregation_prop_label = quantity_date_shared[0]
                    aggregation_prop = candidate_properties[aggregation_prop_label]
                else:
                    if len(selected_constraints) >= 3:
                        # Priority 2: COUNT (property-less)
                        use_count = True
                    else:
                        # Priority 3: String/OrderedString for PERCENTAGE
                        self.operation_sampler.shuffle(shared_property_labels)
                        string_shared = sorted(
                            [
                                label for label in shared_property_labels
                                if candidate_properties[label]['datatype'] in ['OrderedString', 'String']
                                and agg_property_usage_counts[label] < self.max_prop_type_per_category
                            ],
                            key=lambda x: agg_property_usage_counts[x]
                        )
                        if string_shared:
                            aggregation_prop_label = string_shared[0]
                            aggregation_prop = candidate_properties[aggregation_prop_label]
                        else:
                            # no aggregation possible, skip
                            # print(f"No aggregation property available for candidate, skipping...")
                            continue

                # ----------------- execute aggregation -----------------

                if use_count:
                    agg_entity_values = [{'entity_id': eid[0], 'entity_label': eid[1]} for eid in final_filtered_entity_ids]
                    operation, operation_args, result = count_aggregation(agg_entity_values)
                else:
                    entities_with_values = {
                        (ev["entity_id"], ev['entity_label']) 
                        for ev in candidate_properties[aggregation_prop_label]["entities_values"]
                    }.intersection(final_filtered_entity_ids)

                    assert len(entities_with_values) >= self.min_entities_per_query, f"Not enough entities with values for aggregation property '{aggregation_prop_label}' after applying constraints: {len(entities_with_values)} entities with values, but min required is {self.min_entities_per_query}"

                    if len(entities_with_values) < len(final_filtered_entity_ids):
                        selected_constraints.append({
                            'label': aggregation_prop_label,
                            'datatype': aggregation_prop['datatype'],
                            'reference': "Listed/Mentioned",
                            'constraint': "IS",
                            'input_entity_values': copy.deepcopy(final_filtered_entity_ids),
                            'filtered_entity_ids': copy.deepcopy(list(entities_with_values)),
                        })
                        final_filtered_entity_ids = list(entities_with_values)


                    agg_entity_values = [
                        ev for ev in aggregation_prop['entities_values']
                        if (ev['entity_id'], ev['entity_label']) in set(final_filtered_entity_ids)
                    ]
                    agg_result = self.select_operation(
                        aggregation_prop_label,
                        aggregation_prop['datatype'],
                        agg_entity_values,
                    )

                    if agg_result is None:
                        # rollback operation combo count (select_operation increments before calling)
                        # but since it returned None, it never incremented — just skip
                        continue
                    operation, operation_args, result = agg_result
                    agg_property_usage_counts[aggregation_prop_label] += 1

                # update constraint usage counts only after aggregation succeeds
                for c in selected_constraints:
                    const_property_usage_counts[c['label']] += 1

                # ----------------- build query record -----------------

                candidate_num += 1
                num_queries += 1

                query_record = {
                    'id': f"{candidate_num}_{category['id']}",
                    'src_class': {
                        'label': category_label,
                        'count': category['instance_count'],
                        'instances': category['instances'],
                    },
                    'constraint_props': [
                        {
                            'label': c['label'],
                            'datatype': c['datatype'],
                            'list_of_entity_values': c['input_entity_values'],
                            'reference': c['reference'],
                            'constraint': c['constraint'],
                            **(
                                {'unit': candidate_properties[c['label']].get('unit')}
                                if c['datatype'] == 'Quantity' else {}
                            ),
                        }
                        for c in selected_constraints # TODO: only constraints that reduce count
                    ],
                    'aggregation_prop': (
                        {
                            'label': aggregation_prop_label,
                            'datatype': aggregation_prop['datatype'],
                            'list_of_entity_values': agg_entity_values,
                            **(
                                {'unit': aggregation_prop.get('unit')}
                                if aggregation_prop['datatype'] == 'Quantity' else {}
                            ),
                        }
                        if aggregation_prop is not None else None
                    ),
                    'filtered_entity_ids': final_filtered_entity_ids,
                    'operation': operation,
                    'operation_args': operation_args,
                    'final_answer': result,
                }
                all_query_records.append(query_record)

                # ---------------------------------- log selected candidate ----------------------------------

                # continue

                # print(
                #     f"\n\nQuery {num_queries}/{self.max_queries_per_category}: {category_label}, "
                #     f"{len(selected_constraints)} constraints: \n   "
                #     f"{'\n   '.join([f'{c['label']}, {c['constraint']}, {c['reference']}' for c in selected_constraints])}, "
                #     f"\nop={operation}, "
                #     f"agg_prop={aggregation_prop_label}, {operation_args}"
                #     f"\n#entities={len(final_filtered_entity_ids)}, "
                #     f"answer={result}"
                # )

                # continue

                print(f"-------------- Candidate ID: {query_record['id']} --------------", file=self.log)
                print(
                    "Class: {class_label} | #{instance_count}"
                    .format(
                        class_label=query_record['src_class']['label'],
                        instance_count=query_record['src_class']['count'],
                    ),
                    file=self.log
                )
                for ci, cp in enumerate(query_record['constraint_props']):
                    print(
                        "    Constraint {idx}: {label} [{datatype}] | {constraint} {reference}"
                        .format(
                            idx=ci + 1,
                            label=cp['label'],
                            datatype=cp['datatype'],
                            constraint=cp['constraint'],
                            reference=cp['reference'],
                        ),
                        file=self.log
                    )
                if query_record['aggregation_prop'] is not None:
                    print(
                        "Property: {prop_label} [{prop_datatype}]"
                        .format(
                            prop_label=query_record['aggregation_prop']['label'],
                            prop_datatype=query_record['aggregation_prop']['datatype'],
                        ),
                        file=self.log
                    )
                else:
                    print("Property: None (COUNT)", file=self.log)
                print(
                    "Operation: {operation}, {operation_args} ---> Final Answer: {final_answer}"
                    .format(
                        operation=query_record['operation'],
                        operation_args=query_record['operation_args'],
                        final_answer=query_record['final_answer'],
                    ),
                    file=self.log
                )
                agg_values = query_record['aggregation_prop']['list_of_entity_values'] if query_record['aggregation_prop'] is not None else []
                for entity_value in sorted(agg_values, key=lambda x: x['entity_id']):
                    bullet = "[YES]" if (
                        entity_value['entity_id'], entity_value['entity_label']
                    ) in query_record['filtered_entity_ids'] else "[NO]"
                    print(f"  {bullet} {entity_value['entity_label']} ({entity_value['entity_id']}) = {entity_value['value_node']}", file=self.log)
                print("-------------------------------------------------------------------", file=self.log)

            print(f"  >> {category_label}: {num_queries} queries generated in {attempts} attempts\n")

        return all_query_records
