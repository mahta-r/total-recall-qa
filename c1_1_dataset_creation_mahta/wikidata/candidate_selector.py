from collections import Counter, defaultdict
import json
import random
import math

from requests import session
from wikidata.prop_utils import NO_AGGREGATION_PROPS, NO_CONNECTING_PROPS, UNUSABLE_PROPS, prop_operation_mapping
from wikidata.operation_utils import coordinates_filter, numerical_filter, temporal_filter, items_filter



class CandidateSelector:

    def __init__(
        self, 
        candidate_classes,
        # prop_num, 
        # selection_strategy, 
        max_props_per_class = None,
        max_multihop_props_per_class = None,
        max_queries_per_prop = None,
        min_queries_per_prop = None,
        seed = 42,
    ):

        random.seed(seed)

        self.property_stats = defaultdict(dict)
        self.subclass_stats = defaultdict(dict)
        self.datatype_stats = defaultdict(int)
        self.selected_class2props = defaultdict(lambda: defaultdict(list))
        self.selected_props2class = defaultdict(lambda: defaultdict(list))

        self.candidate_classes = {subclass['id']: subclass for subclass in candidate_classes}
        self.max_props_per_class = max_props_per_class
        self.max_multihop_props_per_class = max_multihop_props_per_class
        self.max_queries_per_prop = max_queries_per_prop
        self.min_queries_per_prop = min_queries_per_prop

        for subclass in candidate_classes:
            self.get_all_available_stats(self.property_stats, self.subclass_stats, subclass, max_hops=1)

        # self.prop_num = prop_num
        # self.selection = selection_strategy
        # self.max_props = max_props

        # self.used_property_operation_combos = {}

        # all_props = [
        #     prop_id
        #     for subclass in self.query_generation_classes
        #     for prop_id in subclass['candidate_properties'].keys()
        # ]
        # self.all_props_available = Counter(all_props)

    # def __iter__(self):
    #     for subclass in self.query_generation_classes:
    #         properties = subclass['candidate_properties']
            
    #         if not properties:
    #             continue

    #         property_ids = self.select_properties(properties.keys())

    #         for prop_id in property_ids:
    #             selected_op = self.select_operation(prop_id)
    #             yield subclass, prop_id, selected_op


    # def select_properties(self, property_ids):
    #     assert len(property_ids) > 0

    #     if self.prop_num == "all":
    #         num_to_select = len(property_ids)
    #     elif self.prop_num == "log":
    #         num_to_select = math.floor(math.log(len(property_ids), 2)) + 1
    #     else:
    #         raise ValueError(f"Invalid property per subclass strategy: {self.prop_num}")

    #     num_to_select = min(num_to_select, self.max_props)

    #     if self.selection == "random":
    #         selected_properties = random.sample(property_ids, num_to_select)
    #     elif self.selection == "least":
    #         # TODO: optimize selection
    #         property_count = lambda prop_id: sum(self.used_property_operation_combos.get(prop_id, {}).values())
    #         property_scarcity = lambda prop_id: self.all_props_available[prop_id]
    #         selected_properties = sorted(property_ids, key=property_scarcity)[:num_to_select]
    #     else:
    #         raise ValueError(f"Invalid property selection strategy: {self.selection}")

    #     for prop_id in selected_properties:
    #         if prop_id not in self.used_property_operation_combos:
    #             self.used_property_operation_combos[prop_id] = {}

    #     return selected_properties

    
    # def select_operation(self, prop_id):
    #     valid_ops = prop_operation_mapping[prop_id]
    #     # property_operation_count = lambda op: self.used_property_operation_combos.get(prop_id, {}).get(op, 0)
    #     # selected_op = sorted(valid_ops, key=property_operation_count)[0]

    #     op_counts = [(op, self.used_property_operation_combos[prop_id].get(op, 0)) for op in valid_ops]
    #     op_counts_sorted = sorted(op_counts, key=lambda x: x[1])
    #     min_count = op_counts_sorted[0][1]
    #     least_used_ops = [op for op, count in op_counts_sorted if count == min_count]
    #     selected_op = random.choice(least_used_ops)

    #     if selected_op not in self.used_property_operation_combos[prop_id]:
    #         self.used_property_operation_combos[prop_id][selected_op] = 0
    #     self.used_property_operation_combos[prop_id][selected_op] += 1

    #     return selected_op
    

    def filter_based_on_constraint(self, constraint_datatype, constraint_entity_values):
        if constraint_datatype == 'Quantity':
            values = [
                (entity_value['value_node']['value'], idx)
                for idx, entity_value in enumerate(constraint_entity_values)
            ]
            filtered_indices, reference_idx, operator = numerical_filter(values)
            filtered_ids = [constraint_entity_values[idx]['entity_id'] for idx in filtered_indices]
            reference_entity = constraint_entity_values[reference_idx]
    
        if constraint_datatype == 'GlobeCoordinate':
            coordinates = [
                (entity_value['value_node']['value'], idx)
                for idx, entity_value in enumerate(constraint_entity_values)
            ]
            filtered_indices, reference_idx, operator = coordinates_filter(coordinates)
            filtered_ids = [constraint_entity_values[idx]['entity_id'] for idx in filtered_indices]
            reference_entity = constraint_entity_values[reference_idx]

        if constraint_datatype == 'Time':
            datetimes = [
                (entity_value['value_node']['value'], idx)
                for idx, entity_value in enumerate(constraint_entity_values)
            ]
            filtered_indices, reference_idx, operator = temporal_filter(datetimes)
            filtered_ids = [constraint_entity_values[idx]['entity_id'] for idx in filtered_indices]
            reference_entity = constraint_entity_values[reference_idx]

        if constraint_datatype == 'WikibaseItem':
            items = [
                [entity_value['value_item_id'] for entity_value in entity_values_list['value_node']]
                for entity_values_list in constraint_entity_values   
            ]
            filtered_indices, reference_idx, operator = items_filter(items)
            
            if not filtered_indices:
                return None, None, None
            
            filtered_ids = [constraint_entity_values[idx]['entity_id'] for idx in filtered_indices]
            reference_entity = [
                constraint_entity_values[lst_idx]['value_node'][item_idx]
                for lst_idx, item_idx in reference_idx
            ]

        return filtered_ids, reference_entity, operator
        

    def get_all_available_stats(self, property_stats, subclass_stats, subclass, hops = [], max_hops = 1):
        if len(hops) > max_hops:
            return
        
        subclass_key = subclass['id']
        if subclass_key not in subclass_stats:
            subclass_stats[subclass_key] = {
                "label": subclass['label'],
                "direct": [],
                "multihop": {}
            }

        if 'candidate_properties' in subclass:
            for prop_id, prop in subclass['candidate_properties'].items():
                prop_key = prop_id
                if prop_key not in property_stats:
                    property_stats[prop_key] = {
                        "label": prop['property_info']['label'],
                        "datatype": prop['property_info']['datatype'],
                        "num_classes": 0,
                        "class_ids": []
                    }
                property_stats[prop_key]['num_classes'] += 1
                property_stats[prop_key]['class_ids'].append((*hops,(subclass_key,None)))
            
            subclass_stats[subclass_key]["direct"] = list(subclass['candidate_properties'].keys())
        
        if 'multihop_candidates' in subclass:
            for prop_id, prop in subclass['multihop_candidates'].items():

                hop_class = prop['new_class']
                hops.append((subclass_key, prop_id))
                self.get_all_available_stats(
                    property_stats, 
                    subclass_stats[subclass_key]["multihop"],
                    hop_class, 
                    hops, 
                    max_hops, 
                )
                hops.pop()        
        
        return
    

    def get_selected_stats(self):
        selected_property_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for prop_id, candidates in self.selected_props2class.items():
            selected_property_stats[prop_id]['label'] = self.property_stats[prop_id]['label']
            selected_property_stats[prop_id]['num_classes']['direct'] = len(candidates['direct'])
            selected_property_stats[prop_id]['num_classes']['multihop'] = len(candidates['multihop'])
            for candidate in (candidates['direct'] + candidates['multihop']):    
                if 'constraint_property' in candidate:
                    # FIXME candidate['constraint_property'][0]
                    selected_property_stats[candidate['constraint_property']]['num_classes']['constraint'] += 1
            
        for prop_id, candidates in self.selected_props2class.items():
            selected_property_stats[prop_id]['num_classes']['total'] = (
                    selected_property_stats[prop_id]['num_classes']['direct'] +
                    selected_property_stats[prop_id]['num_classes']['multihop'] +
                    selected_property_stats[prop_id]['num_classes']['constraint']
                )

        selected_class_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for class_id, candidates in self.selected_class2props.items():
            selected_class_stats[class_id]['label'] = self.subclass_stats[class_id]['label']
            selected_class_stats[class_id]['num_properties']['direct'] = len(candidates['direct'])
            selected_class_stats[class_id]['num_properties']['multihop'] = len(candidates['multihop'])
            selected_class_stats[class_id]['num_properties']['total'] = (
                selected_class_stats[class_id]['num_properties']['direct'] +
                selected_class_stats[class_id]['num_properties']['multihop']
            )
        
        return selected_property_stats, selected_class_stats


    def current_queries_per_prop(self, prop_id):
        return (
            self.selected_props2class[prop_id]["direct"] + 
            self.selected_props2class[prop_id]["multihop"]
        )
    
    def current_queries_per_class(self, class_id):
        return (
            self.selected_class2props[class_id]["direct"] + 
            self.selected_class2props[class_id]["multihop"]
        )

    def current_multihop_queries_per_class(self, class_id):
        return self.selected_class2props[class_id]["multihop"]


    def add_selected_pair(self, prop_id, class_path):
        new_candidate = {
            "aggregation_property": prop_id,
            "class_path": class_path
        }
        if len(class_path) > 1:
            self.selected_class2props[class_path[0][0]]["multihop"].append(new_candidate)
            self.selected_props2class[prop_id]["multihop"].append(new_candidate)
        else:
            self.selected_class2props[class_path[0][0]]["direct"].append(new_candidate)
            self.selected_props2class[prop_id]["direct"].append(new_candidate)


    def _get_last_hop(self, class_path):
        hops = []
        src_class_id, connecting_property = class_path[0]
        src_class = self.candidate_classes[src_class_id]
        hops.append(src_class)
        if connecting_property is not None:
            hop_class = src_class['multihop_candidates'][connecting_property]['new_class']    
            hops.append(hop_class)
        return hops[-1]



    def select_class_property_pairs(self):
        # Pass 1: add properties in order of rareness (i.e., number of classes having them) to the classes
        properties_scarcity_low_to_high = sorted(
            self.property_stats.items(), 
            key=lambda x: x[1]['num_classes']
        )
        for prop_id, prop_stats in properties_scarcity_low_to_high:
            if prop_id in (UNUSABLE_PROPS + NO_AGGREGATION_PROPS):
                continue

            classes_having_prop = prop_stats['class_ids']
            available_classes = [
                candid_class for candid_class in classes_having_prop if (
                    # len(self.current_queries_per_class(candid_class[0][0])) < self.max_props_per_class
                    len(self.current_queries_per_class(candid_class[0][0])) < self.max_props_per_class and (
                        len(candid_class) == 1 or
                        len(self.current_multihop_queries_per_class(candid_class[0][0])) < self.max_multihop_props_per_class
                    ) and candid_class[0][1] not in (NO_CONNECTING_PROPS + UNUSABLE_PROPS)
                )
            ]
            if prop_stats['datatype'] == 'WikibaseItem':
                available_classes = [
                    candid_class for candid_class in available_classes
                    if self._get_last_hop(candid_class)['instance_count'] > 2
                ]

            # print(f"pass1: need {self.max_queries_per_prop}, have {len(available_classes)}")

            least_used_class = lambda class_path: len(self.current_queries_per_class(class_path[0][0]))
            has_most_rare_pops = lambda class_path: (
                len(self.current_queries_per_class(class_path[0][0])),
                sum(
                    class_path in self.property_stats[prop_id]['class_ids'] and
                    self.property_stats[prop_id]['num_classes'] == 1
                    for prop_id in self.property_stats
                )
            )

            def used_to_rare_ratio(class_path):
                num_rare_props_in_class = 2 * sum(
                    class_path in self.property_stats[prop_id]['class_ids'] and
                    self.property_stats[prop_id]['num_classes'] == 1
                    for prop_id in self.property_stats
                )
                total_props_selected_for_class = sum([
                    candidate['class_path'] == class_path 
                    for prop_id in self.property_stats 
                    for candidate in self.current_queries_per_prop(prop_id)
                ])
                if num_rare_props_in_class == 0:
                    return 1
                if total_props_selected_for_class == 0:
                    return 1
                return total_props_selected_for_class / num_rare_props_in_class


            for class_path in sorted(
                available_classes, 
                key = has_most_rare_pops
            )[:self.min_queries_per_prop]:
                if len(self.current_queries_per_class(class_path[0][0])) < self.max_props_per_class:
                    self.add_selected_pair(prop_id, class_path)
        
        # Pass 2: for properties used < min_queries_per_prop, add them as constraint to previous candidates
        properties_used_least_to_most = sorted(
            self.property_stats.items(), 
            key=lambda x: len(self.current_queries_per_prop(x[0]))
        )
        
        for prop_id, prop_stats in properties_used_least_to_most:
            if prop_id in UNUSABLE_PROPS or prop_id in NO_AGGREGATION_PROPS:
                continue
            
            total_used = len(self.current_queries_per_prop(prop_id))
            if total_used >= self.max_queries_per_prop:
                continue

            available_query_candidates = defaultdict(list)
            classes_having_prop = prop_stats['class_ids']
            for candid_class in classes_having_prop:
                last_hop = self._get_last_hop(candid_class)
                if last_hop['instance_count'] <= 2:
                    continue

                for query_candidate in self.current_queries_per_class(candid_class[0][0]):
                    if (
                        candid_class == query_candidate['class_path'] and 
                        prop_id != query_candidate['aggregation_property'] and 
                        "constraint_property" not in query_candidate
                    ):
                        # FIXME
                        available_query_candidates[candid_class].append(query_candidate) 
                        # available_query_candidates[candid_class].append((query_candidate,len(candid_class)-1)) 

            # FIXME
            # for candid_class in classes_having_prop:
            #     if len(candid_class) == 1: # first hop of a multihop
            #         for query_candidate in self.selected_class2props[candid_class[0][0]]["multihop"]:
            #             assert len(query_candidate['class_path']) > 1
            #             if (
            #                 candid_class[0] == query_candidate['class_path'][0] and 
            #                 prop_id != query_candidate['aggregation_property'] and 
            #                 "constraint_property" not in query_candidate
            #             ):
            #                 print("adding!")
            #                 available_query_candidates[candid_class].append((query_candidate,0))

            # to keep balance between queries for a class with/without constraints, pick classes with highest difference ratio
            def constraint_diff_ratio(class_path):
                num_with_constraint = sum([
                    "constraint_property" in query_candidate
                    for query_candidate in self.current_queries_per_class(class_path[0][0])
                ])
                num_without_constraint = len(self.current_queries_per_class(class_path[0][0])) - num_with_constraint
                return abs(num_without_constraint - num_with_constraint) / (num_with_constraint + num_without_constraint)
            
            num_queries_to_add = self.max_queries_per_prop - total_used

            # print(f"pass2: need {num_queries_to_add}, have {len(available_query_candidates.keys())}, or {sum([len(v) for v in available_query_candidates.values()])} ")
            for candid_class in sorted(
                available_query_candidates.keys(), 
                key = lambda x: constraint_diff_ratio(x)
            )[:num_queries_to_add]:
                query_candidates = available_query_candidates[candid_class]
                # pick one of the available query candidates to add the constraint to
                
                # FIXME
                selected_query_candidate= random.choice(query_candidates)
                selected_query_candidate['constraint_property'] = prop_id
                # selected_query_candidate, constraint_hop = random.choice(query_candidates)
                # selected_query_candidate['constraint_property'] = (prop_id, constraint_hop)


    def select_candidates(self):
        # self.select_class_property_pairs()

        query_generation_candidates = []

        candidate_num = 0

        for class_id in self.selected_class2props:
            for query_type, query_candidates in self.selected_class2props[class_id].items():
                for query_candidate in query_candidates:
                    
                    candidate_num += 1
                    
                    query_generation_candidate = {}

                    src_class_id, connecting_prop_id = query_candidate['class_path'][0]
                    src_class = self.candidate_classes[src_class_id]
                    assert class_id == src_class['id']
                    assert class_id == src_class_id

                    # query_generation_candidate['src_class'] = src_class
                    query_generation_candidate['src_class'] = {
                        'id': src_class['id'],
                        'label': src_class['label'],
                        'description': src_class['description'],
                        'count': src_class['instance_count'],
                        'instances': src_class['instances'],
                    }

                    if query_type == 'direct':
                        assert len(query_candidate['class_path']) == 1
                        assert connecting_prop_id is None
                        
                        aggregation_class = src_class
                        aggregation_class_id = src_class_id
                    
                    elif query_type == 'multihop':
                        assert len(query_candidate['class_path']) == 2
                        assert connecting_prop_id is not None
                        
                        hop_class_id, next_hop_prop_id = query_candidate['class_path'][1]
                        connecting_prop = src_class['multihop_candidates'][connecting_prop_id]
                        hop_class = connecting_prop['new_class']
                        assert hop_class_id == hop_class['id']
                        assert next_hop_prop_id is None
                        
                        # query_generation_candidate['hop_class'] = hop_class
                        query_generation_candidate['hop_class'] = {
                            'id': hop_class['id'],
                            'label': hop_class['label'],
                            'description': hop_class['description'],
                            'count': hop_class['instance_count'],
                            'instances': hop_class['instances'],
                        }

                        # query_generation_candidate['connecting_property'] = connecting_prop
                        query_generation_candidate['connecting_prop'] = {
                            'id': connecting_prop['property_info']['property_id'],
                            'label': connecting_prop['property_info']['label'],
                            'description': connecting_prop['property_info']['description'],
                            'datatype': connecting_prop['property_info']['datatype'],
                            'shared_time': connecting_prop['shared_time'],
                            'list_of_entity_values': connecting_prop['entities_values'],
                        }
                        
                        aggregation_class = hop_class
                        aggregation_class_id = f"{src_class_id}-{query_generation_candidate['connecting_prop']['id']}-{hop_class['id']}"

                    aggregation_prop_id = query_candidate['aggregation_property']
                    aggregation_prop = aggregation_class['candidate_properties'][aggregation_prop_id]

                    candidate_id = f"{candidate_num}_{aggregation_class_id}_{aggregation_prop_id}"
                    
                    query_generation_candidate['aggregation_prop'] = {
                        'id': aggregation_prop['property_info']['property_id'],
                        'label': aggregation_prop['property_info']['label'],
                        'description': aggregation_prop['property_info']['description'],
                        'datatype': aggregation_prop['property_info']['datatype'],
                        'shared_time': aggregation_prop['shared_time'],
                        'list_of_entity_values': aggregation_prop['entities_values'],
                    }

                    if aggregation_prop['property_info']['datatype'] == 'WikibaseItem':
                        query_generation_candidate['aggregation_prop']['item_class'] = aggregation_prop['property_info']['item_class']
                    if aggregation_prop['property_info']['datatype'] == 'GlobeCoordinate':
                        query_generation_candidate['aggregation_prop']['globe'] = aggregation_prop['property_info']['globe']
                    if aggregation_prop['property_info']['datatype'] == 'Quantity':
                        query_generation_candidate['aggregation_prop']['unit'] = aggregation_prop['property_info']['unit']
                    if aggregation_prop['property_info']['datatype'] == 'Time':
                        query_generation_candidate['aggregation_prop']['precision'] = aggregation_prop['property_info']['precision']
                        query_generation_candidate['aggregation_prop']['calendar'] = aggregation_prop['property_info']['calendar']


                    # pick all and filter later based on constraint, if any
                    final_entity_ids = [entity_value['entity_id'] for entity_value in aggregation_prop['entities_values']]
                    constraint_prop_id = query_candidate.get('constraint_property', None)
                
                    if constraint_prop_id:
                        assert aggregation_class['instance_count'] >= 3
                        assert len(query_generation_candidate['aggregation_prop']['list_of_entity_values']) >= 3
                    
                        constraint_prop = aggregation_class['candidate_properties'][constraint_prop_id]
                        
                        filtered_entity_ids, reference_entity, direction = self.filter_based_on_constraint(
                            constraint_prop['property_info']['datatype'],
                            constraint_prop['entities_values']
                        )
                        if filtered_entity_ids is not None:
                            query_generation_candidate['constraint_prop'] = {
                                'id': constraint_prop['property_info']['property_id'],
                                'label': constraint_prop['property_info']['label'],
                                'description': constraint_prop['property_info']['description'],
                                'datatype': constraint_prop['property_info']['datatype'],
                                'shared_time': constraint_prop['shared_time'],
                                'list_of_entity_values': constraint_prop['entities_values'],
                                'reference_entity': reference_entity,
                                'direction': direction,
                            }

                            if constraint_prop['property_info']['datatype'] == 'WikibaseItem':
                                query_generation_candidate['constraint_prop']['item_class'] = constraint_prop['property_info']['item_class']
                            if constraint_prop['property_info']['datatype'] == 'GlobeCoordinate':
                                query_generation_candidate['constraint_prop']['globe'] = constraint_prop['property_info']['globe']
                            if constraint_prop['property_info']['datatype'] == 'Quantity':
                                query_generation_candidate['constraint_prop']['unit'] = constraint_prop['property_info']['unit']
                            if constraint_prop['property_info']['datatype'] == 'Time':
                                query_generation_candidate['constraint_prop']['precision'] = constraint_prop['property_info']['precision']
                                query_generation_candidate['constraint_prop']['calendar'] = constraint_prop['property_info']['calendar']
                            
                            final_entity_ids = filtered_entity_ids

                            candidate_id += f"-{constraint_prop_id}"
                    
                    assert len(final_entity_ids) >= 2
                    query_generation_candidate['filtered_entity_ids'] = final_entity_ids
                    query_generation_candidate['id'] = candidate_id

                    query_generation_candidates.append(query_generation_candidate)


                    print(f"-------------- Candidate ID: {query_generation_candidate['id']} --------------")
                    print(
                        "Class: {class_label} ({class_id}) | #{instance_count} | {class_description}"
                        .format(
                            class_label = query_generation_candidate['src_class']['label'],
                            class_id = query_generation_candidate['src_class']['id'],
                            instance_count = query_generation_candidate['src_class']['count'],
                            class_description = query_generation_candidate['src_class']['description'],
                        )
                    )
                    if 'connecting_prop' in query_generation_candidate and 'hop_class' in query_generation_candidate:
                        print(
                            "Connecting Property: {prop_label} ({prop_id}) | {datatype} | {prop_description}"
                            .format(
                                prop_label = query_generation_candidate['connecting_prop']['label'],
                                prop_id = query_generation_candidate['connecting_prop']['id'],
                                datatype = query_generation_candidate['connecting_prop']['datatype'],
                                prop_description = query_generation_candidate['connecting_prop']['description'],
                            )
                        )
                        print(
                            "Hop Class: {class_label} ({class_id}) | #{instance_count} | {class_description}"
                            .format(
                                class_label = query_generation_candidate['hop_class']['label'],
                                class_id = query_generation_candidate['hop_class']['id'],
                                instance_count = query_generation_candidate['hop_class']['count'],
                                class_description = query_generation_candidate['hop_class']['description'],
                            )
                        )
                    if 'constraint_prop' in query_generation_candidate:
                        print(
                            "    Constraint Property: {constraint_prop_label} ({constraint_prop_id}) | {constraint_prop_datatype} | {constraint_prop_description}"
                            .format(
                                constraint_prop_label = query_generation_candidate['constraint_prop']['label'],
                                constraint_prop_id = query_generation_candidate['constraint_prop']['id'],
                                constraint_prop_datatype = query_generation_candidate['constraint_prop']['datatype'],
                                constraint_prop_description = query_generation_candidate['constraint_prop']['description'],
                            )
                        )
                        print("    At time: {constraint_shared_time}".format(
                            constraint_shared_time = query_generation_candidate['constraint_prop']['shared_time']
                        ))
                        print("    Reference entity: {reference_entity}".format(
                            reference_entity = query_generation_candidate['constraint_prop']['reference_entity']
                            )
                        )
                        print("    Direction: {direction}".format(
                            direction = query_generation_candidate['constraint_prop']['direction']
                            )
                        )
                    print(
                        "Property: {prop_label} ({prop_id}) [{prop_datatype}] --> {prop_description}"
                        .format(
                            prop_label = query_generation_candidate['aggregation_prop']['label'],
                            prop_id = query_generation_candidate['aggregation_prop']['id'],
                            prop_datatype = query_generation_candidate['aggregation_prop']['datatype'],
                            prop_description = query_generation_candidate['aggregation_prop']['description'],
                        )
                    )
                    if query_generation_candidate['aggregation_prop']['datatype'] == 'WikibaseItem':
                        print("    Item class: {item_class}".format(
                            item_class = query_generation_candidate['aggregation_prop']['item_class']
                        ))
                    print("At time: {shared_time}".format(
                            shared_time = query_generation_candidate['aggregation_prop']['shared_time']
                        ))
                    
                    for idx, entity_value in enumerate(sorted(
                        query_generation_candidate['aggregation_prop']['list_of_entity_values'], 
                        key=lambda x: x['entity_id']
                    )):
                        pass
                        bullet = "[YES]" if entity_value['entity_id'] in query_generation_candidate['filtered_entity_ids'] else "[NO]"
                        print(f"  {bullet} {entity_value['entity_label']} ({entity_value['entity_id']}) = {entity_value['value_node']}")
                        
                        if 'constraint_prop' in query_generation_candidate:

                            just_for_now = sorted(
                                query_generation_candidate['constraint_prop']['list_of_entity_values'], 
                                key=lambda x: x['entity_id']
                            )
                            assert just_for_now[idx]['entity_id'] == entity_value['entity_id']
                            print(f"    |__ Constraint value: {just_for_now[idx]['value_node']}")

                    print("-------------------------------------------------------------------")
        

        return query_generation_candidates
