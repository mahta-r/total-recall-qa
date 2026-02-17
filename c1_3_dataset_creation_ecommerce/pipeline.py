import random
import json

from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
from collections import defaultdict

from io_utils import write_json_to_file
from data_utils import format_values_by_datatype, get_order_key
from candidate_selector import CandidateSelector



def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def process_categories(
    product_dir: str | Path,
    blog_dir: str | Path,
    metadata_dir: str | Path,
    feature_summary_path: str | Path,
):
    candidate_categories = []
    
    feature_summary = load_json(feature_summary_path)
    
    # create sort order keys for all OrderedString features in feature_summary
    for feature_name, category2values in feature_summary.items():
        for category, category_info in category2values.items():
            values = category_info["values"]
            datatype = category_info["datatype"]
            if datatype == "OrderedString":
                category_info["order_key"] = get_order_key(values)


    for product_path in tqdm(product_dir.glob("*.json"), disable=False):
        category_file_name = product_path.stem.split('-')[-1]
        blog_path = blog_dir / f"blog-{category_file_name}.json"
        metadata_path = metadata_dir / f"{category_file_name}.json"

        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for category: {category_file_name}")
        metadata = load_json(metadata_path)
        product_list = load_json(product_path)
        blogs = load_json(blog_path)
        

        category_name = metadata["category_name"]
        category_id = metadata["name_prefix"]
        category_record = {
            'id': category_id.strip('_'),
            'label': category_name,
            'instances': product_list,
            'instance_count': len(product_list),
            'candidate_properties': {}
        }
        
        all_category_features = [f["name"] for f in metadata["features"]]
        for product in product_list:
            for feature, value in product.items():
                if feature not in ["ID", "index"]:
                    assert feature in all_category_features
                    assert feature in feature_summary and category_name in feature_summary[feature]
            
        for feature in all_category_features:

            feature_datatype = feature_summary[feature][category_name]["datatype"]
            feature_values = feature_summary[feature][category_name]["values"]

            property_record = format_values_by_datatype(
                feature=feature,
                feature_datatype=feature_datatype,
                feature_values=feature_values,
                product_list=product_list,
            )

            if property_record is not None:
                category_record['candidate_properties'][feature] = property_record

        candidate_categories.append(category_record)

    
    candidate_selector = CandidateSelector(
        feature_summary=feature_summary,
        candidate_categories=candidate_categories,
        max_queries_per_category=20,
        max_queries_per_prop=20,
        min_queries_per_prop=3,
        min_entities_per_query=4,
        max_entities_per_query=50,
        max_constraints_per_query=4,
        max_prop_type_per_category=5,
        seed=42
    )
    query_records = candidate_selector.select_class_property_pairs()

    return query_records



if __name__ == "__main__":
    
    base_dir = "/work/pi_hzamani_umass_edu/mahta/total-recall-rag/c1_3_dataset_creation_zahra/data/generated-data-v2"
    
    query_records = process_categories(
        product_dir=Path(f"{base_dir}/tables"),
        blog_dir=Path(f"{base_dir}/blogs"),
        metadata_dir=Path(f"{base_dir}/metadata-for-tables"),
        feature_summary_path=Path(f"{base_dir}/feature_summary_datatypes.json"),
    )

    print(f"\nTotal query records generated: {len(query_records)}")

    out_dir = "/work/pi_hzamani_umass_edu/mahta/total-recall-rag/c1_3_dataset_creation_zahra/data/final/candidates"
    output_path = Path(f"{out_dir}/query_generation_candidates.json")

    write_json_to_file(output_path, query_records)

    print(f"Query records written to: {output_path}")
