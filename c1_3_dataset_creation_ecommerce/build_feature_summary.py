import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def is_string_list_without_numbers(values):
    if not isinstance(values, list):
        return False
    for v in values:
        if not isinstance(v, str):
            return False
        if any(ch.isdigit() for ch in v):
            return False
    return True


def build_feature_summary(metadata_dir: str, output_path: str):

    metadata_dir = Path(metadata_dir)
    summary = defaultdict(dict)

    for path in tqdm(list(metadata_dir.glob("*.json"))):
        with open(path, "r") as f:
            data = json.load(f)

        category = data["category_name"]

        for feature in data["features"]:
            feature_name = feature["name"]
            values = feature["values"]

            # ---- datatype decision ----
            if is_string_list_without_numbers(values):
                datatype = "String"
            else:
                print("\n--------------------------")
                print(f"Feature:   {feature_name}")
                print(f"Category:  {category}")
                print(f"Values:    {values}")
                answer = input("Datatype? (o=OrderedString, q=Quantity, s=String): ").strip()

                if answer == "o":
                    datatype = "OrderedString"
                elif answer == "q":
                    datatype = "Quantity"
                elif answer == "s":
                    datatype = "String"
                else:
                    datatype = f"MAHTARESPONSE: {answer}"

            summary[feature_name][category] = {
                "values": values,
                "datatype": datatype,
            }

    # ---- custom pretty printing (3-level indent, values inline) ----
    lines = ["{"]

    feature_items = list(summary.items())
    for fi, (feature_name, category_map) in enumerate(feature_items):
        lines.append(f'  "{feature_name}": {{')

        category_items = list(category_map.items())
        for ci, (category, payload) in enumerate(category_items):
            lines.append(f'    "{category}": {{')

            payload_items = list(payload.items())
            for pi, (k, v) in enumerate(payload_items):
                v_json = json.dumps(v, separators=(", ", ": "), ensure_ascii=False)
                comma = "," if pi < len(payload_items) - 1 else ""
                lines.append(f'      "{k}": {v_json}{comma}')

            cat_comma = "," if ci < len(category_items) - 1 else ""
            lines.append(f"    }}{cat_comma}")

        feature_comma = "," if fi < len(feature_items) - 1 else ""
        lines.append(f"  }}{feature_comma}")

    lines.append("}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return summary




build_feature_summary(
    metadata_dir="/work/pi_hzamani_umass_edu/mahta/total-recall-rag/c1_3_dataset_creation_zahra/data/generated-data-v2/metadata-for-tables",
    output_path="/work/pi_hzamani_umass_edu/mahta/total-recall-rag/c1_3_dataset_creation_zahra/data/generated-data-v2/feature_summary_datatypes.json",
)