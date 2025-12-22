import json


def write_json_to_file(file_path, json_obj):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=False)


def write_jsonl_to_file(file_path, jsonl_obj_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for json_obj in jsonl_obj_list:
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')


def read_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def read_jsonl_from_file(file_path):
    jsonl_obj_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            jsonl_obj_list.append(json.loads(line))
    return jsonl_obj_list

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    

