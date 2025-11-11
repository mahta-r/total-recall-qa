import json

def write_json_to_file(file_path, json_obj):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=False)

def read_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()