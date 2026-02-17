import datetime
import json


def encode_datetime(obj):
    if isinstance(obj, datetime.datetime):
        return {
            "__type__": "datetime",
            "value": obj.isoformat()
        }
    else:
        print(type(obj), obj)
        raise TypeError


def decode_datetime(obj):
    if isinstance(obj, dict) and obj.get("__type__") == "datetime":
        return datetime.datetime.fromisoformat(obj["value"])
    return obj


def write_json_to_file(file_path, json_obj, encoder_hook=encode_datetime):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=False, default=encoder_hook)


def write_jsonl_to_file(file_path, jsonl_obj_list, encoder_hook=encode_datetime):
    with open(file_path, 'w', encoding='utf-8') as f:
        for json_obj in jsonl_obj_list:
            f.write(json.dumps(json_obj, ensure_ascii=False, default=encoder_hook) + '\n')


def read_json_from_file(file_path, decoder_hook=decode_datetime):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f, object_hook=decoder_hook)

  
def read_jsonl_from_file(file_path, decoder_hook=decode_datetime):
    jsonl_obj_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            jsonl_obj_list.append(json.loads(line, object_hook=decoder_hook))
    return jsonl_obj_list


def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    

