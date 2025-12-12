import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
from openai import OpenAI

from utils.io_utils import read_text_from_file


def query_generation(args):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    prompt_template = read_text_from_file(args.prompt_template_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.properties_file = "corpus_datasets/qald_aggregation_samples/properties_results.json"
    args.prompt_template_path = "c1_2_qald_dataset_augmentation/prompts/query_generation_v1.txt"
    
    query_generation(args)
    
    # python c1_2_qald_dataset_augmentation/3_query_generation.py