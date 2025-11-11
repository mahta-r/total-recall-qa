import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):       
        if 'title' in doc_item:
            text = doc_item['contents']
            title = doc_item['title']
        else:
            content = doc_item['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            
        format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
    return format_reference