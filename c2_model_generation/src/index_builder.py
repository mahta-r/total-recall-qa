import os
import json
import torch
import faiss
import shutil
import argparse
import warnings
import datasets
import subprocess
import numpy as np
from tqdm import tqdm
from typing import cast
from transformers import AutoTokenizer, AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast


# input_file = "corpus_datasets/enwiki_20251001.jsonl"
# output_file = "corpus_datasets/enwiki_20251001_1.jsonl"
# with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         if not line.strip():
#             continue  # skip empty lines
#         try:
#             item = json.loads(line)
#             # rename 'content' to 'contents'
#             if 'content' in item:
#                 item['contents'] = item.pop('content')
#             outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
#         except json.JSONDecodeError as e:
#             print(f"Skipping invalid line: {e}")


# == For testing the index ====
def subsample_corpus():
    num_rows = 500000
    input_file = "corpus_datasets/enwiki_20251001.jsonl"
    output_file = f"corpus_datasets/enwiki_20251001_{num_rows}.jsonl"
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i >= num_rows:
                break
            try:
                data = json.loads(line)
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write("\n")
            except json.JSONDecodeError:
                print(f"Skipping line {i}: invalid JSON")


# =============================
MODEL2POOLING = {
    "bm25": "",
    "contriever": "mean",
    "dpr": "pooler",
    "e5": "mean",
    "bge": "cls",
    "reasonir": 'mean'
}

MODEL2PATH = {
    "bm25": "",
    "contriever": "facebook/contriever-msmarco",
    "dpr": "facebook/dpr-ctx_encoder-single-nq-base", # msmarco-distilbert-base-v3
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "reasonir": 'reasonir/ReasonIR-8B'
}

def load_model(retrieval_method, model_path: str, use_fp16: bool = False):
    if retrieval_method == 'dpr':
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path)
        model = DPRContextEncoder.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
        
    return model, tokenizer

def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask = None,
        pooling_method = "mean"
    ):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        ### == More robust version if needed
        # if last_hidden_state is None:
        #     raise ValueError("Mean pooling requires last_hidden_state.")
        # # If no mask is provided, assume all tokens are valid
        # if attention_mask is None:
        #     # [B, L] of ones on same device/dtype as hidden states
        #     attention_mask = torch.ones(
        #         last_hidden_state.size()[:2],
        #         device=last_hidden_state.device,
        #         dtype=torch.long,
        #     )
        # # Expand mask and cast to hidden dtype to avoid fp16/bool issues
        # mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
        # # Zero out paddings
        # masked = last_hidden_state * mask
        # summed = masked.sum(dim=1)  # [B, D]
        # counts = mask.sum(dim=1).clamp(min=1e-9)  # avoid divide-by-zero
        # return summed / counts
        
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

class Index_Builder:
    r"""A tool class used to build an index used in retrieval."""
    def __init__(
            self, 
            retrieval_method,
            model_path,
            corpus_path,
            save_dir,
            max_length,
            batch_size,
            use_fp16,
            pooling_method,
            faiss_type=None,
            embedding_path=None,
            save_embedding=False,
            faiss_gpu=False
        ):
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else 'Flat'
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu

        self.gpu_num = torch.cuda.device_count()
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")
        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")
        self.corpus = load_corpus(self.corpus_path)
        print("Finish loading...")
    
    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content."""
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method."""
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Building BM25 index based on Pyserini library.
        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        # to use pyserini pipeline, we first need to place jsonl file in the folder 
        self.save_dir = os.path.join(self.save_dir, "bm25_index")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)

        shutil.copyfile(self.corpus_path, temp_file_path)
        print("Start building bm25 index...")
        pyserini_args = [
            "--collection", "JsonCollection",
            "--input", temp_dir,
            "--index", self.save_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw"
        ]
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)
        shutil.rmtree(temp_dir)
        
        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(
                embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype
        )
        length = all_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = all_embeddings[i: j]
        else:
            memmap[:] = all_embeddings

    def encode_all(self):
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.batch_size = self.batch_size * self.gpu_num

        all_embeddings = []

        for start_idx in tqdm(range(0, len(self.corpus), self.batch_size), desc='Inference Embeddings:'):

            if 'title' in self.corpus[0]:
                batch_data_title = self.corpus[start_idx:start_idx+self.batch_size]['title']
                batch_data_text = self.corpus[start_idx:start_idx+self.batch_size]['contents']
                batch_data = ['"' + title + '"\n' + text for title, text in zip(batch_data_title, batch_data_text)]
            else:
                batch_data = self.corpus[start_idx:start_idx+self.batch_size]['contents']
            
            if self.retrieval_method == "e5":
                batch_data = [f"passage: {doc}" for doc in batch_data]

            inputs = self.tokenizer(
                        batch_data,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length,
            ).to('cuda')
            inputs = {k: v.cuda() for k, v in inputs.items()}

            if "t5" in self.retrieval_method: #TODO: support encoder-only T5 model
                # T5-based retrieval model
                decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0], 1), dtype=torch.long).to(inputs['input_ids'].device)
                output = self.encoder(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
                embeddings = output.last_hidden_state[:, 0, :]
            
            elif "reasonir" in self.retrieval_method:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(
                    None,
                    output.last_hidden_state,
                    inputs['attention_mask'],
                    self.pooling_method
                )
            
            elif "dpr" in self.retrieval_method:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(
                    output.pooler_output,
                    None,
                    None,
                    self.pooling_method
                )
                # no normalization for DPR
            
            elif "contriever" in self.retrieval_method:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(
                    None,
                    output.last_hidden_state,
                    inputs['attention_mask'],
                    self.pooling_method
                )
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            
            else:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(
                    output.pooler_output, 
                    output.last_hidden_state, 
                    inputs['attention_mask'],
                    self.pooling_method
                )
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    
            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        """Obtain the representation of documents based on the embedding model(BERT-based) and 
        construct a faiss index.
        """
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")
        
        self.encoder, self.tokenizer = load_model(
            retrieval_method=self.retrieval_method,
            model_path = self.model_path,
            use_fp16 = self.use_fp16
        )
            
        if self.embedding_path is not None:
            hidden_size = self.encoder.config.hidden_size
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            all_embeddings = self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            
            del self.corpus

        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:       
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
        
            faiss_index.add(all_embeddings)
            
        # Free big arrays before serializing
        import gc
        try:
            del all_embeddings
        except NameError:
            pass
        gc.collect()

        # If you used GPU elsewhere:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass    
        
        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")

def main():
    parser = argparse.ArgumentParser(description = "Creating index...")

    # Basic parameters
    parser.add_argument('--retrieval_method', type=str, default='contriever', choices=[
        'bm25', 'contriever', 'dpr', 'e5', 'bge'
    ])
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/enwiki_20251001.jsonl')
    parser.add_argument('--save_dir', default= 'corpus_datasets/indices',type=str)
    
    # Parameters for building dense index
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--faiss_type', type=str, default='Flat')
    parser.add_argument('--embedding_path', type=str, default=None)
    parser.add_argument('--save_embedding', action='store_true', default=False)
    parser.add_argument('--use_fp16', action='store_true', default=False)
    parser.add_argument('--faiss_gpu', action='store_true', default=False)
    
    args = parser.parse_args()
    args.model_path = MODEL2PATH[args.retrieval_method]
    pooling_method = MODEL2POOLING[args.retrieval_method]

    index_builder = Index_Builder(
        retrieval_method = args.retrieval_method,
        model_path = args.model_path,
        corpus_path = args.corpus_path,
        save_dir = args.save_dir,
        max_length = args.max_length,
        batch_size = args.batch_size,
        use_fp16 = args.use_fp16,
        pooling_method = pooling_method,
        faiss_type = args.faiss_type,
        embedding_path = args.embedding_path,
        save_embedding = args.save_embedding,
        faiss_gpu = args.faiss_gpu
    )
    index_builder.build_index()


if __name__ == "__main__":
    # subsample_corpus()
    main()
    
    
# python c2_model_generation/src/index_builder.py














# def add_in_batches(index, x, bs=50_000):
#     n = x.shape[0]
#     for i in range(0, n, bs):
#         index.add(x[i:i+bs])
# add_in_batches(faiss_index, all_embeddings, bs=50_000)

# Free big arrays before serializing
# import gc
# try:
#     del all_embeddings
# except NameError:
#     pass
# gc.collect()

# # If you used GPU elsewhere:
# try:
#     import torch
#     torch.cuda.empty_cache()
# except Exception:
#     pass

# if self.have_contents:
#     shutil.copyfile(self.corpus_path, temp_file_path)
# else:
#     with open(temp_file_path, "w") as f:
#         for item in self.corpus:
#             f.write(json.dumps(item) + "\n")

# if args.pooling_method is None:
#     pooling_method = 'mean'
#     for k,v in MODEL2POOLING.items():
#         if k in args.retrieval_method.lower():
#             pooling_method = v
#             break
# else:
#     if args.pooling_method not in ['mean','cls','pooler']:
#         raise NotImplementedError
#     else:
#         pooling_method = args.pooling_method
