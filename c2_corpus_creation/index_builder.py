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
import importlib.util
from tqdm import tqdm
from typing import cast
from transformers import AutoTokenizer, AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast


# == For testing the index ====
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

# def subsample_corpus():
#     num_rows = 5000000
#     input_file = "/projects/0/prjs0834/heydars/INDICES/enwiki_20251001.jsonl"
#     # output_file = f"./enwiki_20251001_{num_rows}.jsonl"
#     output_file = f"/projects/0/prjs0834/heydars/INDICES/enwiki_20251001_{num_rows}.jsonl"
    
#     with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
#          open(output_file, 'w', encoding='utf-8') as outfile:
#         for i, line in enumerate(infile):
#             if i >= num_rows:
#                 break
#             try:
#                 data = json.loads(line)
#                 json.dump(data, outfile, ensure_ascii=False)
#                 outfile.write("\n")
#             except json.JSONDecodeError:
#                 print(f"Skipping line {i}: invalid JSON")

# def print_by_id(doc_id):
#     input_file = "/projects/0/prjs0834/heydars/INDICES/enwiki_20251001.jsonl"
#     with open(input_file, 'r', encoding='utf-8', errors='replace') as infile:
#         for line in infile:
#             try:
#                 data = json.loads(line)
#                 if data.get('id') == doc_id:
#                     print(json.dumps(data, indent=2, ensure_ascii=False))
#                     return
#             except json.JSONDecodeError:
#                 continue
#     print(f"Document with id {doc_id} not found.")
    


# =============================
MODEL2POOLING = {
    "bm25": "",
    "contriever": "mean",
    "dpr": "pooler",
    "e5": "mean",
    "bge": "cls",
    "reasonir": 'mean',
    "spladepp": None
}

MODEL2PATH = {
    "bm25": "",
    "contriever": "facebook/contriever-msmarco",
    "dpr": "facebook/dpr-ctx_encoder-single-nq-base", # msmarco-distilbert-base-v3
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "reasonir": 'reasonir/ReasonIR-8B',
    "spladepp": "naver/splade-cocondenser-ensembledistil"
}

SPLADE_FIELD_DELIMITER = "<<<PY_FIELD_SEPARATOR>>>"

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
        self.index_save_path = None
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

        self.index_save_path = self._resolve_index_path()
        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")
        if self.retrieval_method in {"bm25", "spladepp"}:
            self.corpus = None
            print(f"Skipping in-memory corpus load for {self.retrieval_method}.")
        else:
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

    @staticmethod
    def _ensure_module_available(module_name, extra_message=""):
        if importlib.util.find_spec(module_name) is None:
            hint = f" Please install it via `{extra_message}`." if extra_message else ""
            raise ModuleNotFoundError(f"Required module '{module_name}' is not installed.{hint}")

    def _resolve_index_path(self):
        if self.retrieval_method == "bm25":
            return os.path.join(self.save_dir, "bm25_index")
        if self.retrieval_method == "spladepp":
            return os.path.join(self.save_dir, "spladepp_index")
        return os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method."""
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        elif self.retrieval_method == "spladepp":
            self.build_spladepp_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Building BM25 index based on Pyserini library.
        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        index_dir = self.index_save_path
        os.makedirs(index_dir, exist_ok=True)
        temp_dir = os.path.join(index_dir, "temp")
        temp_file_path = os.path.join(temp_dir, "temp.jsonl")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        shutil.copyfile(self.corpus_path, temp_file_path)
        print("Start building bm25 index...")
        pyserini_args = [
            "--collection", "JsonCollection",
            "--input", temp_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw"
        ]
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args, check=True)
        shutil.rmtree(temp_dir)
        
        print(f"Finish! BM25 index stored at {index_dir}")

    def build_spladepp_index(self):
        """Build a SPLADE++ impact index using direct encoding."""
        if not self.model_path:
            raise ValueError("SPLADE++ requires a valid model path.")

        self._ensure_module_available("pyserini", "pip install pyserini[impact]")

        vector_dir = os.path.join(self.save_dir, f"{self.retrieval_method}_vectors")
        index_dir = self.index_save_path

        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir)
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)

        os.makedirs(vector_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        # Load SPLADE model directly
        print(f"Loading SPLADE model: {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModel.from_pretrained(self.model_path)
        model.eval()
        model.cuda()
        if self.use_fp16:
            model = model.half()

        # Encode corpus
        print(f"Encoding corpus with SPLADE++ (batch size: {self.batch_size})...")
        self._encode_splade_corpus(model, tokenizer, vector_dir)

        # Clean up model to free memory
        del model, tokenizer
        torch.cuda.empty_cache()

        # Build Lucene impact index
        cpu_threads = os.cpu_count() or 1
        cpu_threads = max(1, min(cpu_threads, 32))
        index_cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonVectorCollection",
            "--input", vector_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(cpu_threads),
            "--impact",
            "--pretokenized"
        ]

        print("Building Lucene impact index for SPLADE++ outputs...")
        subprocess.run(index_cmd, check=True)

        if not self.save_embedding and os.path.isdir(vector_dir):
            shutil.rmtree(vector_dir)
            print("Temporary SPLADE vector directory removed.")
        else:
            print(f"SPLADE vector dumps preserved at {vector_dir}")

        print(f"Finish! SPLADE++ index stored at {index_dir}")

    @torch.no_grad()
    def _encode_splade_corpus(self, model, tokenizer, output_dir):
        """Encode corpus with SPLADE model and save as Pyserini JsonVectorCollection format.
        Uses streaming to avoid loading entire corpus into memory.
        """
        batch_size = self.batch_size
        # Count total documents first (for progress bar)
        print("Counting documents...")
        total_docs = 0
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    total_docs += 1
        print(f"Total documents: {total_docs:,}")
        
        # Process in batches with streaming
        output_file = os.path.join(output_dir, "vectors.jsonl")
        
        with open(self.corpus_path, 'r', encoding='utf-8') as in_f, \
             open(output_file, 'w', encoding='utf-8') as out_f:
            
            batch_texts = []
            batch_ids = []
            doc_count = 0
            
            pbar = tqdm(total=total_docs, desc='Encoding')
            
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                
                doc = json.loads(line)
                
                # Extract text and ID
                text = doc.get('contents', '')
                doc_id = doc.get('id', str(doc_count))
                
                batch_texts.append(text)
                batch_ids.append(doc_id)
                doc_count += 1
                
                # Process batch when full
                if len(batch_texts) >= batch_size:
                    self._process_splade_batch(
                        batch_texts, batch_ids, model, tokenizer, out_f
                    )
                    pbar.update(len(batch_texts))
                    batch_texts = []
                    batch_ids = []
            
            # Process remaining documents
            if batch_texts:
                self._process_splade_batch(
                    batch_texts, batch_ids, model, tokenizer, out_f
                )
                pbar.update(len(batch_texts))
            
            pbar.close()
        
        print(f"Encoded vectors saved to {output_file}")
    
    def _process_splade_batch(self, batch_texts, batch_ids, model, tokenizer, output_file):
        """Process a batch of documents with SPLADE encoding."""
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Forward pass
        outputs = model(**inputs)
        
        # Get SPLADE representation
        # Standard SPLADE: log(1 + ReLU(hidden_states)) with max pooling
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, vocab_size]
        
        # Apply ReLU and log transformation
        relu_log = torch.log(1 + torch.relu(hidden_states))
        
        # Max pool over sequence dimension, considering attention mask
        # Shape: [batch, seq_len, vocab_size] -> [batch, vocab_size]
        mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand_as(relu_log)
        relu_log = relu_log * mask_expanded
        logits = torch.max(relu_log, dim=1)[0]
        
        # Convert to sparse representation
        for doc_id, doc_logits in zip(batch_ids, logits):
            # Get non-zero terms
            non_zero_indices = torch.nonzero(doc_logits > 0, as_tuple=True)[0]
            
            if len(non_zero_indices) > 0:
                # Get tokens and weights
                tokens = tokenizer.convert_ids_to_tokens(non_zero_indices.cpu().tolist())
                weights = doc_logits[non_zero_indices].cpu().tolist()
                
                # Create sparse vector dict
                vector = {token: float(weight) for token, weight in zip(tokens, weights)}
            else:
                vector = {}
            
            # Write in JsonVectorCollection format
            output_doc = {
                'id': doc_id,
                'contents': '',  # Empty contents for impact index
                'vector': vector
            }
            output_file.write(json.dumps(output_doc) + '\n')
        
        # Clean up
        del inputs, outputs, logits
        torch.cuda.empty_cache()

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
    parser.add_argument('--retrieval_method', type=str, default='spladepp', choices=['bm25', 'contriever', 'dpr', 'e5', 'bge', 'spladepp'])
    parser.add_argument('--corpus_path', type=str, default='corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl')
    parser.add_argument('--save_dir', default= '/projects/0/prjs0834/heydars/CORPUS_Mahta/indices',type=str)
    
    # Parameters for building dense index
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--faiss_type', type=str, default='Flat')
    parser.add_argument('--embedding_path', type=str, default=None)
    parser.add_argument('--save_embedding', action='store_true', default=True)
    parser.add_argument('--use_fp16', action='store_true', default=True)
    parser.add_argument('--faiss_gpu', action='store_true', default=False)
    
    args = parser.parse_args()
    args.model_path = MODEL2PATH[args.retrieval_method]
    pooling_method = MODEL2POOLING.get(args.retrieval_method)

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
    # print_by_id('307-0000')
    # subsample_corpus()
    main()
    
    
# python c2_corpus_creation/index_builder.py














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
