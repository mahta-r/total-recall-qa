# ===========================
# === Src: https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py
# ===========================
import json
import faiss
import torch
import warnings
import datasets
import numpy as np
from tqdm import tqdm
from typing import List
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModel
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast


MODEL2PATH = {
    "contriever": "facebook/contriever-msmarco",
    "dpr": "facebook/dpr-question_encoder-single-nq-base",
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "reasonir": 'reasonir/ReasonIR-8B'
}

MODEL2POOLING = {
    "contriever": "mean",
    "dpr": "pooler",
    "e5": "mean",
    "bge": "cls",
    "reasonir": 'mean'
}

def load_model(retrieval_method, model_path: str, use_fp16: bool = False):
    if retrieval_method == 'dpr':
        tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path)
        model = DPRQuestionEncoder.from_pretrained(model_path)
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

# def load_docs(corpus, doc_idxs):
#     results = [corpus[str(idx)] for idx in doc_idxs]
#     # results = [corpus[idx] for idx in doc_idxs]
#     return results

def load_docs(corpus, doc_idxs):
    # doc_idxs can be a NumPy array; make sure it's a plain list of ints
    doc_idxs = [int(i) for i in doc_idxs]

    # pandas DataFrame: select rows by position
    if hasattr(corpus, "iloc"):
        # returns a DataFrame of the selected rows
        return corpus.iloc[doc_idxs]

    # HuggingFace Dataset or list-like
    if isinstance(corpus, list):
        return [corpus[i] for i in doc_idxs]

    # dict keyed by integer ids
    if isinstance(corpus, dict):
        # only use this if your keys are ints that align with positions
        return [corpus[i] for i in doc_idxs]

    # Fallback: try positional access
    return [corpus[int(i)] for i in doc_idxs]

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(
            retrieval_method=self.model_name,
            model_path = self.model_path,
            use_fp16 = self.use_fp16
        )
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        name = self.model_name.lower()

        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in name:
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]
        if "bge" in name:
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]
        # DPR/Contriever: no prefixes

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # ----- forward + pooling -----
        model_cls = type(self.model).__name__

        if "t5" in model_cls.lower():
            # T5-based retrieval model: take first token of decoder output
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        
        elif "reasonir" in model_cls.lower() or "reasonir" in name:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                None,
                output.last_hidden_state,
                inputs['attention_mask'],
                self.pooling_method
            )
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        
        elif "dpr" in name or "dpr" in model_cls.lower(): 
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                None,
                None,
                self.pooling_method
            )
            # do NOT normalize for DPR   
        
        elif "contriever" in name or "contriever" in model_cls.lower():
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                None,
                output.last_hidden_state,
                inputs['attention_mask'],
                self.pooling_method
            )
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs['attention_mask'],
                self.pooling_method
            )
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retriever_name = config.retriever_name
        self.corpus_path = config.corpus_path
        self.topk = config.retrieval_topk
        self._docid_to_doc = None
        if config.retriever_name in ['bm25', 'rerank_l6', 'rerank_l12']:
            self.index_path = f"{config.index_dir}/bm25_index"
            self.pooling_method = None
            self.retrieval_model_path = "cross-encoder/ms-marco-MiniLM-L-6-v2" # "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L12-v2"
        else:
            self.index_path = f"{config.index_dir}/{config.retriever_name}_Flat.index"
            self.retrieval_model_path = MODEL2PATH[config.retriever_name]
            self.pooling_method = MODEL2POOLING[config.retriever_name]

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25(config.bm25_k1, config.bm25_b)
        
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class RerankRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        # Frist-stage
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25(config.bm25_k1, config.bm25_b)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
        # Second-stage
        self.cross_encoder = CrossEncoder(self.retrieval_model_path, max_length=config.retrieval_query_max_length)
    
    def set_topk(self, new_k):
        self.topk = new_k
      
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None
    
    def _rerank_documents(self, query, contents):
        query_doc_pairs = [(query, doc['contents']) for doc in contents]
        scores = self.cross_encoder.predict(query_doc_pairs)        
        reranked_docs = sorted(zip(scores, contents), key=lambda x: x[0], reverse=True)[:self.topk]
        scores, sorted_contents = zip(*reranked_docs)
        return list(sorted_contents), list(scores)
    
    def _search(self, query: str, num: int = None, return_score: bool = False):
        first_stage_num = 1000
        if num is None:
            num = self.topk
            
        # First-stage
        hits = self.searcher.search(query, first_stage_num)
        if len(hits) < 1:
            return ([], []) if return_score else []
        
        if len(hits) < first_stage_num:
            warnings.warn('Not enough documents retrieved for first-stage!')
        else:
            hits = hits[:first_stage_num]
        
        if self.contain_doc:
            # all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw()) for hit in hits]
        else:
            docids = [hit.docid for hit in hits]
            all_contents = load_docs(self.corpus, docids)
        
        # Second-stage
        if len(all_contents) > 0:
            results, scores = self._rerank_documents(query, all_contents)
        
        return (results, scores) if return_score else results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        pass

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        print('loading index ...')
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            print("Using FAISS with GPU ...")
            # --- Multi-GPUs: Only run with A100 (2 GPUs), H100 leads to an error
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

            # --- Single-GPU
            # device_id = torch.cuda.current_device()
            # print(f'Using faiss_gpu on process {device_id}...')
            # res = faiss.StandardGpuResources() # Get GPU resource for this device
            # co = faiss.GpuClonerOptions()
            # co.useFloat16 = True
            # self.index = faiss.index_cpu_to_gpu(res, device_id, self.index, co)

        print('loading corpus ...')
        self.corpus = load_corpus(self.corpus_path)
        print(self.corpus)
        print(self.corpus[0])
        self.encoder = Encoder(
            model_name = config.retriever_name,
            model_path = self.retrieval_model_path,
            pooling_method = self.pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config. retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()
        
        results = load_docs(self.corpus, idxs)
        return (results, scores) if return_score else results
        
        # results = load_docs(self.corpus, idxs)
        # if return_score:
        #     return results, scores.tolist()
        # else:
        #     return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

