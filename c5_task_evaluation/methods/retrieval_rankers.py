import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import transformers
from typing import List, Dict, Tuple

from c4_task_evaluation.src.retrieval_models_local import BM25Retriever, RerankRetriever, DenseRetriever
from c4_task_evaluation.src.llm_generator import LLMGenerator_api, LLMGenerator_hf_local


class BaseRetriever:
    """Base class for all retrieval models."""

    def __init__(self, device, args):
        self.args = args
        self.device = device

    def retrieve(self, query: str) -> Tuple[List[Dict], List[str]]:
        """
        Retrieve documents for a query.

        Args:
            query: The search query

        Returns:
            Tuple of (documents, document_ids) where:
                - documents: List of retrieved documents with metadata
                - document_ids: List of document IDs in ranked order
        """
        raise NotImplementedError


class SingleStepRetriever(BaseRetriever):
    """Single-step retrieval without any reasoning or iteration."""

    def __init__(self, device, args):
        super().__init__(device, args)

        # Initialize the appropriate retriever
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)
        else:
            raise ValueError(f"Unknown retriever: {args.retriever_name}")

    def retrieve(self, query: str) -> Tuple[List[Dict], List[str]]:
        """Perform single-step retrieval."""
        docs = self.retriever.search(query)
        doc_ids = [doc['id'] for doc in docs]
        return docs, doc_ids


class AgenticRetriever(BaseRetriever):
    """
    Agentic retrieval that uses an LLM to iteratively refine queries.
    This is a placeholder for more sophisticated agentic retrieval strategies.
    """

    def __init__(self, device, args):
        super().__init__(device, args)

        # Initialize retriever
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)
        else:
            raise ValueError(f"Unknown retriever: {args.retriever_name}")

        # Initialize LLM generator for query refinement
        if args.model_source == 'api':
            self.generator = LLMGenerator_api(args.model_name_or_path)
        elif args.model_source == 'hf_local':
            backbone_model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                dtype=torch.bfloat16
            ).to(device)
            backbone_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
            self.generator = LLMGenerator_hf_local(backbone_model, backbone_tokenizer, device, args)
        else:
            raise NotImplementedError(f"Unknown model source: {args.model_source}")

    def retrieve(self, query: str) -> Tuple[List[Dict], List[str]]:
        """
        Perform agentic retrieval with query refinement.

        TODO: Implement the actual agentic retrieval logic:
        1. Generate initial query/queries
        2. Retrieve documents
        3. Optionally refine query based on results
        4. Retrieve again if needed
        5. Rank and deduplicate results
        """
        # For now, just do single-step retrieval as placeholder
        docs = self.retriever.search(query)
        doc_ids = [doc['id'] for doc in docs]
        return docs, doc_ids


class RetrievalAgentWithReasoning(BaseRetriever):
    """
    Advanced retrieval agent that uses reasoning to decompose queries
    and perform multi-step retrieval.
    """

    def __init__(self, device, args):
        super().__init__(device, args)

        # Initialize retriever
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['contriever', 'dpr', 'e5', 'bge']:
            self.retriever = DenseRetriever(args)
        else:
            raise ValueError(f"Unknown retriever: {args.retriever_name}")

        # Initialize LLM generator
        if args.model_source == 'api':
            self.generator = LLMGenerator_api(args.model_name_or_path)
        elif args.model_source == 'hf_local':
            backbone_model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                dtype=torch.bfloat16
            ).to(device)
            backbone_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
            self.generator = LLMGenerator_hf_local(backbone_model, backbone_tokenizer, device, args)
        else:
            raise NotImplementedError(f"Unknown model source: {args.model_source}")

    def retrieve(self, query: str) -> Tuple[List[Dict], List[str]]:
        """
        Perform reasoning-based multi-step retrieval.

        TODO: Implement the full reasoning pipeline:
        1. Decompose complex query into sub-queries
        2. Retrieve for each sub-query
        3. Analyze retrieved documents
        4. Generate follow-up queries if needed
        5. Aggregate and rank all retrieved documents
        """
        # Placeholder: single-step retrieval
        docs = self.retriever.search(query)
        doc_ids = [doc['id'] for doc in docs]
        return docs, doc_ids


# Factory function to create retriever instances
def create_retriever(retriever_type: str, device, args) -> BaseRetriever:
    """
    Create a retriever instance based on the specified type.

    Args:
        retriever_type: Type of retriever ('single_step', 'agentic', 'reasoning')
        device: CUDA device
        args: Arguments namespace

    Returns:
        BaseRetriever instance
    """
    if retriever_type == 'single_step':
        return SingleStepRetriever(device, args)
    elif retriever_type == 'agentic':
        return AgenticRetriever(device, args)
    elif retriever_type == 'reasoning':
        return RetrievalAgentWithReasoning(device, args)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
