# Total Recall QA: A Verifiable Evaluation Suite for Deep Research Agents

<!-- [Paper](LINK_TO_ARXIV) |  -->
<!-- [Hugging Face Dataset](https://huggingface.co/datasets/mahtaa/trqa) |  -->
<!-- [Evaluation](https://github.com/mahta-r/total-recall-qa/tree/main/c5_task_evaluation) -->

Total Recall QA (TRQA) is a benchmark designed to evaluate deep research systems on total-recall queries i.e. (question answering tasks where accurate generation of the answer requires retrieving **all relevant documents** for a given question from a large corpus, as well as reasoning and synthesizing information across all relevant documents). Unlike traditional QA benchmarks that reward partial retrieval, TRQA evaluates systems in settings where complete recall is necessary for correct reasoning.

The dataset consists of three subsets:
- **Wiki1**: Questions about encycolpedic knowledge from wikipedia, aggregating information over a complete set of entities (e.g all U.S. states)
- **Wiki2**: Questions built on top of [QALD-10](https://github.com/KGQA/QALD-10) and [QUEST](https://github.com/google-research/language/tree/master/language/quest) queries, aggregating information from the target entity sets of these queries.
- **Ecommerce**: Queries about a synthetically-generated E-commerce domain, asking about product specifiations/statistics in the dataset.

<!-- Each subset provides:
- validation queries
- validation qrels
- test queries
- test qrels
- corpus -->

<!-- --- -->

## Links
- [Hugging Face](https://huggingface.co/datasets/mahtaa/trqa)
- [Dataset Overview](#dataset-overview)
- [Getting Started](#getting-started)
- [Evaluation](#evaluation)
<!-- - [Citation](#citation) -->

<!-- --- -->


## Getting Started

### Installation

To load the dataset:

```bash
pip install datasets
```

<!-- --- -->

### Loading the Dataset

```python
from datasets import load_dataset

# Load test queries from wiki1
queries = load_dataset("mahtaa/trqa", "queries", split="wiki1_test")

# Load qrels
qrels = load_dataset("mahtaa/trqa", "qrels", split="wiki1_test")

# Load corpus (wiki1 and wiki2 share the same Wikipedia corpus)
corpus = load_dataset("mahtaa/trqa", "corpus", split="wiki")

# Load ecommerce data
ecom_queries = load_dataset("mahtaa/trqa", "queries", split="ecommerce_test")
ecom_corpus = load_dataset("mahtaa/trqa", "corpus", split="ecommerce")
```

<!-- --- -->

## Dataset Overview

The dataset contains three subsets:

| Subset     | Domain        | #Queries (Validation) | #Queries (Test) | Corpus Size  |
|------------|--------------|-----------------------|-----------------|--------------|
| wiki1      | Wikipedia     | 91                    | 169             | 57,745,780 (shared) |
| wiki2      | Wikipedia     | 1,083                 | 1,258           | 57,745,780 (shared) |
| ecommerce  | E-commerce    | 321                   | 900             | 3,282,927    |

The dataset on [Hugging Face](https://huggingface.co/datasets/mahtaa/trqa/) is organized into three configs, each with multiple splits:

| Config | Splits | Description |
|--------|--------|-------------|
| `queries` (default) | `wiki1_test`, `wiki1_validation`, `wiki2_test`, `wiki2_validation`, `ecommerce_test`, `ecommerce_validation` | Questions with single-response numerical answers |
| `qrels` | `wiki1_test`, `wiki1_validation`, `wiki2_test`, `wiki2_validation`, `ecommerce_test`, `ecommerce_validation` | Relevance judgments for passages |
| `corpus` | `wiki`, `ecommerce` | Passage collections |

Note: wiki1 and wiki2 share the same Wikipedia corpus (`wiki` split).

<!-- --- -->

### Data Format

#### Queries

Each line is a JSON object:

```json
{
  "id": "31_Q19598654_P8986-P7391",
  "question": "What is the median graph girth among Platonic graphs whose graph radius is less than or equal to 3?",
  "answer": 3.0
}
```

Fields:

- `id` — Query ID
- `question` — Total-recall query
- `answer` — Ground-truth numerical final answer 

---

#### Qrels 

Each line is a JSON object:

```json
{
  "query_id": "31_Q19598654_P8986-P7391",
  "iteration": "0",
  "doc_id": "30606-0001",
  "relevance": 1
}
```

Fields:

- `query_id` — Query ID
- `iteration` — Typically 0
- `doc_id` — Passage ID
- `relevance` — Relevance label (1 = relevant)

The qrels are converted from TREC format - each line in the TREC qrels is converted to a json line.

---

#### Corpus

Each line is a JSON object representing a passage:

```json
{
  "id": "30606-0001",
  "title": "Tetrahedron",
  "contents": "In the case of a tetrahedron, the base is a triangle (any of the four faces can be considered the base), so a tetrahedron is also known as a \"triangular pyramid\". The graph of a tetrahedron has shortest cycles of length 3.0..."
}
```

Fields:

- `id` — Passage ID. The format is `<document_id>-<chunk_id>`, where the number after the last dash is the chunk index within the original document (e.g., `30606-0001` is chunk `0001` of document `30606`).
- `title` — Passage title
- `contents` — Passage text

---

## Evaluation

Evaluation scripts are available in the `c5_task_evaluation/` directory of this repository.

Please refer to:

[EVALUATION README](LINK_TO_EVALUATION_README)

<!-- ---

## Citation

If you use TRQA in your research, please cite:

```bibtex
@inproceedings{trqa2026,
  title={Total Recall QA: A Verifiable Evaluation Suite for Deep Research Agents},
  author={TBD},
  booktitle={Proceedings of SIGIR 2026},
  year={2026}
}
``` -->