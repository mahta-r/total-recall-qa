# QALD10 Dataset Augmentation Pipeline

This directory contains a complete pipeline for augmenting QALD10 dataset with Total Recall queries.

## Pipeline Overview

The pipeline consists of 3 sequential steps:

```
Input JSONL → [Step 1] → [Step 2] → [Step 3] → Total Recall Queries
```

**Input**: `corpus_datasets/qald_aggregation_samples/wikidata_aggregation.jsonl`
**Output Directory**: `corpus_datasets/dataset_creation_heydar/qald10/`

### Step 1: Get Annotations
**Script**: [1_get_annotation.py](1_get_annotation.py)

- Processes SPARQL queries from the input dataset
- Extracts main entities and properties
- Generates intermediate query results
- Identifies entity types using Wikidata
- Updates answers by re-running SPARQL queries

**Outputs** (in `corpus_datasets/dataset_creation_heydar/qald10/`):
- `wikidata_totallist.jsonl` - Annotated dataset with entities and properties
- `entity_types_mapping.jsonl` - Mapping of entity types

### Step 2: Get Properties
**Script**: [2_get_properties.py](2_get_properties.py)

- Queries Wikidata for aggregatable properties
- Filters properties by datatype (Quantity, Time, GlobeCoordinate)
- Applies quality filters to exclude internal/non-aggregatable properties
- Finds properties shared by all intermediate entities

**Output** (in `corpus_datasets/dataset_creation_heydar/qald10/`):
- `wikidata_totallist_with_properties.jsonl` - Dataset with aggregatable properties

### Step 3: Generate Queries
**Script**: [3_query_generation.py](3_query_generation.py)

- Finds valid (entities, property) pairs where all entities have values
- Generates Total Recall queries using LLM
- Calculates answers based on aggregation functions
- Supports resume functionality for long runs

**Output** (in `corpus_datasets/dataset_creation_heydar/qald10/`):
- `wikidata_total_recall_queries.jsonl` - Final generated queries with answers

## Quick Start

### Run Complete Pipeline

```bash
# Set your API key
export OPENAI_API_KEY="your-openrouter-api-key"

# Run all 3 steps with GPT-4o (default)
python run_pipeline.py

# Or specify a different model
python run_pipeline.py --model anthropic/claude-3.5-sonnet
```

### Use Different Models

```bash
# Use Claude 3.5 Sonnet
python run_pipeline.py --model anthropic/claude-3.5-sonnet

# Use GPT-4o Mini (faster/cheaper)
python run_pipeline.py --model openai/gpt-4o-mini
```

## Pipeline Runner Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model MODEL` | Model to use for LLM steps | openai/gpt-4o |

The pipeline runs all 3 steps sequentially. Step 3 will automatically resume from the last processed QID if interrupted.

## Running Individual Steps

You can still run individual scripts directly:

### Step 1
```bash
python 1_get_annotation.py --model_name_or_path openai/gpt-4o
```

### Step 2
```bash
python 2_get_properties.py \
  --dataset_file corpus_datasets/dataset_creation_heydar/qald10/wikidata_totallist.jsonl \
  --output_file corpus_datasets/dataset_creation_heydar/qald10/wikidata_totallist_with_properties.jsonl
```

### Step 3
```bash
python 3_query_generation.py \
  --dataset_file corpus_datasets/dataset_creation_heydar/qald10/wikidata_totallist_with_properties.jsonl \
  --output_file corpus_datasets/dataset_creation_heydar/qald10/wikidata_total_recall_queries.jsonl \
  --prompt_template_path prompts/query_generation_v1.txt \
  --model_name_or_path openai/gpt-4o \
  --resume
```

## File Structure

```
c1_2_dataset_creation_heydar/qald10/
├── run_pipeline.py              # Main pipeline orchestrator
├── 1_get_annotation.py          # Step 1: Extract annotations
├── 2_get_properties.py          # Step 2: Get aggregatable properties
├── 3_query_generation.py        # Step 3: Generate Total Recall queries
├── src/
│   └── prompt_templetes.py      # Prompt templates for LLM
├── utils/
│   ├── general_utils.py         # General utilities
│   └── io_utils.py              # I/O utilities
├── prompts/
│   └── query_generation_v1.txt  # Prompt for query generation
├── files/                       # Additional files
└── README.md                    # This file

corpus_datasets/dataset_creation_heydar/qald10/  # Output directory
├── wikidata_totallist.jsonl
├── entity_types_mapping.jsonl
├── wikidata_totallist_with_properties.jsonl
└── wikidata_total_recall_queries.jsonl
```

## Requirements

- Python 3.7+
- OpenRouter API key (for OpenAI and Anthropic models)
- Required packages:
  - `openai`
  - `requests`
  - `SPARQLWrapper`
  - `tqdm`

## Environment Variables

```bash
export OPENAI_API_KEY="your-openrouter-api-key"
```

Note: Despite the name, this works with OpenRouter which provides access to multiple models including GPT-4o and Claude.

## Output Files

All output files are JSONL format (one JSON object per line):

1. **wikidata_totallist.jsonl**: Contains annotations with:
   - Original query and answer
   - Main entities and properties
   - Intermediate QIDs and their entity types

2. **wikidata_totallist_with_properties.jsonl**: Adds:
   - Aggregatable properties for each query
   - Property metadata (ID, label, datatype)

3. **wikidata_total_recall_queries.jsonl**: Final output with:
   - Generated Total Recall query
   - Aggregation function
   - Calculated answer
   - Property values for all entities

## Notes

- Step 1 and 3 use LLM and may take longer
- Step 2 queries Wikidata extensively (includes rate limiting)
- Step 3 supports resume to handle interruptions
- All steps include error handling and progress logging

## Troubleshooting

### "OPENAI_API_KEY environment variable not set"
Set your API key: `export OPENAI_API_KEY="your-key"`

### "Input file not found"
Check the input file path. Default is `corpus_datasets/qald_aggregation_samples/wikidata_aggregation.jsonl`

### "Missing required input files for step N"
Run previous steps first, or use `--start-from` to specify a different starting point

### Rate limiting from Wikidata
The pipeline includes delays between requests. If you still hit rate limits, increase the delays in step 2 and 3.
