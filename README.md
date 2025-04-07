# RAG-Enhanced LLM Question Answering System

This project implements a Retrieval-Augmented Generation (RAG) system using Qwen2.5-1.5B-Instruct as the base model, enhanced with QLoRA fine-tuning for efficient adaptation to specific domains.

## Key Features

- **QLoRA Fine-tuning**: Efficient parameter-efficient fine-tuning of Qwen2.5-1.5B-Instruct model
- **Hypothetical Document Embeddings (HyDE)**: Enhanced query transformation for better retrieval
- **Document Retrieval**: Vector-based semantic search using sentence transformer embeddings
- **Evaluation Framework**: Comprehensive evaluation using LlamaIndex metrics

## Project Structure

```
.
├── data/                      # Document corpus for retrieval
├── models/                    # Trained model checkpoints
├── notebooks/                 # Jupyter notebooks for demonstrations
│   └── demo.ipynb             # Interactive demo notebook
├── src/                       # Source code
│   ├── evaluation/            # Evaluation module
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation metrics implementation
│   ├── model/                 # Model module
│   │   ├── __init__.py
│   │   ├── interface.py       # Model interface for inference
│   │   └── qlora.py           # QLoRA implementation
│   ├── retrieval/             # Retrieval module
│   │   ├── __init__.py
│   │   ├── document_loader.py # Document loading and processing
│   │   └── hyde.py            # HyDE implementation
│   └── __init__.py
├── .gitignore                 # Git ignore file
├── config.py                  # Configuration settings
├── infer.py                   # Inference script
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
└── train.py                   # Training script
```

## Implementation Details

- Base LLM: Qwen2.5-1.5B-Instruct
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Fine-tuning Method: QLoRA (Quantized Low-Rank Adaptation)
- Retrieval Framework: LlamaIndex
- Query Enhancement: HyDE (Hypothetical Document Embeddings)

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python train.py
   ```

3. Run inference:
   ```
   python infer.py --query "Your question here" --use_hyde --evaluate
   ```

## Performance Evaluation

The system is evaluated on several metrics:
- Answer relevancy score: Measures how well the answer addresses the query
- Context relevancy score: Measures how relevant the retrieved contexts are to the query
- Faithfulness: Measures how well the answer is grounded in the retrieved contexts

## Interactive Demo

You can try the system interactively using the Jupyter notebook:
```
jupyter notebook notebooks/demo.ipynb
```

## Key Components

### QLoRA Fine-tuning

The system uses Quantized Low-Rank Adaptation (QLoRA) to efficiently fine-tune the Qwen2.5-1.5B-Instruct model. This approach:
- Quantizes the base model to 4-bit precision to reduce memory usage
- Applies Low-Rank Adaptation for parameter-efficient fine-tuning
- Preserves model performance while reducing computational requirements

### Hypothetical Document Embeddings (HyDE)

HyDE improves retrieval by:
1. Using the LLM to generate a hypothetical document that could answer the query
2. Using this hypothetical document for retrieval instead of (or alongside) the original query
3. Leveraging the LLM's knowledge to bridge the lexical gap between queries and documents

### Comprehensive Evaluation

The evaluation framework provides:
- Quantitative metrics for system performance
- Qualitative feedback for answer quality
- Comparative analysis between different retrieval approaches 