"""
Inference script for RAG-Enhanced LLM Question Answering System.

This script provides a command-line interface for question answering using
the RAG-enhanced Qwen2.5 model.
"""

import argparse
from llama_index.core.settings import _Settings as Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.model.interface import load_model
from src.retrieval.document_loader import DocumentProcessor
from src.retrieval.hyde import create_hyde_query_engine
from src.evaluation.metrics import RAGEvaluator

from config import EMBEDDING_MODEL, DATA_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with RAG-enhanced Qwen2.5 model")
    
    parser.add_argument(
        "--query",
        type=str,
        default="What can we gain from studying at SJTU?",
        help="Query to answer"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help="Directory containing documents for retrieval"
    )
    
    parser.add_argument(
        "--use_hyde",
        action="store_true",
        help="Whether to use HyDE for query transformation"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to evaluate the response"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Size of text chunks for retrieval"
    )
    
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=20,
        help="Overlap between consecutive chunks"
    )
    
    return parser.parse_args()


def main():
    """Run the inference process."""
    args = parse_args()
    
    print(f"Initializing RAG-enhanced QA system...")
    
    # Load model
    llm = load_model()
    
    # Load embedding model
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    
    # Configure settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Initialize document processor
    doc_processor = DocumentProcessor(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Create index and query engine
    print(f"Loading and indexing documents from {args.data_dir}...")
    index = doc_processor.create_index()
    base_query_engine = index.as_query_engine()
    
    # Use HyDE if specified
    if args.use_hyde:
        print("Using HyDE for query transformation...")
        query_engine = create_hyde_query_engine(base_query_engine, llm)
    else:
        query_engine = base_query_engine
    
    # Run query
    print(f"\nQuestion: {args.query}")
    response = query_engine.query(args.query)
    
    # Print results
    print(f"\nAnswer: {response}")
    
    # Extract retrieved contexts
    retrieved_contexts = [node.get_content() for node in response.source_nodes]
    
    # Print source documents
    print("\nRetrieved contexts:")
    for i, context in enumerate(retrieved_contexts):
        print(f"\nContext {i+1}:\n{context[:200]}..." if len(context) > 200 else context)
    
    # Evaluate if specified
    if args.evaluate:
        print("\nEvaluating response...")
        evaluator = RAGEvaluator(llm)
        
        results = evaluator.evaluate_complete(
            query=args.query,
            response=str(response),
            contexts=retrieved_contexts
        )
        
        # Print evaluation results
        print("\nEvaluation results:")
        for metric_name, result in results.items():
            print(f"{metric_name}: {result.score:.2f} - {result.feedback}")
        
        # Format as dataframe
        df = evaluator.format_results_as_dataframe(results)
        print("\nSummary:")
        print(df)
    
    return response, retrieved_contexts


if __name__ == "__main__":
    main() 