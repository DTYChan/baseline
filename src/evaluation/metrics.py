"""
Evaluation metrics for RAG system assessment.

This module provides evaluation metrics for assessing the performance of
retrieval-augmented generation systems, including answer relevancy,
context relevancy, and query-document relevance.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator, 
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)


@dataclass
class EvaluationResult:
    """
    Container for evaluation results.
    
    Attributes:
        metric_name (str): Name of the evaluation metric
        score (float): Evaluation score
        feedback (str): Feedback or explanation for the score
        query (str): The query that was evaluated
        response (str): The response that was evaluated
        contexts (List[str]): The contexts that were evaluated (if applicable)
        metadata (Dict[str, Any]): Additional metadata
    """
    metric_name: str
    score: float
    feedback: str
    query: str
    response: Optional[str] = None
    contexts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    

class RAGEvaluator:
    """
    Comprehensive evaluator for RAG systems.
    
    This class provides methods to evaluate various aspects of a RAG system,
    including answer relevancy, context relevancy, and faithfulness.
    """
    
    def __init__(self, llm):
        """
        Initialize the RAG evaluator.
        
        Args:
            llm: The language model to use for evaluation
        """
        self.llm = llm
        self.answer_evaluator = AnswerRelevancyEvaluator(llm)
        self.context_evaluator = ContextRelevancyEvaluator(llm)
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm)
        self.relevancy_evaluator = RelevancyEvaluator(llm)
        
    def evaluate_answer_relevancy(self, query: str, response: str) -> EvaluationResult:
        """
        Evaluate the relevancy of an answer to a query.
        
        Args:
            query (str): The query
            response (str): The response to evaluate
            
        Returns:
            EvaluationResult: The evaluation result
        """
        result = self.answer_evaluator.evaluate(query=query, response=response)
        return EvaluationResult(
            metric_name="answer_relevancy",
            score=result.score,
            feedback=result.feedback,
            query=result.query,
            response=response
        )
        
    def evaluate_context_relevancy(self, query: str, contexts: List[str]) -> EvaluationResult:
        """
        Evaluate the relevancy of contexts to a query.
        
        Args:
            query (str): The query
            contexts (List[str]): The contexts to evaluate
            
        Returns:
            EvaluationResult: The evaluation result
        """
        result = self.context_evaluator.evaluate(query=query, contexts=contexts)
        return EvaluationResult(
            metric_name="context_relevancy",
            score=result.score,
            feedback=result.feedback,
            query=result.query,
            contexts=contexts
        )
        
    def evaluate_faithfulness(self, query: str, response: str, contexts: List[str]) -> EvaluationResult:
        """
        Evaluate the faithfulness of a response to the provided contexts.
        
        Args:
            query (str): The query
            response (str): The response to evaluate
            contexts (List[str]): The contexts used to generate the response
            
        Returns:
            EvaluationResult: The evaluation result
        """
        result = self.faithfulness_evaluator.evaluate(
            query=query, response=response, contexts=contexts
        )
        return EvaluationResult(
            metric_name="faithfulness",
            score=result.score,
            feedback=result.feedback,
            query=result.query,
            response=response,
            contexts=contexts
        )
        
    def evaluate_query_relevancy(self, query: str, text: str) -> EvaluationResult:
        """
        Evaluate the relevancy of a document to a query.
        
        Args:
            query (str): The query
            text (str): The document text to evaluate
            
        Returns:
            EvaluationResult: The evaluation result
        """
        result = self.relevancy_evaluator.evaluate(query=query, text=text)
        return EvaluationResult(
            metric_name="query_relevancy",
            score=result.score,
            feedback=result.feedback,
            query=result.query,
            contexts=[text]
        )
        
    def evaluate_complete(self, query: str, response: str, contexts: List[str]) -> Dict[str, EvaluationResult]:
        """
        Run all evaluations on a query, response, and contexts.
        
        Args:
            query (str): The query
            response (str): The response to evaluate
            contexts (List[str]): The contexts used to generate the response
            
        Returns:
            Dict[str, EvaluationResult]: Dictionary of evaluation results
        """
        results = {
            "answer_relevancy": self.evaluate_answer_relevancy(query, response),
            "context_relevancy": self.evaluate_context_relevancy(query, contexts),
            "faithfulness": self.evaluate_faithfulness(query, response, contexts)
        }
        
        # Also evaluate relevancy for each context individually
        for i, context in enumerate(contexts):
            results[f"context_{i}_relevancy"] = self.evaluate_query_relevancy(query, context)
            
        return results
        
    def format_results_as_dataframe(self, results: Dict[str, EvaluationResult]) -> pd.DataFrame:
        """
        Format evaluation results as a pandas DataFrame.
        
        Args:
            results (Dict[str, EvaluationResult]): Dictionary of evaluation results
            
        Returns:
            pd.DataFrame: DataFrame containing the evaluation results
        """
        data = []
        for metric_name, result in results.items():
            data.append({
                "metric": metric_name,
                "score": result.score,
                "feedback": result.feedback
            })
            
        return pd.DataFrame(data) 