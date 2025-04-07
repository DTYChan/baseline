"""
HyDE (Hypothetical Document Embeddings) implementation for enhanced query transformation.

This module implements the HyDE approach which generates a hypothetical document
that could answer the query, then uses that document for retrieval instead of
the original query. This approach often improves retrieval performance by
bridging the lexical gap between queries and documents.
"""

import sys
import os
from typing import List, Optional, Union

from llama_index.core import QueryBundle
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.llms import ChatMessage, CompletionResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import HYDE_PROMPT


class EnhancedHyDEQueryTransform(HyDEQueryTransform):
    """
    Enhanced implementation of HyDE query transformation for improved retrieval.
    
    This class extends the base HyDEQueryTransform from llama_index with additional
    functionality and better integration with custom LLMs.
    """
    
    def __init__(self, llm, hyde_prompt=HYDE_PROMPT, include_original=True, **kwargs):
        """
        Initialize the EnhancedHyDEQueryTransform.
        
        Args:
            llm: The language model to use for generating hypothetical documents
            hyde_prompt (str): The prompt template for generating hypothetical documents
            include_original (bool): Whether to include the original query alongside the HyDE query
            **kwargs: Additional arguments to pass to HyDEQueryTransform
        """
        super().__init__(llm=llm, hyde_prompt=hyde_prompt, include_original=include_original)
        self._hyde_prompt = hyde_prompt
        self._include_original = include_original
        
    def _run(self, query_bundle: QueryBundle, **kwargs) -> Union[QueryBundle, List[QueryBundle]]:
        """
        Generate a hypothetical document and transform the query.
        
        Args:
            query_bundle: The original query bundle
            **kwargs: Additional arguments
            
        Returns:
            Union[QueryBundle, List[QueryBundle]]: The transformed query bundle(s)
        """
        query_str = query_bundle.query_str
        
        # Format the prompt with the query
        prompt = self._hyde_prompt.format(query_str=query_str)
        
        # Generate the hypothetical document
        response = self._llm.complete(prompt)
        
        # Extract the content from the response
        if hasattr(response, "message") and isinstance(response.message, ChatMessage):
            hyde_doc = response.message.content
        elif isinstance(response, CompletionResponse):
            hyde_doc = response.text
        else:
            hyde_doc = str(response)
            
        # Log the generated document
        print(f"Generated Hypothetical Document: {hyde_doc[:500]}..." 
              if len(hyde_doc) > 500 else hyde_doc)
            
        # Create a new query bundle with the hypothetical document
        hyde_query_bundle = QueryBundle(query_str=hyde_doc)
        
        # Return both the original and transformed query if include_original is True
        if self._include_original:
            return [query_bundle, hyde_query_bundle]
        
        return hyde_query_bundle


def create_hyde_query_engine(base_query_engine, llm, include_original=True):
    """
    Create a query engine that uses HyDE for improved retrieval.
    
    Args:
        base_query_engine: The base query engine to transform
        llm: The language model to use for generating hypothetical documents
        include_original (bool): Whether to include the original query
        
    Returns:
        TransformQueryEngine: A query engine that uses HyDE for retrieval
    """
    from llama_index.core.query_engine import TransformQueryEngine
    
    # Create the HyDE query transform
    hyde_transform = EnhancedHyDEQueryTransform(
        llm=llm,
        hyde_prompt=HYDE_PROMPT,
        include_original=include_original
    )
    
    # Create and return the transform query engine
    return TransformQueryEngine(base_query_engine, hyde_transform) 