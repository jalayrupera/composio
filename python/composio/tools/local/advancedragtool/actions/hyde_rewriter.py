"""
HyDE Query Rewriter Module for Advanced RAG

This module implements the Hypothetical Document Embeddings (HyDE) approach
for query rewriting. It generates a hypothetical document snippet that directly
answers the user's query, which can then be used for embedding-based retrieval.
"""

import os
import typing as t
import google.genai as genai
from pydantic import BaseModel, Field
from composio.tools.base.local import LocalAction


class HydeQueryRewriterInput(BaseModel):
    """Input schema for HyDE query rewriting"""

    original_query: str = Field(..., description="The original user query to rewrite.")


class HydeQueryRewriterOutput(BaseModel):
    """Output schema for HyDE query rewriting"""

    hypothetical_document: str = Field(
        ..., description="The generated hypothetical document snippet."
    )


class HydeQueryRewriter(LocalAction[HydeQueryRewriterInput, HydeQueryRewriterOutput]):
    """
    Rewrites a user query into a hypothetical document snippet using HyDE
    (Hypothetical Document Embeddings) approach. This snippet can then be
    used for embedding-based retrieval.
    """

    _tags: t.List[str] = ["Knowledge Base", "RAG", "Query Processing"]

    def get_api_key(self) -> t.Optional[str]:
        """
        Retrieve the API key from environment variables

        Returns:
            API key string or None if not found
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY not found in environment")
        return api_key

    def create_client(self, api_key: str) -> t.Optional[genai.Client]:
        """
        Create a Google Generative AI client with the provided API key

        Args:
            api_key: The API key to use for authentication

        Returns:
            Google Generative AI client or None if creation fails
        """
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            print(f"Error creating Gemini client: {e}")
            return None

    def generate_hypothetical_document(
        self, client: genai.Client, query: str
    ) -> t.Optional[str]:
        """
        Generate a hypothetical document snippet that answers the query

        Args:
            client: Google Generative AI client
            query: The original user query

        Returns:
            Generated hypothetical document or None if generation fails
        """
        try:
            # Construct the prompt for the LLM
            prompt = f"""
            Given the user query: "{query}"
            Generate a concise, plausible document snippet that directly answers this query. 
            This snippet will be used to improve search results. Output *only* the generated 
            snippet itself, without any preamble or explanation.
            """

            # Call the model
            response = client.models.generate_content(
                model="gemini-1.5-flash-latest", contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            print(f"Error generating hypothetical document: {e}")
            return None

    def execute(
        self, request: HydeQueryRewriterInput, **kwargs
    ) -> HydeQueryRewriterOutput:
        """
        Execute the HyDE query rewriting action

        Args:
            request: The query rewriting request containing the original query

        Returns:
            HydeQueryRewriterOutput containing the generated hypothetical document
        """
        original_query = request.original_query

        try:
            # Get API key and create client
            api_key = self.get_api_key()
            if not api_key:
                return HydeQueryRewriterOutput(hypothetical_document=original_query)

            client = self.create_client(api_key)
            if not client:
                return HydeQueryRewriterOutput(hypothetical_document=original_query)

            # Generate the hypothetical document
            hypothetical_doc = self.generate_hypothetical_document(
                client, original_query
            )

            # If generation failed, fall back to the original query
            if not hypothetical_doc:
                print(f"Warning: HyDE generation failed for query: {original_query}")
                return HydeQueryRewriterOutput(hypothetical_document=original_query)

            return HydeQueryRewriterOutput(hypothetical_document=hypothetical_doc)

        except Exception as e:
            print(f"Error in HydeQueryRewriter execute: {e}")
            # Fallback to original query on any error
            return HydeQueryRewriterOutput(hypothetical_document=original_query)
