"""
RAG Query Module for Advanced RAG

This module provides functionality to query the knowledge base, retrieve relevant
document chunks, and generate responses based on the user's query.
"""

import os
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from composio.tools.base.local import LocalAction


class AdvancedRagToolQueryRequest(BaseModel):
    """Request schema for querying the knowledge base"""

    query: str = Field(..., description="The query to search in the knowledge base")
    document_id: Optional[str] = Field(
        None,
        description="Optional document ID to restrict search to a specific document",
    )
    document_type: Optional[str] = Field(
        None,
        description="Optional document type to filter by ('pdf', 'docx', 'txt', etc.)",
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters to apply to the search"
    )
    include_metadata: bool = Field(
        False, description="Whether to include metadata in the response"
    )
    max_results: int = Field(5, description="Maximum number of results to return")


class RagResultItem(BaseModel):
    """Schema for an individual result item"""

    content: str = Field(..., description="The content that matched the query")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata about the content"
    )
    score: Optional[float] = Field(None, description="Relevance score for this result")


class AdvancedRagToolQueryResponse(BaseModel):
    """Response schema for knowledge base query"""

    results: List[RagResultItem] = Field(..., description="The search results")
    response: Optional[str] = Field(
        None, description="The combined response to the query from the knowledge base"
    )


class AdvancedRagToolQuery(
    LocalAction[AdvancedRagToolQueryRequest, AdvancedRagToolQueryResponse]
):
    """
    Tool for querying a knowledge base.
    """

    _tags = ["Knowledge Base", "rag"]

    def get_embedchain_config(self) -> Optional[Dict]:
        """
        Get configuration for embedchain based on available API keys

        Returns:
            Configuration dictionary or None
        """
        if "GEMINI_API_KEY" in os.environ:
            return {
                "llm": {
                    "provider": "google",
                    "config": {
                        "model": "gemini-2.0-flash",
                        "api_key": os.environ["GEMINI_API_KEY"],
                    },
                },
                "embedder": {
                    "provider": "google",
                    "config": {
                        "model": "models/embedding-001",
                    },
                },
                "vectordb": {
                    "provider": "chroma",
                    "config": {"collection_name": "composio_docs", "dir": "db"},
                },
            }
        return None

    def build_filter(self, request: AdvancedRagToolQueryRequest) -> Dict:
        """
        Build the filter dictionary for the query

        Args:
            request: The query request containing filter parameters

        Returns:
            Filter dictionary for embedchain query
        """
        where_filter = {}

        # Handle folder queries using parent_folder_id if specified as folder type
        if request.document_type == "folder" and request.document_id:
            where_filter["parent_folder_id"] = request.document_id
        else:
            # Apply standard document_id and document_type filters
            if request.document_id:
                where_filter["document_id"] = request.document_id
            if request.document_type:
                where_filter["document_type"] = request.document_type

        # Add any custom metadata filters
        if request.metadata_filters:
            where_filter.update(request.metadata_filters)

        return where_filter

    def convert_sources_to_results(
        self, sources: List[tuple], include_metadata: bool, max_results: int
    ) -> List[RagResultItem]:
        """
        Convert embedchain sources to RagResultItem objects

        Args:
            sources: List of (content, metadata) tuples from embedchain
            include_metadata: Whether to include metadata in results
            max_results: Maximum number of results to return

        Returns:
            List of RagResultItem objects
        """
        results = []

        if sources:
            # Limit the number of sources returned if needed
            limited_sources = sources[:max_results]

            for content, meta in limited_sources:
                results.append(
                    RagResultItem(
                        content=content,
                        metadata=meta if include_metadata else None,
                        score=meta.get(
                            "score", None
                        ),  # Attempt to get score if available
                    )
                )

        return results

    def execute(
        self, request: AdvancedRagToolQueryRequest, metadata: Dict
    ) -> AdvancedRagToolQueryResponse:
        """
        Query the knowledge base and return the response

        Args:
            request: The query request containing the query and filters
            metadata: Additional metadata

        Returns:
            AdvancedRagToolQueryResponse with results and generated response
        """
        try:
            from embedchain import App
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Failed to import App from embedchain: {e}"
            ) from e

        try:
            # Get configuration and initialize App
            config = self.get_embedchain_config()
            app = App.from_config(config=config) if config else App()

            # Build filters based on request
            where_filter = self.build_filter(request)

            # Execute query with filters
            # Embedchain's query returns (answer, sources)
            # sources is a list of tuples: (content, metadata)
            answer, sources = app.query(
                request.query,
                citations=True,  # Ensure sources are returned
                where=where_filter if where_filter else None,
            )

            # Convert sources to result items
            results = self.convert_sources_to_results(
                sources, request.include_metadata, request.max_results
            )

            return AdvancedRagToolQueryResponse(results=results, response=answer)

        except Exception as e:
            print(f"Error during query: {str(e)}")
            return AdvancedRagToolQueryResponse(
                results=[], response=f"Error querying the knowledge base: {str(e)}"
            )
