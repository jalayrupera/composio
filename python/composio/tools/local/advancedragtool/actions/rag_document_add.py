"""
Document Addition Module for Advanced RAG

This module provides functionality to add documents to the knowledge base.
It supports various document types including PDF, DOCX, Markdown, and folders
containing multiple documents. Documents are processed, split into chunks,
and added to the knowledge base with appropriate metadata.
"""

import os
import uuid
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedchain.models.data_type import DataType

from composio.tools.base.local import LocalAction


class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    source: str = Field(..., description="Source of the document")
    type: str = Field(
        ..., description="Type of the document (e.g., 'pdf', 'docx', 'folder')"
    )
    page_numbers: Optional[List[int]] = Field(
        None, description="Page numbers for the document (if applicable)"
    )
    additional_info: Optional[Dict] = Field(
        None, description="Additional information about the document"
    )


class AdvancedRagToolDocumentAddRequest(BaseModel):
    """Request schema for adding documents to the knowledge base."""

    document_content: str = Field(
        ...,
        description="Path to the document or folder to add to the knowledge base.",
        json_schema_extra={"file_readable": False},
    )
    document_type: str = Field(
        ..., description="Type of document ('pdf', 'docx', 'md', 'txt', 'folder')"
    )
    document_metadata: Optional[Dict] = Field(
        None, description="Additional metadata about the document"
    )


class AdvancedRagToolDocumentAddResponse(BaseModel):
    """Response schema for document addition operation."""

    status: str = Field(..., description="Status of the addition to the knowledge base")
    document_id: Optional[str] = Field(
        None, description="ID of the document or folder added to the knowledge base"
    )
    chunks_count: Optional[int] = Field(
        None, description="Number of chunks the document was split into"
    )
    file_ids: Optional[List[str]] = Field(
        None, description="List of file document IDs when adding a folder"
    )


class AdvancedAddDocumentToRagTool(
    LocalAction[AdvancedRagToolDocumentAddRequest, AdvancedRagToolDocumentAddResponse]
):
    """Tool for adding documents to the knowledge base"""

    _tags = ["Knowledge Base", "Document Processing"]

    # Text splitter configuration
    CHUNK_SIZE = 10000
    CHUNK_OVERLAP = 100

    def __init__(self):
        """Initialize the text splitter used across document processing."""
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def _validate_file_path(self, file_path: str) -> None:
        """
        Validate that a file path exists

        Args:
            file_path: Path to validate

        Raises:
            ValueError: If the path is invalid or doesn't exist
        """
        if not os.path.isfile(file_path):
            raise ValueError(
                f"The file path {file_path} is not valid or does not exist"
            )

    def _process_pdf(self, file_path: str, metadata: Dict) -> List[Tuple[str, Dict]]:
        """
        Process PDF content and return chunks with metadata

        Args:
            file_path: Path to the PDF file
            metadata: Document metadata

        Returns:
            List of (content, metadata) tuples
        """
        try:
            self._validate_file_path(file_path)

            # Process the PDF using LangChain's PyPDFLoader
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            result_chunks = []

            # Process each page and split into smaller chunks if needed
            for i, page in enumerate(pages):
                if not page.page_content.strip():  # Skip empty pages
                    continue

                # Create base metadata for this page
                page_metadata = metadata.copy() if metadata else {}
                page_metadata.update(
                    {
                        "page": i + 1,
                        "total_pages": len(pages),
                        "document_type": "pdf",
                        "source": page.metadata.get("source", file_path),
                    }
                )

                # Split the page content into smaller chunks
                page_chunks = self.text_splitter.split_text(page.page_content)

                # Add each chunk with appropriate metadata
                for j, chunk in enumerate(page_chunks):
                    chunk_metadata = page_metadata.copy()
                    chunk_metadata.update(
                        {"chunk": j + 1, "total_chunks_in_page": len(page_chunks)}
                    )
                    result_chunks.append((chunk, chunk_metadata))

            return result_chunks
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def _process_docx(self, file_path: str, metadata: Dict) -> List[Tuple[str, Dict]]:
        """
        Process DOCX content and return chunks with metadata

        Args:
            file_path: Path to the DOCX file
            metadata: Document metadata

        Returns:
            List of (content, metadata) tuples
        """
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx is not installed. Please install it.")

        try:
            self._validate_file_path(file_path)

            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            # Join all paragraphs and then split into optimal chunks
            full_text = "\n".join(paragraphs)
            text_chunks = self.text_splitter.split_text(full_text)

            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update(
                        {
                            "chunk": i + 1,
                            "total_chunks": len(text_chunks),
                            "document_type": "docx",
                            "source": file_path,
                        }
                    )
                    chunks.append((chunk_text, chunk_metadata))

            return chunks
        except Exception as e:
            raise Exception(f"Error processing DOCX: {str(e)}")

    def _process_markdown(
        self, file_path: str, metadata: Dict
    ) -> List[Tuple[str, Dict]]:
        """
        Process markdown content

        Args:
            file_path: Path to the markdown file
            metadata: Document metadata

        Returns:
            List of (content, metadata) tuples
        """
        try:
            self._validate_file_path(file_path)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                return []

            # Split the content into chunks
            chunks = self.text_splitter.split_text(content)

            result_chunks = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update(
                        {
                            "document_type": "markdown",
                            "source": file_path,
                            "chunk": i + 1,
                            "total_chunks": len(chunks),
                        }
                    )
                    result_chunks.append((chunk, chunk_metadata))

            return result_chunks
        except Exception as e:
            raise Exception(f"Error processing markdown file: {str(e)}")

    def _process_txt(self, file_path: str, metadata: Dict) -> List[Tuple[str, Dict]]:
        """
        Process plain text content

        Args:
            file_path: Path to the text file
            metadata: Document metadata

        Returns:
            List of (content, metadata) tuples
        """
        try:
            self._validate_file_path(file_path)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                return []

            # Split the content into chunks
            chunks = self.text_splitter.split_text(content)

            result_chunks = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update(
                        {
                            "document_type": "txt",
                            "source": file_path,
                            "chunk": i + 1,
                            "total_chunks": len(chunks),
                        }
                    )
                    result_chunks.append((chunk, chunk_metadata))

            return result_chunks
        except Exception as e:
            raise Exception(f"Error processing text file: {str(e)}")

    def _process_folder(
        self, folder_path: str, metadata: Dict
    ) -> List[Tuple[str, Dict]]:
        """
        Process all documents in a folder

        Args:
            folder_path: Path to the folder
            metadata: Document metadata

        Returns:
            List of (content, metadata) tuples
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"The path {folder_path} is not a valid directory")

        results = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = Path(file).suffix.lower()

                # Generate unique ID per file with folder context
                file_metadata = metadata.copy() if metadata else {}
                file_id = str(uuid.uuid4())[:8]
                file_metadata.update(
                    {
                        "document_id": file_id,
                        "parent_folder_id": metadata.get("folder_id"),
                        "source": file_path,
                        "folder": folder_path,
                        "filename": file,
                    }
                )

                # Process based on file type
                if file_extension == ".pdf":
                    results.extend(self._process_pdf(file_path, file_metadata))
                elif file_extension == ".docx":
                    results.extend(self._process_docx(file_path, file_metadata))
                elif file_extension == ".md":
                    results.extend(self._process_markdown(file_path, file_metadata))
                elif file_extension == ".txt":
                    results.extend(self._process_txt(file_path, file_metadata))

        return results

    def _get_config(self) -> Optional[Dict]:
        """
        Get configuration based on available API keys

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
        return None  # Will use embedchain defaults

    def _process_document_by_type(
        self, document_path: str, document_type: str, document_metadata: Dict
    ) -> Tuple[List[Tuple[str, Dict]], str]:
        """
        Process document based on its type

        Args:
            document_path: Path to the document
            document_type: Type of document ('pdf', 'docx', 'md', 'txt', 'folder')
            document_metadata: Document metadata

        Returns:
            Tuple of (list of content chunks, document ID)
        """
        # Generate appropriate IDs based on document type
        if document_type == "folder":
            folder_id = str(uuid.uuid4())[:8]
            document_metadata["folder_id"] = folder_id
            # Use folder_id as the main document_id for the operation
            document_id = folder_id
        else:
            document_id = str(uuid.uuid4())[:8]
            document_metadata["document_id"] = document_id

        # Process document based on type
        if document_type == "pdf":
            chunks = self._process_pdf(document_path, document_metadata)
        elif document_type == "docx":
            chunks = self._process_docx(document_path, document_metadata)
        elif document_type == "md":
            chunks = self._process_markdown(document_path, document_metadata)
        elif document_type == "txt":
            chunks = self._process_txt(document_path, document_metadata)
        elif document_type == "folder":
            chunks = self._process_folder(document_path, document_metadata)
        else:
            raise ValueError(
                f"Unsupported document type: {document_type}. Supported types are: 'pdf', 'docx', 'md', 'txt', 'folder'"
            )

        return chunks, document_id

    def _add_chunks_to_knowledge_base(
        self, chunks: List[Tuple[str, Dict]], document_id: str, document_type: str
    ) -> Tuple[int, Optional[List[str]]]:
        """
        Add content chunks to the knowledge base

        Args:
            chunks: List of (content, metadata) tuples
            document_id: Document ID
            document_type: Type of document

        Returns:
            Tuple of (number of chunks added, list of file IDs for folder)
        """
        try:
            from embedchain import App
        except ImportError as e:
            raise ImportError(f"Failed to import App from embedchain: {e}") from e

        if not chunks:
            return 0, None

        # Dynamic configuration based on available API keys
        config = self._get_config()
        app = App.from_config(config=config) if config else App()

        # Clear existing chunks for this document
        # try:
        #     # Revert back to 'where' as it's standard for ChromaDB - Still causing issues, commenting out for now.
        #     app.db.delete(where={"document_id": document_id})
        # except Exception as e:
        #     print(f"Warning: Could not clear existing document chunks: {str(e)}")

        # Add each chunk with its metadata
        chunk_count = 0
        file_ids = set()

        for content, chunk_metadata in chunks:
            try:
                if content.strip():
                    app.add(
                        content,
                        data_type=DataType.TEXT,
                        metadata={
                            "document_id": chunk_metadata.get(
                                "document_id", document_id
                            ),
                            "source_path": chunk_metadata.get("source", ""),
                            "document_type": chunk_metadata.get("document_type", ""),
                            **chunk_metadata,
                        },
                    )
                    chunk_count += 1

                    # Collect file_id if this is part of a folder
                    if document_type == "folder" and "document_id" in chunk_metadata:
                        file_ids.add(chunk_metadata["document_id"])

            except Exception as chunk_error:
                print(f"Error adding chunk: {str(chunk_error)}")

        return chunk_count, list(file_ids) if file_ids else None

    def execute(
        self, request: AdvancedRagToolDocumentAddRequest, metadata: Dict
    ) -> AdvancedRagToolDocumentAddResponse:
        """
        Add document content to the knowledge base

        Args:
            request: Document addition request
            metadata: Additional metadata

        Returns:
            AdvancedRagToolDocumentAddResponse with status and information
        """
        document_type = request.document_type.lower()
        document_metadata = request.document_metadata or {}

        try:
            # Process the document
            chunks, document_id = self._process_document_by_type(
                request.document_content, document_type, document_metadata
            )

            # Add chunks to knowledge base
            chunks_count, file_ids = self._add_chunks_to_knowledge_base(
                chunks, document_id, document_type
            )

            return AdvancedRagToolDocumentAddResponse(
                status="Document processed and added successfully",
                chunks_count=chunks_count,
                document_id=document_id,
                file_ids=file_ids,
            )

        except Exception as e:
            print(f"Error during document processing: {str(e)}")
            return AdvancedRagToolDocumentAddResponse(
                status=f"Error during document processing: {str(e)}",
                document_id=document_metadata.get("document_id"),
                chunks_count=0,
                file_ids=None,  # Explicitly set file_ids to None in error case
            )
