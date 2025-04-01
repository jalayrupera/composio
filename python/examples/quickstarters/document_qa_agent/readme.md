# Document QA Agent

This Document QA Agent processes documents (PDFs, DOCXs, or folders of documents) and allows you to ask questions about their content using advanced RAG capabilities.

## Features

- Process PDF documents with automatic text extraction
- Handle Word documents (DOCX format)
- Process entire folders of mixed document types
- Interactive question-answering about document content
- Metadata-rich responses that show source information
- Command-line interface for automation

## Setup

This agent requires components from the local `composio` development codebase. To set up the environment, run:

```bash
./setup.sh
```

This script will:
1. Create and activate a virtual environment
2. Install dependencies from `requirements.txt`
3. Install required local `composio` and `composio-crewai` packages
4. Set up the `.env` file template

You'll need to add your `GEMINI_API_KEY`, `COMPOSIO_API_KEY` to the created `.env` file before running the agent.

**Note:** If you modify the `composio` core or plugin code, re-run `./setup.sh` to update the installed packages.

## Usage

### Interactive Mode

To run the Document QA Agent in interactive mode, simply execute:

```bash
python main.py
```

The agent will prompt you to:
1. Enter a path to a document or folder to process
2. Ask questions about the document after processing is complete

## Supported Document Types

The agent currently supports processing the following:

- PDF files (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)
- Markdown files (`.md`)
- Folders containing any of the above supported file types.

## How It Works

The Document QA Agent uses a powerful RAG (Retrieval-Augmented Generation) system to:

1. Process documents by extracting their text content
2. Split content into manageable chunks
3. Store the chunks with relevant metadata (page numbers, source information, etc.)
4. Create embeddings for semantic search
5. Retrieve the most relevant information when questions are asked
6. Generate comprehensive, accurate answers using a large language model

## Advanced Configuration

The agent uses the following configurations:
- Vector database: ChromaDB (local storage)
- Embedding model: Google's Gemini Embedding model
- LLM: Gemini 2.0 Flash for fast, accurate responses
