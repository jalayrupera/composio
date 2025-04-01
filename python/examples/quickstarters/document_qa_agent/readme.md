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

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your environment variables in a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
# COMPOSIO_API_KEY=your_composio_key_here (if needed by tools)
```

## Local Development Setup (Using Local Composio Code)

If you are developing the main `composio` package or the `composio-crewai` plugin and want this example to use your *latest local changes* instead of the versions specified in `requirements.txt` or installed from PyPI, follow these steps **after** activating the virtual environment (`source venv/bin/activate`):

1.  **Navigate to the Project Root:**
    Ensure you are in the main project directory (`/Users/username/composio`).

2.  **Install Local Packages:**
    Run the following commands to build and install the `composio` core package and the `composio-crewai` plugin from your local source code into the example's active virtual environment:
    ```bash
    pip install ./python/
    pip install ./python/plugins/crew_ai
    ```

    **Important:** This installation method copies the code at the time of installation. If you make further changes to the code in `./python/` or `./python/plugins/crew_ai/`, you **must re-run** these `pip install` commands to update the versions used by this example.

Now, when you run `python main.py` from the `python/examples/quickstarters/document_qa_agent/` directory (with the venv active), it will use the locally built versions of `composio` and `composio-crewai`.

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
