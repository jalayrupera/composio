"""
Advanced RAG (Retrieval-Augmented Generation) Document Question-Answering System

This module implements an interactive QA system that allows users to:
1. Process documents into a knowledge base
2. Query the knowledge base using natural language questions
3. Get answers with supporting evidence from the processed documents

The system uses a two-agent architecture:
- Document Ingestion Agent: Processes documents and adds them to the knowledge base
- Query Agent: Uses advanced RAG techniques to answer questions based on document content
"""

import os
import json
import ast
import re
from pathlib import Path
import textwrap
from typing import Optional, List, Dict

import dotenv
from textwrap import dedent
from composio_crewai import App, ComposioToolSet
from crewai import Agent, Crew, Process, Task, LLM

# Load environment variables from .env file
dotenv.load_dotenv()

# Constants for configuration
GEMINI_MODEL = "gemini/gemini-2.0-flash"
TEMPERATURE = 0.3
INITIAL_RETRIEVAL_COUNT = 20
TOP_N_RERANKED = 5
RERANK_SCORE_THRESHOLD = 1.0
ORIGINAL_SCORE_THRESHOLD = 0.3


def get_document_type(path: str) -> str:
    """
    Determine the document type based on file extension or if it's a folder

    Args:
        path: Path to document or folder

    Returns:
        String indicating document type: "folder", "pdf", "docx", or "txt"
    """
    if os.path.isdir(path):
        return "folder"

    file_extension = Path(path).suffix.lower()
    if file_extension == ".pdf":
        return "pdf"
    elif file_extension == ".docx":
        return "docx"
    elif file_extension == ".md":
        return "md"
    else:
        return "txt"


def extract_json_from_response(result_str: str) -> Optional[Dict]:
    """
    Extract and parse JSON data from a response string, handling cases where
    it might be wrapped in code blocks or have other formatting issues.

    Args:
        result_str: String containing JSON data, possibly with formatting

    Returns:
        Parsed JSON as dictionary, or None if parsing fails
    """
    # Check if the result is wrapped in markdown code blocks and extract content
    if "```" in result_str:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result_str)
        if match:
            result_str = match.group(1).strip()

    # Try multiple parsing approaches
    parsing_methods = [
        # Standard JSON parsing
        lambda s: json.loads(s),
        # Try with cleaned string
        lambda s: json.loads(s.strip()),
        # Last resort: Python literal evaluation
        lambda s: ast.literal_eval(s),
    ]

    for parse_method in parsing_methods:
        try:
            return parse_method(result_str)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue

    # If all parsing attempts fail, return None
    return None


def display_formatted_answer(answer: str, sources: Optional[List[Dict]] = None) -> None:
    """
    Displays the answer and supporting sources in a readable format

    Args:
        answer: The text answer to display
        sources: List of source documents that support the answer
    """
    # Determine display width based on terminal size
    terminal_width = os.get_terminal_size().columns
    width = min(terminal_width, 100)

    # Display answer section
    print("\n" + "=" * width)
    print("ANSWER")
    print("-" * width)

    # Format and print the answer with proper text wrapping
    wrapper = textwrap.TextWrapper(width=width, initial_indent="", subsequent_indent="")
    for line in wrapper.wrap(answer):
        print(line)

    # Display sources section if available sources exist
    if not sources:
        print("\n" + "=" * width)
        return

    # Filter sources to only show those with high relevance scores
    filtered_sources = []
    for item in sources:
        rerank_score = item.get("rerank_score")
        orig_score = item.get("score")

        if (
            rerank_score is not None
            and rerank_score > RERANK_SCORE_THRESHOLD
            or orig_score is not None
            and orig_score > ORIGINAL_SCORE_THRESHOLD
        ):
            filtered_sources.append(item)

    # Skip sources section if no highly relevant sources
    if not filtered_sources:
        print("\n" + "=" * width)
        return

    # Display sources header
    print("\n" + "-" * width)
    print("SOURCES")
    print("-" * width)

    # Format and display each source
    for i, item in enumerate(filtered_sources):
        content = item.get("content", "N/A")
        metadata = item.get("metadata", {})

        # Display source number and content
        print(f"\n[{i + 1}]")

        source_wrapper = textwrap.TextWrapper(
            width=width, initial_indent="    ", subsequent_indent="    "
        )

        for line in source_wrapper.wrap(content):
            print(line)

        # Display source file
        if metadata.get("source_path"):
            file_path = os.path.basename(metadata["source_path"])
            print(f"    Source: {file_path}")

        print("    " + "-" * (width - 4))

    print("\n" + "=" * width)


def create_agents(llm: LLM, tools: List) -> tuple:
    """
    Create and configure the agents used in the RAG system

    Args:
        llm: The language model to use
        tools: List of tools available to the agents

    Returns:
        Tuple of (ingestion_agent, query_agent)
    """
    # Document ingestion agent handles adding documents to the knowledge base
    ingestion_agent = Agent(
        role="Document Ingestion Specialist",
        goal="Efficiently process documents and add their content to the knowledge base",
        verbose=True,
        backstory="Expert in document ingestion and processing for knowledge retrieval",
        llm=llm,
        tools=tools,
        allow_delegation=False,
    )

    # Query agent handles retrieving information and answering questions
    query_agent = Agent(
        role="Advanced RAG Query Agent",
        goal="Answer questions accurately using advanced RAG techniques",
        verbose=True,
        backstory="Expert in information retrieval and question answering",
        llm=llm,
        tools=tools,
        allow_delegation=False,
    )

    return (ingestion_agent, query_agent)


def process_document(document_path: str, ingestion_agent: Agent) -> str:
    """
    Process a document or folder and add it to the knowledge base

    Args:
        document_path: Path to the document or folder
        ingestion_agent: Agent to use for document processing

    Returns:
        Result message from processing
    """
    doc_type = get_document_type(document_path)

    # Create the document processing task
    process_doc_task = Task(
        description=f"Process the {doc_type} at: {document_path}",
        expected_output="Processing complete message",
        agent=ingestion_agent,
    )

    # Execute the task with a sequential crew
    process_crew = Crew(
        agents=[ingestion_agent],
        tasks=[process_doc_task],
        process=Process.sequential,
        verbose=True,
    )

    return str(process_crew.kickoff())


def create_query_tasks(query: str, query_agent: Agent) -> List[Task]:
    """
    Create the sequence of tasks needed to answer a user query

    Args:
        query: The user's natural language query
        query_agent: Agent to use for querying the knowledge base

    Returns:
        List of tasks to execute sequentially
    """
    # Task 1: Rewrite the query using HyDE approach
    rewrite_query_task = Task(
        description=dedent(
            f"""\
            Rewrite the user's query using the HyDE approach to generate a hypothetical document snippet.
            User Query: "{query}"
            Use the ADVANCED_RAG_TOOL_HYDE_QUERY_REWRITER tool.
            """
        ),
        expected_output="The generated hypothetical document snippet string.",
        agent=query_agent,
    )

    # Task 2: Retrieve relevant document chunks
    retrieve_task = Task(
        description=dedent(
            f"""\
            Retrieve relevant document chunks using the hypothetical document snippet generated in the previous step (available in context).
            **CRITICAL**: Use the exact hypothetical document string from the previous task's output as the 'query' parameter for the ADVANCED_RAG_TOOL_ADVANCED_RAG_TOOL_QUERY.
            Query the entire knowledge base.
            Set include_metadata to True.
            Request {INITIAL_RETRIEVAL_COUNT} results (max_results={INITIAL_RETRIEVAL_COUNT}) to provide candidates for re-ranking.

            IMPORTANT TOOL USAGE NOTE: When constructing the JSON input for ADVANCED_RAG_TOOL_ADVANCED_RAG_TOOL_QUERY:
            - Include 'query' (the hypothetical snippet) and 'max_results'.
            - Include 'include_metadata' set to true.
            - **Omit** the optional 'document_id', 'document_type', and 'metadata_filters' parameters entirely from the JSON object.

            **CRITICAL OUTPUT NOTE**: After receiving the output from the tool (which looks like `{{'data': {{'response': '...', 'results': [...]}}, ...}}`), extract the inner dictionary located at the 'data' key. Your final output MUST be a **valid JSON string** representation of this inner dictionary, using double quotes for all keys and string values. Example: '{{"response": "...", "results": [{{"content": "...", "metadata": {{...}}}}]}}'. Do NOT output a Python string representation using single quotes.
            """
        ),
        expected_output=f"A valid JSON string representing the dictionary containing 'response' and potentially up to {INITIAL_RETRIEVAL_COUNT} 'results', extracted from the tool's output.",
        agent=query_agent,
        context=[rewrite_query_task],
    )

    # Task 3: Re-rank and generate the final answer
    rerank_and_answer_task = Task(
        description=dedent(
            f"""\
            Re-rank the initially retrieved document chunks and generate the final answer for the original user query: "{query}".

            1. Take the JSON string received from the previous 'retrieve_task' (available in context).
            2. Parse this JSON string using `json.loads()` to get a Python dictionary. Extract the 'results' list from this dictionary.
            3. If the 'results' list is not empty:
               a. Prepare the input for the ADVANCED_RAG_TOOL_RERANK_RESULTS tool.
               b. The 'original_query' parameter should be the string: "{query}".
               c. The 'results' parameter should be a JSON string representation of the list of results with double quotes.
                  Use `json.dumps(results)` to properly convert the Python list to a JSON string.
               d. The 'top_n' parameter should be the integer {TOP_N_RERANKED}.
               e. Call the ADVANCED_RAG_TOOL_RERANK_RESULTS tool with the properly formatted JSON input.
               f. Extract the 'reranked_results' list from the tool's output.
            4. If the initial 'results' list was empty or the re-ranking tool failed/returned empty, use an empty list for the next step.
            5. Synthesize a final, comprehensive answer to the original user query: "{query}", based *only* on the content of the 'reranked_results'.
            6. Prepare the final output dictionary containing the synthesized 'response' and the 'reranked_results' list received from the reranker tool.

            **CRITICAL NOTE**: DO NOT wrap your output in triple backticks or markdown code blocks. The output should be a plain JSON string only.
            Your final output MUST be a **valid JSON string** representation of the Python dictionary containing the final synthesized 'response' and the 'reranked_results' list, using double quotes for all keys and string values. Example: '{{"response": "The final answer is...", "results": [reranked_chunk_1, ...]}}'.
            """
        ),
        expected_output=f"A valid JSON string representing the dictionary containing the final synthesized 'response' and the top {TOP_N_RERANKED} re-ranked 'results'.",
        agent=query_agent,
        context=[retrieve_task],
    )

    return [rewrite_query_task, retrieve_task, rerank_and_answer_task]


def process_query(query: str, query_agent: Agent) -> Dict:
    """
    Process a user query through the RAG pipeline

    Args:
        query: The user's natural language query
        query_agent: Agent to use for querying the knowledge base

    Returns:
        Dictionary containing the answer and supporting sources
    """
    # Create the tasks for the query processing pipeline
    tasks = create_query_tasks(query, query_agent)

    # Execute the tasks with a sequential crew
    query_crew = Crew(
        agents=[query_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    # Get and parse the results
    result_str = str(query_crew.kickoff())
    parsed_result = extract_json_from_response(result_str)

    if parsed_result and isinstance(parsed_result, dict):
        return parsed_result
    else:
        # If parsing fails, return a dict with just the raw response
        return {"response": result_str, "results": []}


def interactive_mode():
    """
    Run the document QA agent in interactive mode, allowing users to
    process documents and ask questions about the knowledge base
    """
    print("===== Starting Document QA Agent =====")
    print("Initializing Agents and Tools...")

    try:
        # Initialize tools and language model
        toolset = ComposioToolSet()
        tools = toolset.get_tools(apps=[App.ADVANCED_RAG_TOOL])

        llm = LLM(
            model=GEMINI_MODEL,
            api_key=os.environ["GEMINI_API_KEY"],
            temperature=TEMPERATURE,
        )

        # Create the agents
        ingestion_agent, query_agent = create_agents(llm, tools)
        print("Initialization complete.")

    except Exception as e:
        print(f"FATAL: Failed to initialize components: {e}")
        return

    # Display welcome message
    print("\nProcess documents/folders and ask questions about the knowledge base.")
    print(
        "Type 'process' to add a document/folder, 'query' to ask a question, 'exit' to quit."
    )

    # Start in process mode by default
    mode = "process"

    # Main interaction loop
    while True:
        if mode == "process":
            # Document processing mode
            document_path = input(
                "\nEnter document or folder path to process (or type 'query', 'exit'): "
            ).strip()

            # Handle mode switching or exit
            if document_path.lower() == "exit":
                break
            elif document_path.lower() == "query":
                mode = "query"
                continue
            elif not document_path:
                continue

            # Validate the path exists
            if not os.path.exists(document_path):
                print(f"Error: Path '{document_path}' does not exist.")
                continue

            # Process the document
            print(f"\nProcessing '{document_path}'...")
            try:
                result = process_document(document_path, ingestion_agent)
                print("\n===== Document Processing Result =====")
                print(result)
            except Exception as e:
                print(f"\nError processing document: {e}")
                continue

        elif mode == "query":
            # Query mode
            query = input("\nEnter your query (or type 'process', 'exit'): ").strip()

            # Handle mode switching or exit
            if query.lower() == "exit":
                break
            elif query.lower() == "process":
                mode = "process"
                continue
            elif not query:
                continue

            # Process the query
            print(f"\nAnswering: {query}\n")
            print("Processing query... This may take a moment.")

            try:
                # Run the query through the RAG pipeline
                result = process_query(query, query_agent)

                # Display the formatted answer and sources
                answer = result.get("response", "No answer provided")
                sources = result.get("results", [])
                display_formatted_answer(answer, sources)

            except Exception as e:
                print(f"\nError during query: {e}")


def main():
    """Main entry point for the RAG document QA system"""
    interactive_mode()


if __name__ == "__main__":
    main()
