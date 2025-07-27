import os
import io
import requests
import logging
import re
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Configuration & Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Objects ---
embedding_function = None
llm = None

def initialize_models():
    """Initializes the embedding model and LLM."""
    global embedding_function, llm
    if embedding_function is None:
        logging.info("Initializing embedding model...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_function = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    if llm is None:
        logging.info("Initializing LLM...")
        llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    logging.info("Models are ready.")

def preprocess_text_for_waiting_periods(text: str) -> str:
    """
    Injects waiting period context into list items for better retrieval.
    """
    logging.info("Starting context-aware pre-processing...")
    lines = text.split('\n')
    processed_lines = []
    current_context = ""

    patterns = {
        "90 Days Waiting Period": re.compile(r"i\.\s+90\s+Days\s+Waiting\s+Period", re.IGNORECASE),
        "One year waiting period": re.compile(r"ii\.\s+One\s+year\s+waiting\s+period", re.IGNORECASE),
        "Two years waiting period": re.compile(r"iii\.\s+Two\s+years\s+waiting\s+period", re.IGNORECASE),
        "Three years waiting period": re.compile(r"iv\.\s+Three\s+years\s+waiting\s+period", re.IGNORECASE),
    }
    
    active_context = None
    for line in lines:
        # Check if the line is a new heading
        is_heading = False
        for context, pattern in patterns.items():
            if pattern.search(line):
                active_context = context
                is_heading = True
                break
        
        # If it's a list item under an active context, inject it
        if active_context and re.match(r'^\s*[a-z]\.\s+', line.strip()):
            clean_line = line.strip().split('.', 1)[-1].strip()
            processed_lines.append(f"Regarding the {active_context}: the procedure '{clean_line}' is included.")
        else:
            processed_lines.append(line)
            # Reset context if the line was a heading, so we don't apply it to subsequent non-list items
            if is_heading:
                active_context = None


    logging.info("Pre-processing complete.")
    return "\n".join(processed_lines)

def get_retriever_from_url(document_url: str):
    """Downloads, processes, and creates a retriever for a given URL."""
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        
        temp_pdf_path = "temp_document_for_qt_test.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        os.remove(temp_pdf_path)

        # --- FIX: Apply context-aware pre-processing ---
        full_text = "\n".join([doc.page_content for doc in docs])
        processed_text = preprocess_text_for_waiting_periods(full_text)
        processed_docs = [Document(page_content=processed_text)]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(processed_docs)

        vectorstore = Chroma.from_documents(chunks, embedding_function)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logging.error(f"Failed to process document: {e}")
        return None

def main():
    """
    This script demonstrates and tests the Query Transformation strategy.
    """
    initialize_models()

    # --- 1. The Query Transformation Chain (IMPROVED PROMPT) ---
    rewrite_template = """
You are an expert at rewriting user questions into more effective search queries.
Based on the user's question, generate 3 additional, different, and simple questions that are likely to find relevant information in a long insurance policy document.
Each rewritten query should be a plain, natural language question on a new line. Do not use boolean operators or complex syntax.

Original Question:
{question}

Rewritten Queries:
"""
    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    
    query_rewriter_chain = (
        rewrite_prompt 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.strip().split("\n"))
    )

    # --- 2. Setup the Document and Retriever ---
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    retriever = get_retriever_from_url(document_url)
    if not retriever:
        return

    # --- 3. The Main RAG Chain ---
    rag_template = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- 4. The Full Chain with Query Transformation ---
    # This chain will take the original question, generate new questions,
    # retrieve documents for all of them, and then answer the original question.
    def retrieve_for_multiple_queries(original_question):
        rewritten_queries = query_rewriter_chain.invoke({"question": original_question})
        all_queries = [original_question] + [q for q in rewritten_queries if q.strip()]
        
        print("\n--- Generated Search Queries ---")
        for q in all_queries:
            print(q)

        retrieved_docs = retriever.batch(all_queries)
        
        unique_docs = {doc.page_content: doc for sublist in retrieved_docs for doc in sublist}
        print(f"\n--- Retrieved {len(unique_docs)} unique documents for context ---")
        return list(unique_docs.values())

    final_rag_chain = (
        {"context": (lambda x: format_docs(retrieve_for_multiple_queries(x["question"]))), "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # --- 5. Run the Experiment ---
    original_question = "What is the waiting period for cataract surgery?"
    logging.info(f"Original question: '{original_question}'")

    final_answer = final_rag_chain.invoke({"question": original_question})

    print("\n--- Final Answer ---")
    print(final_answer)

if __name__ == "__main__":
    main()
