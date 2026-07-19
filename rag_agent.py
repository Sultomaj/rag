import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
# 1. Configuration & API Key

load_dotenv()

def build_vector_database(pdf_path, db_path="faiss_index"):
    print(f"[*] Loading document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("[*] Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # --- NEW SAFETY CHECK ---
    print(f"[*] Successfully extracted {len(docs)} text chunks.")
    if len(docs) == 0:
        raise ValueError("Error: No text was extracted. The PDF might be a scanned image or empty.")
    # ------------------------

    print("[*] Generating embeddings and building vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(db_path)
    print("[*] Vector database successfully built and saved!")
    return vector_store

def setup_rag_chain(vector_store):
    """Configures the LLM, the retrieval mechanism, and the grounding prompt.""" 
    # Initialize the LLM (using Llama-3.1 via Groq for extreme speed)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    # System Prompt designed for strict grounding (crucial for WHO Quality & Evaluation)
    system_prompt = (
        "You are an AI assistant designed to help health professionals safely extract "
        "information from official medical guidelines. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you do not know the answer, or if the answer is not contained in the context, "
        "say exactly 'I cannot answer this based on the provided document.' Do not hallucinate.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Build the chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant chunks
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    # Download a sample WHO PDF (e.g., a malaria or COVID-19 guideline) and place it in the folder
    sample_pdf = "who_guideline.pdf"
    
    # Only build the DB if it doesn't exist yet
    if not os.path.exists("faiss_index"):
        vector_db = build_vector_database(sample_pdf)
    else:
        print("[*] Loading existing vector database...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Setup the Agent
    agent = setup_rag_chain(vector_db)

    # Interactive Loop
    print("\n=== WHO Guideline RAG Agent Initialized ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_query = input("Ask a question about the document: ")
        if user_query.lower() == 'exit':
            break
            
        response = agent.invoke({"input": user_query})
        print(f"\nAgent: {response['answer']}\n")
        print("-" * 50)