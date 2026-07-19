# WHO Clinical Guideline RAG Agent

An AI-powered Retrieval-Augmented Generation (RAG) pipeline engineered to securely parse, embed, and query World Health Organization (WHO) clinical guidelines with strict hallucination guardrails. 

This project demonstrates a production-ready approach to AI data engineering and evaluation, ensuring that Large Language Models (LLMs) provide highly grounded, verifiable answers for healthcare professionals.

## 🏗 Architecture & Tech Stack

This pipeline is built for high speed, strict context adherence, and local embedding privacy:

*   **Framework:** LangChain (`langchain-classic`)
*   **Vector Database:** FAISS (Facebook AI Similarity Search) for ultra-fast local retrieval.
*   **Embeddings:** Hugging Face `all-MiniLM-L6-v2` (Sentence-Transformers) for localized, private vector generation.
*   **LLM Engine:** Meta Llama-3.1-8B via the Groq API for near-zero latency inference.
*   **Data Ingestion:** `PyPDFLoader` with `RecursiveCharacterTextSplitter`.

## 🛡️ Key Features & Evaluation Focus

1.  **Strict Grounding (Zero Hallucination Tolerance):** 
    The system prompt is heavily constrained and evaluated with `temperature=0`. If the answer is not explicitly found in the retrieved vector chunks, the agent is forced to reply: *"I cannot answer this based on the provided document."*
2.  **Context Preservation:** 
    PDFs are ingested and split using a 1000-character chunk size with a 200-character overlap to ensure medical context and multi-sentence guidelines are not orphaned during vectorization.
3.  **Source Citations:**
    The agent successfully synthesizes standing recommendations while maintaining the integrity of original WHO references and URLs.

## 🚀 Installation & Usage

### 1. Clone the repository
```bash

```

### 2. Install Dependecies
``` bash
pip install langchain langchain-classic langchain-community langchain-groq sentence-transformers faiss-cpu pypdf
```
### 3. Environment Variables
Obtain a free API key from Groq and set it in the script or export it to your environment:
``` bash
export GROQ_API_KEY="your_api_key_here"
```
### 4. Run the Agent
Place any text-based PDF (e.g., who_guideline.pdf) in the root directory and execute the pipeline:
```
python3 rag_agent.py
```

## 📊 Example Output
User Query: how many years since first SARS-Cov-2 infections were reported
Agent: According to the context, nearly five years have passed since the first SARS-CoV-2 infections were reported.
User Query: should mothers suspected or confirmed with COVID-19 continue breastfeeding?
Agent: According to the provided context, mothers with suspected or confirmed COVID-19 should be encouraged to initiate and continue breastfeeding. Based on the available evidence, mothers should be counselled that the benefits of breastfeeding substantially outweigh the potential risks of transmission.
