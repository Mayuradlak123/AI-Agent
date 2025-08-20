# 🧠 RAG-based Chatbot with MongoDB + FAISS

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that answers user questions using data stored in MongoDB and embeddings stored in FAISS (or Pinecone).

---

## 🚀 How It Works

1. **User Query**
   - The user asks a question in natural language.

2. **Embedding Generation**
   - The query is converted into an embedding vector using a pre-trained model (e.g., OpenAI, Gemini).

3. **Vector Search**
   - The embedding is compared against document embeddings stored in FAISS.
   - The top-k most relevant documents are retrieved.

4. **Intent Recognition**
   - The query is dynamically analyzed to determine the user’s intent.
   - Example:  
     - `"Who founded the company?"` → intent = `founded_by`  
     - `"What is the company size?"` → intent = `company_size`

5. **Answer Construction**
   - The retrieved documents are passed to the LLM.
   - The LLM generates a natural answer based on both **intent** and **context**.

6. **Final Response**
   - The chatbot responds with context-aware, non-hardcoded answers.

---

## 📂 Project Structure

