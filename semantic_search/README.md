
# Semantic Search Module

This module is part of the `rag-pipeline` project and is responsible for handling semantic search over PDF documents. It includes the following components:

## 📄 PDF Ingestion & Paragraph Extraction
- Uses `PyMuPDF` to extract text from PDF files.
- Splits the text into paragraphs and truncates long ones to maintain consistency.

## 🔍 Semantic Embedding
- Utilizes `sentence-transformers` (specifically `all-MiniLM-L6-v2`) to generate semantic embeddings for each paragraph.
- Embeddings capture contextual meaning, enabling semantic search beyond keyword matching.

## 🧠 Vector Storage in Milvus
- Stores paragraph embeddings and their corresponding text in a Milvus vector database.
- Creates an IVF_FLAT index for efficient similarity search.

## 🔎 Semantic Search
- Accepts a natural language query.
- Embeds the query and retrieves the top matching paragraphs from Milvus based on semantic similarity.

## 📘 Sample PDF
- The included PDF (`ST_SGB_2019_7-EN.pdf`) is a publicly available UN environmental policy document.
- It serves as a sample for demonstrating semantic search capabilities.

## 🚀 Future Plans
- Integration with LLMs (e.g., Qwen, DeepSeek) to generate answers based on retrieved content.
- Expansion to support multiple document sources and chunking strategies.

---

This module is designed to be modular and extensible, forming the foundation of a Retrieval
