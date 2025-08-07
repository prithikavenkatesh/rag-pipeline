

# Semantic Search Module

This folder is part of the `rag-pipeline` project. It contains code for searching PDF documents based on meaning, not just keywords.

## What It Does

1. **Extracts paragraphs from PDFs**
   - Uses PyMuPDF to read and split text into paragraphs.

2. **Generates semantic embeddings**
   - Uses the `sentence-transformers` model to turn paragraphs into vectors that capture meaning.

3. **Stores data in Milvus**
   - Embeddings and text are saved in a Milvus vector database.
   - An index is created for fast searching.

4. **Performs semantic search**
   - You can enter a question or phrase.
   - The system finds the most relevant paragraphs based on meaning.

## Sample PDF

- The included PDF (ST_SGB_2019_7-EN.pdf) is a public UN environmental policy document.
- It’s used to demonstrate how semantic search works.
- Source: UN Digital Library (https://digitallibrary.un.org/record/3827063?v=pdf#files)

## Future Plans

- Add LLMs like Qwen or DeepSeek to generate answers from retrieved content.
- Support more document types and smarter chunking.

---

This module is designed to be modular and easy to extend as part of a larger Retrieval-Augmented Generation (RAG) system.

