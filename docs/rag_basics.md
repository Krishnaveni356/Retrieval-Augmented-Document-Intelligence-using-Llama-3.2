# RAG basics

Retrieval-Augmented Generation (RAG) gives a language model access to your own documents at answer time.

The pipeline has two phases. During ingestion, documents are split into chunks, each chunk is converted into an embedding vector, and the vectors are stored in a vector database. During querying, the user's question is embedded, the most similar chunks are retrieved, and they are placed in the prompt so the model can answer from them.

## Chunking

Chunk size is a trade-off. Small chunks give precise retrieval but may lose context. Large chunks keep context but dilute the embedding. A common starting point is about 800 characters with 15 to 20 percent overlap.

## Embeddings

An embedding model maps text to a vector so that similar meaning lands close together. Cosine similarity is the usual distance measure.

## Evaluation

Measure retrieval separately from generation. Hit rate at k asks whether the right document appears in the top k results. Mean reciprocal rank rewards ranking the right document first.
