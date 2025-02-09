import hashlib
from typing import List
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)
from langchain_core.documents import Document

def _get_string_hash(input_string: str) -> str:
    sha256_hash = hashlib.sha256()
    # Convert the string to bytes and update the hash
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

def _get_ids(retriever: PineconeHybridSearchRetriever):
    # Access the underlying Pinecone index
    index = retriever.index

    # List all IDs
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    if total_vectors == 0:
        return []

    # Create a zero vector of the same dimension as your embeddings
    # Usually 1536 for OpenAI embeddings, adjust if using different embeddings
    zero_vector = [0] * 384

    results = index.query(
        vector=zero_vector,
        top_k=total_vectors,
        include_metadata=True,
        namespace=""  # specify namespace if you're using one
    )
    all_ids = [match.id for match in results.matches]
    return all_ids

def add_unique_documents(documents: List[Document], retriever: PineconeHybridSearchRetriever):

    doc_ids = [_get_string_hash(doc.page_content) for doc in documents]
    documents = [(doc, doc_ids[idx]) for idx, doc in enumerate(documents) if doc_ids[idx] not in _get_ids(retriever)]
    if len(documents) > 0:
        retriever.add_texts(texts=[doc[0].page_content for doc in documents], ids=[doc[1] for doc in documents])
        print(f"Added {len(documents)} unique documents to the vector store. (pinecone)")
    else:
        print("All documents already exist in the vector store. (pinecone)")