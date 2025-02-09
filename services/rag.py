from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.documents import Document
from starlette.datastructures import UploadFile as FastAPIUploadFile
from constants import NVIDIA_API_KEY, PINECONE_API_KEY
from typing import List, Union
import tempfile
import os

def load_document(file: FastAPIUploadFile) -> List[Document]:
    """
    Handles file loading for FastAPI's UploadFile.

    Args:
        file: FastAPI's UploadedFile.

    Returns:
        Loaded document.
    """

    file_name = file.filename
    file_content = file.file.read()

    # Create a temporary file for processing
    file_extension = file_name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    # Determine the loader based on file type
    if file_extension == "pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == "txt":
        loader = TextLoader(temp_file_path)
    else:
        os.remove(temp_file_path)  # Cleanup
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        # Load the document
        document = loader.load()
    finally:
        # Ensure temporary file is cleaned up
        os.remove(temp_file_path)

    return document

def split_document(documents: List[Document]) -> List[Document]:
    """
    This function is used to split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# For this open-source option if someone intend to use it.
def embedder_by_huggingface(model_name: str = 'all-MiniLM-L12-v2') -> HuggingFaceEmbeddings:
    """
    This function is used to embed the documents using huggingface embedding model.
    """
    embeddings=HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def llm_by_nvidia(
        model: str = "deepseek-ai/deepseek-r1",
        api_key: str = NVIDIA_API_KEY
    ) -> ChatNVIDIA:
    """
    This function is used to create a llm using nvidia llm model. By default deepseek.
    """
    llm = ChatNVIDIA(model=model, api_key=api_key)
    return llm

def create_or_retrieve_pinecone_index_retriever(
        embeddings: HuggingFaceEmbeddings,
        pinecone_index_name: str,
) -> Pinecone:
    """
    This function is used to create or load the pinecone vector store.
    If the provided index exists, it will load the index.
    If user provided name does not exist, it will create a new index.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=384,  # dimensionality of dense model
            metric="dotproduct",  # sparse values supported only for dotproduct
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index=pc.Index(pinecone_index_name)
    retriever=PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=BM25Encoder().default(),index=index)
    return retriever