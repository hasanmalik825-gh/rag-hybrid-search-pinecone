from langchain_core.runnables.base import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)

def inference_chain_rag(
        llm: ChatNVIDIA,
        retriever: PineconeHybridSearchRetriever,
        prompt_template: PromptTemplate,
        retrieve_k: int,
    ) -> Runnable:
    """
    This function integrates a re-ranker into the Retrieval-QA chain.
    Args:
        llm: nvidia llm
        vectorstorage: vector store
        prompt_template: prompt template
        reranker: reranker from langchain_nvidia_ai_endpoints
        retrieve_k: number of initial documents to retrieve from vector store
        rerank_top_n: number of top-ranked documents to use after re-ranking
    """
    # Create a retriever from the vector store
    retriever.top_k = retrieve_k
    
    # Create the QA chain
    qa_chain = create_stuff_documents_chain(
        llm=llm, 
        prompt=prompt_template,
    )

    # Combine retriever with QA chain
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    return retrieval_chain