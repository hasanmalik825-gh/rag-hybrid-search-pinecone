from fastapi import APIRouter, Query, File, UploadFile
from services.rag import load_document, split_document, embedder_by_huggingface, create_or_retrieve_pinecone_index_retriever, llm_by_nvidia
from utils.document_comparison import add_unique_documents
from services.chain import inference_chain_rag
from langchain_core.prompts import ChatPromptTemplate
from constants import RETRIEVE_K

query_document_router = APIRouter()


@query_document_router.post("/query_document")
async def query_document(
    query: str = Query(..., description="Query for the document"),
    file: UploadFile = File(..., description="File to be queried"),
    index_name: str = Query(..., description="Name of the index"),
    retrieve_k: int = Query(None, description="Number of documents to retrieve"),
):
    documents = load_document(file)
    documents = split_document(documents)
    embeddings = embedder_by_huggingface()
    retriever = create_or_retrieve_pinecone_index_retriever(pinecone_index_name=index_name, embeddings=embeddings)
    add_unique_documents(documents=documents, retriever=retriever)
    template = [
        ("system", "You are a helpful assistant that answers concisely. You are given the following context: {context}."),
        ("human", "{input}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=template)
    if retrieve_k is None:
        retrieve_k = RETRIEVE_K
    llm = llm_by_nvidia()
    chain = inference_chain_rag(
        retriever=retriever, 
        llm=llm,
        prompt_template=prompt_template,
        retrieve_k=retrieve_k,
    )
    response = chain.invoke({"input": query})
    return {"chain_response": response["answer"]}
