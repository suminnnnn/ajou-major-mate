from typing import List, Optional, Dict
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, FilterSelector 
from langchain_core.documents import Document

COLLECTION_NAME = "ajou_documents"

VECTOR_SIZE_BY_MODEL = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072, 
}

client = QdrantClient(host="qdrant", port=6333)

def ensure_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE_BY_MODEL["text-embedding-3-large"],
                distance=Distance.COSINE
            )
        )
        

def add_documents(domain: str, docs: List[Document]):
    ensure_collection()
    for doc in docs:
        doc.metadata["domain"] = domain
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    vectordb.add_documents(docs)


def delete_documents(domain: str):
    ensure_collection()
    
    filter = models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.domain",
                match=models.MatchValue(
                    value=domain
                ),
            ),
        ]
    )
    
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(filter=filter)
    )
    

def similarity_search(
    query: str,
    domain: str,
    k: int = 5,
    metadata_filters: Optional[Dict[str, str]] = None
) -> List[Dict]:
    ensure_collection()

    conditions = [FieldCondition(key="metadata.domain", match=MatchValue(value=domain))]
    if metadata_filters:
        for key, value in metadata_filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    filter = Filter(should=conditions)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    docs: List[Document] = vectordb.similarity_search(query=query, k=k, filter=filter)
    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]

def similarity_search_multiple_departments(
    query: str,
    domain: str,
    departments: List[str],
    per_department_k: int = 3
) -> List[Dict]:
    ensure_collection()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    all_results = []

    for dept in departments:
        filter = Filter(
            must=[
                FieldCondition(key="metadata.domain", match=MatchValue(value=domain)),
                FieldCondition(key="metadata.department", match=MatchValue(value=dept)),
            ]
        )
        docs: List[Document] = vectordb.similarity_search(query=query, k=per_department_k, filter=filter)
        all_results.extend([{"text": doc.page_content, "metadata": doc.metadata} for doc in docs])

    return all_results
