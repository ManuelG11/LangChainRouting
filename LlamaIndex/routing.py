import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils.config_setup import export_api_keys
from llama_index.llms.together import TogetherLLM
import os
from vectordbsetup import retrieve_relevant_chunks

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

templates = [
        "How does LlamaIndex work?",
        "What are the LlamaIndex APIs?",
        "How to index documents with LlamaIndex?",
        "LlamaIndex use case examples.",
        "What are the pre-requirements for LlamaIndex?",
        "How to install LlamaIndex on a cloud environment",
        "How to use LlamaIndex for retrieval.",
        "What APIs does LlamaIndex provide?",
        "How to perform query on a LlamaIndex index?",
        "How to update an existing index in LlamaIndex?",
        "How to integrate FAISS with LlamaIndex?",
        "How to improve performance of LlamaIndex?",
        "Best practices to use LlamaIndex with LLMs.",
        "How to implement advanced chunking in LlamaIndex.",
        "How to manage big documents with LlamaIndex."
]

templates_embeddings = np.array([embed_model.get_query_embedding(template) for template in templates])
templates_embeddings = templates_embeddings / np.linalg.norm(templates_embeddings, axis=1, keepdims=True)


def semantic_router(query, threshold=0.8):
    query_embedding = np.array(embed_model.get_query_embedding(query))
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    similarities = np.dot(templates_embeddings, query_embedding)
    max_similarity = np.max(similarities)

    return "llamaindex_specific" if max_similarity > threshold else "generic"


def query_model(query):
    category = semantic_router(query, threshold=0.6)

    if category == "llamaindex_specific":
        context = "\n".join(retrieve_relevant_chunks(query, top_k=5))
        prompt = f"Use this LlamaIndex context to answer the query:\n{context}\n\nQuery: {query}"
    else:
        prompt = f"Query: {query}"

    model = TogetherLLM(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", api_key=os.getenv("TOGETHER_API_KEY"))
    print(prompt)
    return model.complete(prompt)


if __name__ == "__main__":
    export_api_keys()
    queries = [
        "How do I index documents in LlamaIndex?",
        "How to develop a chatbot with LlamaIndex",
        "How to develop a chatbot",
        "Capital city of Spain"
    ]

    for query in queries:
        print("--------------------------")
        print(query_model(query))
        print("--------------------------")
