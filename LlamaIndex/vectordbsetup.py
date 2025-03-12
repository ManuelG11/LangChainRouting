import json
import os
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import numpy as np
from utils.document_loader import load_docs
from utils.config_setup import export_api_keys

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = "faiss_index.index"
CHUNK_DB_PATH = "chunks.json"


def chunk_document(documents, chunk_size=512, overlap=50):
    parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = parser.get_nodes_from_documents(documents)
    return chunks


def create_faiss_index(documents):
    all_chunks = []
    embeddings = []

    chunks = chunk_document(documents)
    for chunk in chunks:
        all_chunks.append({"id": len(all_chunks), "text": chunk.text})
        embeddings.append(embed_model.get_text_embedding(chunk.text))

    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNK_DB_PATH, "w") as f:
        json.dump(all_chunks, f)

    print(f"FAISS index and chunks successfully stored ({len(all_chunks)} chunks).")
    return index


def load_faiss_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNK_DB_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNK_DB_PATH, "r") as f:
            chunks = json.load(f)
        return index, chunks
    else:
        return None, None


def retrieve_relevant_chunks(query, top_k=3):
    index, chunks = load_faiss_index()

    query_embedding = embed_model.get_query_embedding(query)

    D, I = index.search(np.array([query_embedding]), top_k)

    relevant_chunks = [chunks[i]['text'] for i in I[0] if i < len(chunks)]
    return relevant_chunks


if __name__ == "__main__":
    export_api_keys()

    docs = load_docs("./docs", lib="llamaindex")
    create_faiss_index(documents=docs)
    rel = retrieve_relevant_chunks("Explain machine learning")

