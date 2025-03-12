import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.document_loader import load_docs

DOCS_DIR = "docs"
TEMPLATE_PICKLE = "./faiss_index/representative_templates.pkl"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_representative_templates(vectordb, num_templates=5):
    retriever = vectordb.as_retriever(search_kwargs={"k": num_templates})
    query = "concepts, tutorials, how-to, errors"
    relevant_docs = retriever.invoke(query)

    templates = [doc.page_content for doc in relevant_docs]
    return templates


docs = load_docs(DOCS_DIR, lib="langchain")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
split_docs = splitter.split_documents(docs)

vectordb = FAISS.from_documents(split_docs, embedding_model)
vectordb.save_local("faiss_index")

print(f"VectorDB created with {len(split_docs)} chunks!")

representative_templates = get_representative_templates(vectordb, num_templates=10)
with open(TEMPLATE_PICKLE, "wb") as f:
    pickle.dump(representative_templates, f)
