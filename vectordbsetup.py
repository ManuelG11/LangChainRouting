import json
import os
import pickle
import re
import markdown
from bs4 import BeautifulSoup
import nbformat
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "./docs"
TEMPLATE_PICKLE = "./faiss_index/representative_templates.pkl"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def extract_markdown_content(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = markdown.markdown(f.read())
        text_content = BeautifulSoup(md_content, "html.parser").get_text()
        return text_content.strip()


def clean_markdown(text):
    text = re.sub(r'\[([^]]+)]\([^)]+\)', r'\1', text)
    return text


def extract_tables_from_markdown(text):
    table_pattern = r'((?:\|[^\n]+\|\n)+)'
    tables = re.findall(table_pattern, text)
    return tables if tables else []


def extract_text_and_code_from_ipynb(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    extracted_content = []

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cleaned_text = clean_markdown(cell.source)
            tables = extract_tables_from_markdown(cleaned_text)
            extracted_content.append(cleaned_text)
            for table in tables:
                extracted_content.append(table)

        elif cell.cell_type == "code":
            extracted_content.append(cell.source)

            for output in cell.get("outputs", []):
                if "text/html" in output.get("data", {}):
                    extracted_content.append(output["data"]["text/html"])
                elif "application/json" in output.get("data", {}):
                    extracted_content.append(json.dumps(output["data"]["application/json"], indent=2))

    return "\n\n".join(extracted_content)


def extract_python_content(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_docs(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            content = ""

            if file.endswith(".mdx") or file.endswith(".md"):
                content = extract_markdown_content(file_path)
            elif file.endswith(".ipynb"):
                content = extract_text_and_code_from_ipynb(file_path)
            elif file.endswith(".py"):
                content = extract_python_content(file_path)
            else:
                continue

            if content:
                documents.append(Document(page_content=content, metadata={"source": file_path}))

    return documents


def get_representative_templates(vectordb, num_templates=5):
    retriever = vectordb.as_retriever(search_kwargs={"k": num_templates})
    query = "concepts, tutorials, how-to, errors"
    relevant_docs = retriever.invoke(query)

    templates = [doc.page_content for doc in relevant_docs]
    return templates


docs = load_docs(DOCS_DIR)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
split_docs = splitter.split_documents(docs)

vectordb = FAISS.from_documents(split_docs, embedding_model)
vectordb.save_local("faiss_index")

print(f"VectorDB created with {len(split_docs)} chunks!")

representative_templates = get_representative_templates(vectordb, num_templates=10)
with open(TEMPLATE_PICKLE, "wb") as f:
    pickle.dump(representative_templates, f)
