import json
import os
import re
import markdown
import nbformat
from bs4 import BeautifulSoup
from langchain_core.documents import Document as DocumentLangChain
from llama_index.core import Document as DocumentLlamaIndex


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


def load_docs(directory, lib="langchain"):
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
                if lib == "langchain":
                    documents.append(DocumentLangChain(page_content=content, metadata={"source": file_path}))
                elif lib == "llamaindex":
                    documents.append(DocumentLlamaIndex(text=content, metadata={"source": file_path}))

    return documents
