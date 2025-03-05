import os
import configparser
import pickle

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.utils.math import cosine_similarity

config = configparser.ConfigParser()
config.read("config.ini")

if "API" not in config or "TOGETHER_API_KEY" not in config["API"]:
    raise ValueError("API key not found")

os.environ["TOGETHER_API_KEY"] = config["API"]["TOGETHER_API_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    openai_api_key=os.environ["TOGETHER_API_KEY"],
    openai_api_base="https://api.together.xyz/v1"
)

if not os.path.exists("faiss_index"):
    raise FileNotFoundError("VectorDB not found!")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

TEMPLATE_PICKLE = "./faiss_index/representative_templates.pkl"
with open(TEMPLATE_PICKLE, "rb") as f:
    representative_templates = pickle.load(f)

template_embeddings = np.array([embedding_model.embed_query(t) for t in representative_templates])

rag_prompt_format = PromptTemplate.from_template(
    "LangChain documentation:\n{context}\n\nQuestion: {query}\nAnswer:"
)

generic_prompt_format = PromptTemplate.from_template("Question: {query}\nAnswer:")


def generate_rag_response(query: str, k: int) -> str:
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = rag_prompt_format.format(query=query, context=context)
    return prompt


def generate_generic_response(query: str) -> str:
    prompt = generic_prompt_format.format(query=query)
    return prompt


def query_router(inputs: dict) -> str:
    query = inputs.get("query", "").lower()
    query_embedding = np.array(embedding_model.embed_query(query)).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, template_embeddings)[0]
    similarity_threshold = 0.6  # this parameter can be learnt through i.e. reinforcement learning

    if np.max(similarities) > similarity_threshold:
        print("Specific branch (RAG)")
        return generate_rag_response(query, 10)
    else:
        print("Generic branch")
        return generate_generic_response(query)


chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(query_router)
        | llm
        | StrOutputParser()
)

# you can type your queries here
test_queries = [
    "Can you write me some python code to develop a chatbot with LangChain?",
    "Explain me the architecture in Langchain",
    "How to develop a chat bot",
    "Capital city of Spain"
]

for query in test_queries:
    print(f"-------------------------------------------------------------------------------------\nQuestion: {query}")
    output = chain.invoke(query)
    print(f"Answer: {output}\n-------------------------------------------------------------------------------------")
