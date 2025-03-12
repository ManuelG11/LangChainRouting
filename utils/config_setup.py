import configparser
import os


def export_api_keys(tokenizer_parallelism=False):
    config = configparser.ConfigParser()
    config.read("../config.ini")

    if "API" not in config:
        raise ValueError("API keys not found")
    if "OPENAI_API_KEY" not in config["API"]:
        raise ValueError("OpenAI key not found")
    if "TOGETHER_API_KEY" not in config["API"]:
        raise ValueError("TogetherAI key not found")

    os.environ["OPENAI_API_KEY"] = config["API"]["OPENAI_API_KEY"]
    os.environ["TOGETHER_API_KEY"] = config["API"]["TOGETHER_API_KEY"]
    if not tokenizer_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
