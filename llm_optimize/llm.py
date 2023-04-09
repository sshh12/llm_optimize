from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

LLMModel = BaseChatModel

default_llm_options = dict(model_name="gpt-4")


def get_default_llm() -> LLMModel:
    chat = ChatOpenAI(**default_llm_options)
    return chat
