import jwt
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

import os


def check_apptoken_from_apikey(apikey: str):
    if not apikey:
        return None
    apisecret = os.environ.get('APP_SECRET')
    if apikey:
        try:
            payload = jwt.decode(apikey, apisecret, algorithms=['HS256'])
            uid = payload.get('uid')
            if uid :
                return uid
        except Exception as e:
            return None
    return None

def get_global_datadir(subpath: str = None):
    """
    获取全局数据目录。

    Args:
        subpath (str, optional): 子路径。默认为None。

    Returns:
        str: 数据目录路径。
    """
    datadir = os.environ.get("DATA_DIR", "/tmp/teamsgpt")
    if subpath:
        datadir = os.path.join(datadir, subpath)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    return datadir



def get_azure_llm():
    return AzureOpenAI(
        model="gpt-4-turbo-preview",
        deployment_name="gpt-4-turbo-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )


def get_azure_embedding():
    return AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        deployment_name="text-embedding-3-large",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )


def write_stream_text(placeholder, response):
    """写入流式响应。"""
    full_response = ""
    for token in response.response_gen:
        text = token
        if text is not None:
            full_response += text
            placeholder.markdown(full_response)
        placeholder.markdown(full_response)
    return full_response

