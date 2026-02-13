import os
from pathlib import Path
import boto3
from dotenv import load_dotenv

from langchain_aws import BedrockEmbeddings
from bedrock_llm import BedrockLLM

load_dotenv()

boto3.setup_default_session(
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
llm = BedrockLLM()

VECTORSTORE_ROOT = Path("vectorstores")
VECTORSTORE_ROOT.mkdir(exist_ok=True)

MAX_CONTEXT_CHARS = 6000
