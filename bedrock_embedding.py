# bedrock_embedding.py
import os
import json
import boto3
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class BedrockEmbedding:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.model_id = os.getenv("BEDROCK_EMBD_MODEL_ID")

    def embed(self, text: str):
        body = {"inputText": text}

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        result = json.loads(response["body"].read())
        return np.array(result["embedding"], dtype=np.float32)