# bedrock_llm.py
import json
import os
import re
import boto3
from dotenv import load_dotenv

load_dotenv()

class BedrockLLM:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.model_id = os.getenv("BEDROCK_MODEL_ID")

    def invoke(self, prompt: str) -> str:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        result = json.loads(response["body"].read())

        # OpenAI-style models
        if "choices" in result:
            content = result["choices"][0]["message"]["content"]
            return self._clean_text(content)

        # Anthropic-style
        if "output" in result:
            return self._clean_text(
                result["output"]["message"]["content"][0]["text"]
            )

        # Titan-style
        if "content" in result:
            return self._clean_text(result["content"][0]["text"])

        raise RuntimeError(f"Unexpected Bedrock response: {result}")

    def invoke_json(self, prompt: str):
        """
        Strict JSON invocation.
        This is the ONLY place json.loads() is allowed.
        """
        raw = self.invoke(prompt)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM did not return valid JSON.\n\nRAW OUTPUT:\n{raw[:1500]}"
            ) from e

    def _clean_text(self, text: str) -> str:
        # Remove reasoning blocks
        text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.S)

        # Remove markdown fences
        text = text.replace("```json", "").replace("```", "")

        return text.strip()
