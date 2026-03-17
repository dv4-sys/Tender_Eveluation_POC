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
                    "content": prompt,
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
            return self._clean_text(result["output"]["message"]["content"][0]["text"])

        # Titan-style
        if "content" in result:
            return self._clean_text(result["content"][0]["text"])

        raise RuntimeError(f"Unexpected Bedrock response: {result}")

    def invoke_json(self, prompt: str, retries: int = 3):
        """
        Robust JSON invocation for GenAI flows.

        Handles:
        - Extra explanation text
        - Markdown fences
        - Single quotes
        - Trailing commas
        - Truncated JSON
        - Empty dict fallback
        - Safe retry without re-injecting raw model output
        """

        last_error = None
        original_prompt = prompt
        raw = ""

        for attempt in range(retries + 1):
            try:
                raw = self.invoke(prompt)
            except Exception as invoke_error:
                last_error = invoke_error
                if attempt < retries:
                    prompt = self._build_json_repair_prompt(original_prompt, str(invoke_error))
                    continue
                raise

            if not raw or raw.strip() == "":
                last_error = "Empty response from LLM"
                if attempt < retries:
                    prompt = self._build_json_repair_prompt(original_prompt, str(last_error))
                    continue
                break

            cleaned = raw.strip()
            cleaned = cleaned.replace("`json", "").replace("`", "")

            # 1) Direct parse
            try:
                parsed = json.loads(cleaned)
                if parsed == {}:
                    raise ValueError("LLM returned empty JSON object.")
                return parsed
            except Exception as e:
                last_error = e

            # 2) Extract JSON block
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
            if match:
                json_text = match.group(1).strip()
                json_text = re.sub(r"(?<!\\)'", '"', json_text)
                json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
                json_text = re.sub(r"[\x00-\x1f]+", "", json_text)
                try:
                    parsed = json.loads(json_text)
                    if parsed == {}:
                        raise ValueError("LLM returned empty JSON object.")
                    return parsed
                except Exception as e:
                    last_error = e

            # 3) Auto-close truncated JSON
            auto = cleaned
            open_brackets = auto.count("{")
            close_brackets = auto.count("}")
            open_square = auto.count("[")
            close_square = auto.count("]")
            if open_brackets > close_brackets:
                auto += "}" * (open_brackets - close_brackets)
            if open_square > close_square:
                auto += "]" * (open_square - close_square)

            try:
                parsed = json.loads(auto)
                if parsed == {}:
                    raise ValueError("LLM returned empty JSON object.")
                return parsed
            except Exception:
                pass

            if attempt < retries:
                prompt = self._build_json_repair_prompt(original_prompt, str(last_error))
            else:
                break

        raise ValueError(
            f"\nLLM failed to return valid JSON after {retries + 1} attempts.\n\n"
            f"Last error: {last_error}\n\n"
            f"Raw output:\n{raw[:2000]}"
        )

    def _build_json_repair_prompt(self, original_prompt: str, error_text: str) -> str:
        safe_original = original_prompt[:18000]
        safe_error = (error_text or "Unknown parsing error")[:500]
        return f"""
You produced invalid JSON.

ERROR:
{safe_error}

Now regenerate ONLY valid JSON for the original task below.

Rules:
- Double quotes only
- No trailing commas
- No explanation text
- No markdown
- Output must start with {{ or [
- Output must end with }} or ]

Original task:
{safe_original}
"""

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.S)
        text = text.replace("`json", "").replace("`", "")
        return text.strip()