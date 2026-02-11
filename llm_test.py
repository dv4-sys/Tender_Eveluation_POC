import os
import sys
from bedrock_llm import BedrockLLM


def main():
    print("Checking environment variables:")
    print("  AWS_REGION:", os.getenv("AWS_REGION"))
    print("  AWS_ACCESS_KEY_ID:", bool(os.getenv("AWS_ACCESS_KEY_ID")))
    print("  AWS_SECRET_ACCESS_KEY:", bool(os.getenv("AWS_SECRET_ACCESS_KEY")))
    print("  BEDROCK_MODEL_ID:", os.getenv("BEDROCK_MODEL_ID"))

    try:
        llm = BedrockLLM()
    except Exception as e:
        print("Failed to initialize BedrockLLM:", e)
        sys.exit(2)

    prompt_text = (
        "Please respond with a short greeting and then output a JSON object only on a new line:\n"
        "First a one-line greeting, then on the next line ONLY valid JSON: {\"status\":\"ok\", \"message\":\"hello\"}"
    )

    print("\nInvoking LLM (raw)...")
    try:
        raw = llm.invoke(prompt_text)
        print("RAW OUTPUT:\n", raw)
    except Exception as e:
        print("LLM invoke failed:", e)
        sys.exit(3)

    print("\nAttempting strict JSON parse using `invoke_json` (expects JSON output):")
    try:
        # A prompt designed to get JSON-only output
        json_prompt = "Please output ONLY this JSON: {\"status\": \"ok\", \"message\": \"hello_json\"}"
        parsed = llm.invoke_json(json_prompt)
        print("PARSED JSON:\n", parsed)
    except Exception as e:
        print("invoke_json failed:", e)


if __name__ == "__main__":
    main()
