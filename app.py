import os
from openai import OpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if __name__ == "__main__":
    if not GROQ_API_KEY:
        raise ValueError("Set GROQ_API_KEY first. Get one free at https://console.groq.com/keys")

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    MODEL = "openai/gpt-oss-20b"

    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            break
        if not prompt:
            continue

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            print(f"Bot: {response.choices[0].message.content}")
        except Exception as e:
            print(f"Error: {e}")