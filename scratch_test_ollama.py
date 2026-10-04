import httpx
import json
import sys

def test_ollama():
    print("1. Checking if Ollama is reachable...")
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            print(" -> Success! Ollama is running.")
            models = resp.json().get("models", [])
            print(f" -> Found {len(models)} models locally.")
        else:
            print(f" -> Failed: HTTP {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f" -> Failed to connect: {e}")
        sys.exit(1)

    model_to_test = "llava"
    print(f"\n2. Checking for model '{model_to_test}'...")
    has_model = any(m.get("name") == model_to_test or m.get("name") == f"{model_to_test}:latest" for m in models)
    
    if not has_model:
        print(f" -> Model '{model_to_test}' not found. Pulling via API (this may take a bit)...")
        pull_resp = httpx.post("http://localhost:11434/api/pull", json={"name": model_to_test}, timeout=900.0)
        if pull_resp.status_code == 200:
            print(f" -> Successfully pulled '{model_to_test}'.")
        else:
            print(f" -> Failed to pull: {pull_resp.text}")
            sys.exit(1)
    else:
        print(f" -> Model '{model_to_test}' is already downloaded.")

    print(f"\n3. Sending a 'hi' message to '{model_to_test}'...")
    payload = {
        "model": model_to_test,
        "messages": [
            {"role": "user", "content": "Hi! Are you online?"}
        ],
        "stream": False
    }
    
    try:
        chat_resp = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=60.0)
        if chat_resp.status_code == 200:
            reply = chat_resp.json()["message"]["content"]
            print(f"\n[Ollama Reply]: {reply}")
        else:
            print(f"\n -> Error sending message: HTTP {chat_resp.status_code} - {chat_resp.text}")
    except Exception as e:
        print(f"\n -> Failed during chat: {e}")

if __name__ == "__main__":
    test_ollama()
