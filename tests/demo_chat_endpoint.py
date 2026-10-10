import requests

BASE_URL = "http://127.0.0.1:8000"
QUERY = "what is the difference between online and offline evals"

def demo_chat_streaming(query: str) -> None:
    print(f"Query: {query}\n")
    print("Response: ", end="", flush=True)

    with requests.post(
        f"{BASE_URL}/chat",
        json={"query": query},
        stream=True
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            print(chunk, end="", flush=True)

    print("\n")

if __name__ == "__main__":
    demo_chat_streaming()