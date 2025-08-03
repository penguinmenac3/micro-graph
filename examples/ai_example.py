import os
import dotenv
from micro_graph.ai.llm import LLM
from micro_graph.ai.openai_server import serve

def example():
    dotenv.load_dotenv()
    llm = LLM(
        api_endpoint=os.environ.get("API_ENDPOINT", "http://localhost:11434"),
        api_key=os.environ.get("API_KEY", "ollama"),
        provider=os.environ.get("PROVIDER", "ollama"),
        model=os.environ.get("MODEL", "AUTODETECT")
    )
    serve(llm, debug=True)

if __name__ == "__main__":
    example()
