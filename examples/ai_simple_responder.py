import dotenv
from micro_graph import OutputWriter
from micro_graph.ai.llm import get_llm_and_model_from_env
from micro_graph.ai.openai_server import serve
from micro_graph.ai.types import ChatMessage


def example():
    dotenv.load_dotenv()
    llm, model = get_llm_and_model_from_env()

    async def run_planner(
        output: OutputWriter, chat_messages: list[ChatMessage], max_tokens: int
    ):
        response = llm.chat_stream(
            model,
            chat_messages[-10:],
            max_tokens=max_tokens,
        )
        for chunk in response:
            output.default(chunk, end="")

    serve({"simple-responder": run_planner})

if __name__ == "__main__":
    example()
