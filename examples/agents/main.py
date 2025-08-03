import dotenv
from micro_graph.ai.openai_server import serve, wrap_agent
from micro_graph.ai.llm import get_llm_and_model_from_env

from examples.agents.planner import planner_agent


def main():
    # Setup
    agents = {}
    nodes = {}
    dotenv.load_dotenv()
    llm, model = get_llm_and_model_from_env()

    # Add agents
    agents["planner"], nodes["plan"] = planner_agent(llm, model=model, max_iterations=3)

    # Serve agents via chat api
    serve({k: wrap_agent(llm, model, v) for k, v in agents.items()})


if __name__ == "__main__":
    main()
