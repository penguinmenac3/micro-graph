import pytest
from micro_graph import Node, NodeResult, OutputWriter


@pytest.fixture
def graph() -> Node:
    async def hello_world(output: OutputWriter, shared: dict, **kwargs) -> NodeResult:
        return {"message": "Hello World!"}

    async def loop(
        output: OutputWriter, shared: dict, iter: int = 0, **kwargs
    ) -> NodeResult:
        if iter < 5:
            return "default", {"iter": iter + 1}
        else:
            return "exit"

    loop_node = Node(run=loop, max_retries=1)
    hello_world_node = Node(run=hello_world)
    loop_node.then(default=loop_node, exit=hello_world_node)
    return loop_node


@pytest.fixture
def output() -> OutputWriter:
    return OutputWriter()


@pytest.mark.asyncio
async def test_node_only(graph, output):
    node_only_result = await graph(output, {}, only_this_node=True)
    assert node_only_result == {'iter': 1}, f"Expected '{{'iter': 1}}', got {node_only_result}"


@pytest.mark.asyncio
async def test_full_execution(graph, output):
    result_full = await graph(output, {})
    assert result_full == {"message": "Hello World!"}, f"Expected Hello World message, got {result_full}"
