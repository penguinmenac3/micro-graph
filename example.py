import asyncio
from micro_graph import Node, NodeResult


async def example() -> None:
    async def hello_world(shared: dict, **kwargs) -> NodeResult:
        return {"message": "Hello World!"}

    async def loop(shared: dict, iter: int = 0, **kwargs) -> NodeResult:
        if iter < 5:
            print(f"Loop iteration {iter}")
            return "default", {"iter": iter + 1}
        else:
            # raise RuntimeError("Reached maximum iterations, exiting loop.")
            return "exit"

    # Build the flow with a loop and an output message
    loop_node = Node(run=loop, max_retries=1)
    hello_world_node = Node(run=hello_world)
    loop_node.then(default=loop_node, exit=hello_world_node)

    # Run the flow by executing the start node (here loop)
    print(await loop_node({}, only_this_node=True))
    print(await loop_node({}))


if __name__ == "__main__":
    asyncio.run(example())
