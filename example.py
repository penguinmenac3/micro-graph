import asyncio
from micro_graph import Node

async def example():
    class OutputMessage(Node):
        def __init__(self, message: str):
            super().__init__()
            self.message = message

        async def run(self, shared: dict, **kwargs):
            return {"message": self.message}

    class Loop(Node):
        async def run(self, shared: dict, iter: int = 0, **kwargs):
            if iter < 5:
                print(f"Loop iteration {iter}")
                return "default", {"iter": iter + 1}
            else:
                #raise RuntimeError("Reached maximum iterations, exiting loop.")
                return "exit"

    # Build the flow with a loop and an output message
    loop = Loop(max_retries=1)
    hello_world = OutputMessage("Hello World!")
    loop.then(default=loop, exit=hello_world)

    # Run the flow by executing the start node (here loop)
    print(await loop({}))

if __name__ == "__main__":
    asyncio.run(example())
