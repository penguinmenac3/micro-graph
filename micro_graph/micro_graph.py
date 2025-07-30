from typing import Any, Callable, Coroutine
import asyncio

GraphResult = dict[str, Any] | None
NodeResult = GraphResult | tuple[str, GraphResult] | str
RunFunction = Callable[[dict], Coroutine[Any, Any, NodeResult]]


async def _run_with_retries(function, max_retries, **kwargs):
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    e = RuntimeError(f"Execution failed after {max_retries} retries.")
    for _ in range(max_retries + 1):
        try:
            return await function(**kwargs)
        except Exception as exc:
            e = exc
    raise e


class Node:
    """
    A nodes in a micro-graph allows connection to other nodes by using `then`
    and the `run` defines what happens when a node is executed.
    Optionally for paralell processing:
        `prep` defines the task inputs, `run` processes a single task, and `post` combines results.
    """

    def __init__(self, run: RunFunction | None = None, max_retries: int = 0):
        self._next_nodes: dict[str, Node] = {}
        self._max_retries = max_retries
        if run is not None:
            self.run = run  # type: ignore

    def then(self, default: "Node", **kwargs) -> "Node":
        self._next_nodes["default"] = default
        self._next_nodes.update(kwargs)
        return default

    async def prep(self, shared: dict, **kwargs) -> list[GraphResult]:
        return [kwargs]

    async def run(self, shared: dict, **kwargs) -> NodeResult:
        return None

    async def post(self, shared: dict, results: list[NodeResult]) -> NodeResult:
        return results[0] if results else None

    async def __call__(self, shared: dict, only_this_node=False, **kwargs) -> GraphResult:
        tasks: list[GraphResult] = await self.prep(shared, **kwargs)
        task_results: list[NodeResult] = list(await asyncio.gather(*[
            _run_with_retries(
                self.run, self._max_retries, shared=shared, **(result or {})
            )
            for result in tasks
        ]))
        result: NodeResult = await self.post(shared, task_results)
        if isinstance(result, tuple):
            action, result = result
        elif isinstance(result, str):
            action, result = result, {}
        else:
            action, result = "default", result
        if not only_this_node and action in self._next_nodes:
            return await self._next_nodes[action](shared, only_this_node, **(result or {}))
        return result
