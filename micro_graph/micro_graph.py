from typing import Any, Callable, Coroutine
import asyncio

from micro_graph.output_writer import OutputWriter

GraphResult = dict[str, Any] | None
NodeResult = GraphResult | tuple[str, GraphResult] | str
RunFunction = Callable[[OutputWriter, dict], Coroutine[Any, Any, NodeResult]]


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


def template_formatting(template: str, shared: dict, **kwargs) -> str:
    """
    Fill a template string using keys from shared and kwargs.

    A template string looks like this "Hello {user}!".
    Where `user` can be either provided in shared or kwargs.
    """
    context = {**shared, **kwargs}
    try:
        return template.format(**context)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(f"Missing key '{missing}' for template formatting.") from e


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

    async def prep(self, output: OutputWriter, shared: dict, **kwargs) -> list[GraphResult]:
        return [kwargs]

    async def run(self, output: OutputWriter, shared: dict, **kwargs) -> NodeResult:
        return None

    async def post(
        self, output: OutputWriter, shared: dict, results: list[NodeResult]
    ) -> NodeResult:
        return results[0] if results else None

    async def __call__(
        self, output: OutputWriter, shared: dict, only_this_node=False, **kwargs
    ) -> GraphResult:
        tasks: list[GraphResult] = await self.prep(output, shared, **kwargs)
        task_results: list[NodeResult] = list(
            await asyncio.gather(
                *[
                    _run_with_retries(
                        self.run, self._max_retries, output=output, shared=shared, **(result or {})
                    )
                    for result in tasks
                ]
            )
        )
        result: NodeResult = await self.post(output, shared, task_results)
        if isinstance(result, tuple):
            action, result = result
        elif isinstance(result, str):
            action, result = result, {}
        else:
            action, result = "default", result
        if not only_this_node:
            action = action.lower()
            if action not in self._next_nodes:
                if action == "default":
                    return result
                raise KeyError(
                    f"Action '{action}' not found in next nodes: {list(self._next_nodes.keys())}"
                )
            return await self._next_nodes[action](output, shared, only_this_node, **(result or {}))
        return result
