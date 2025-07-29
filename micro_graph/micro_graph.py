from typing import Any
import asyncio
from abc import ABC, abstractmethod

NodeResult = dict[str, Any] | None


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


class Node(ABC):
    """
    A base class for nodes in a micro-graph, which can be extended to implement specific functionality.
    A node can have multiple next nodes and can be configured to retry on failure.

    You should overwrite the `run` method to define the node's behavior.
    """
    def __init__(self, max_retries: int = 0):
        self._next_nodes: dict[str, Node] = {}
        self._max_retries = max_retries

    @abstractmethod
    async def run(self, shared: dict, **kwargs) -> NodeResult | tuple[str, NodeResult] | str:
        pass

    def then(self, default: 'Node', **kwargs) -> 'Node':
        self._next_nodes["default"] = default
        self._next_nodes.update(kwargs)
        return default

    async def __call__(self, shared: dict, **kwargs) -> NodeResult:
        res = await _run_with_retries(self.run, self._max_retries, shared=shared, **kwargs)
        if isinstance(res, tuple):
            action, res = res
        elif isinstance(res, str):
            action, res = res, {}
        else:
            action, res = "default", res
        if action in self._next_nodes:
            return await self._next_nodes[action](shared, **res)
        return res


class ParalellNode(Node):
    """
    A node that processes a batch of tasks in paralell, allowing for retries on each task.

    You should overwrite `prep`, `run_single`, and `post` methods to define the processing logic.
    `prep` splits the input into tasks, `run_single` processes a single task, and `post` combines the results.
    """
    def __init__(self, max_retries: int = 0, max_retries_per_batch: int = 0):
        super().__init__(max_retries)
        self._max_retries_per_batch = max_retries_per_batch

    @abstractmethod
    async def prep(self, shared: dict, **kwargs) -> list[NodeResult]:
        pass

    @abstractmethod
    async def run_single(self, shared: dict, **kwargs) -> NodeResult:
        pass

    @abstractmethod
    async def post(self, shared: dict, results: list[NodeResult]) -> NodeResult | tuple[str, NodeResult] | str:
        pass

    async def run(self, shared, **kwargs) -> NodeResult | tuple[str, NodeResult] | str:
        batch_args = await self.prep(shared, **kwargs)
        batch_tasks = [
            _run_with_retries(self.run_single, self._max_retries_per_batch, shared=shared, **(result or {}))
            for result in batch_args
        ]
        results = await asyncio.gather(*batch_tasks)
        return await self.post(shared, list(results))
