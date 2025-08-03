from asyncio import Queue


class OutputWriter:
    def __init__(self, queue: Queue | None = None):
        self._state = "default"
        self._topic = "default"
        self._queue = queue

    def thought(self, text: str, end="\n\n") -> None:
        self.write(text + end, message_type="thought")

    def default(self, text: str, end="\n\n") -> None:
        self.write(text + end, message_type="default")

    def detail(self, topic: str, text: str, end="\n\n") -> None:
        self.write(text + end, message_type=topic)

    def write(self, text: str, message_type: str | None = None) -> None:
        if message_type is not None:
            self._change_state(message_type)
        if self._queue is not None:
            self._queue.put_nowait(text)
        else:
            print(text, end="")

    def _change_state(self, state: str) -> None:
        topic: str = state
        state = state if state in ["default", "thought"] else "detail"
        if state != self._state:
            if self._state == "thought":
                self.write("\n</think>\n")
            elif self._state == "detail":
                self.write("\n</details>\n")
            if state == "thought":
                self.write("\n<think>\n")
            elif state == "detail":
                self.write(f"\n<details><summary><b>{topic}:</b></summary>\n\n")
        elif self._topic != topic:
            self.write("\n</details>\n")
            self.write(f"\n<details><summary><b>{topic}:</b></summary>\n\n")
        self._state = state
        self._topic = topic
