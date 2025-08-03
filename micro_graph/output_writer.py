class OutputWriter:
    def __init__(self):
        self._state = "default"
        self._topic = ""

    def thought(self, text: str, end="\n") -> None:
        self._change_state("thought")
        self.write(text + end)

    def default(self, text: str, end="\n") -> None:
        self._change_state("default")
        self.write(text + end)

    def detail(self, topic: str, text: str, end="\n") -> None:
        self._change_state("detail", topic)
        self.write(text + end)

    def write(self, text: str) -> None:
        """
        Do not use this function, but overwrite it, if you want to output to http or a file.
        """
        print(text, end="")

    def _change_state(self, state: str, topic: str = "") -> None:
        if state != self._state:
            if self._state == "thought":
                self.write("\n</think>\n")
            elif self._state == "detail":
                self.write("\n</details>\n")
            if state == "thought":
                self.write("\n<think>\n")
            elif state == "detail":
                self.write(f"\n<details><summary><b>{topic}:</b></summary>\n\n")
            self._state = state
            self._topic = topic
        elif self._topic != topic:
            self.write("\n</details>\n")
            self.write(f"\n<details><summary><b>{topic}:</b></summary>\n\n")
