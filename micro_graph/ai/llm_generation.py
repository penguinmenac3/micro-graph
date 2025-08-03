from micro_graph.micro_graph import Node, OutputWriter, template_formatting
from micro_graph.ai.llm import LLM
from micro_graph.ai.types import ChatMessage
from typing import List


class LLMGenerateNode(Node):
    def __init__(self, llm: LLM, model: str, prompt_template: str, field: str = "response", shared: bool = False, output: str = "", max_tokens: int = -1):
        super().__init__()
        self._llm = llm
        self._prompt_template = prompt_template
        self._model = model
        self._field = field
        self._shared = shared
        self._output = output
        self._max_tokens = max_tokens
        self.system_prompt = "You are an expert ant answering user requests. " \
            "Your responses are always exactly what the users asks, without any additional information. " \
            "You are not allowed to add any additional information, only the exact answer to the question.\n\n" \
            "For example if the user asks what is 2+2, you answer '4', nothing else, not even 'The answer is 4'."

    async def run(self, output: OutputWriter, shared: dict, **kwargs):
        prompt = template_formatting(self._prompt_template, shared, **kwargs)
        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=prompt)
        ]
        response = self._llm.chat(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            stream=self._output != ""  # only stream if we want to output the response
        )
        answer = ""
        if isinstance(response, str):
            answer = response
        else:
            for chunk in response:
                answer += chunk
                if self._output == "default":
                    output.default(chunk, end="")
                elif self._output == "thought":
                    output.thought(chunk, end="")
                elif self._output != "":
                    output.detail(self._output, chunk, end="")
            output.write("\n") # Add a new line after we finished streaming
        if self._shared:
            shared[self._field] = answer
            return kwargs
        else:
            return {self._field: answer}


class LLMDecisionNode(Node):
    def __init__(self, llm: LLM, model: str, prompt_template: str, field: str = "", shared: bool = False, max_tokens: int = -1):
        super().__init__()
        self._llm = llm
        self._prompt_template = prompt_template
        self._model = model
        self._field = field
        self._shared = shared
        self._max_tokens = max_tokens
        self.system_prompt = "You are an expert at deciding what to do next. " \
            "Your outputs only consist of a single word based on the options the user provides you. " \
            "The user should always provide some context for the decision you make and you must decide based on the context.\n\n" \
            "For example if the user says, chose between `banana_split` and `chocolate` " \
            "and says in the context he likes bananas, you choose `banana_split`.\n\n" \
            "Note: that the spelling must be exactly the same as the options provided by the user including the capitalization." \

    async def run(self, output: OutputWriter, shared: dict, **kwargs):
        prompt = template_formatting(self._prompt_template, shared, **kwargs)
        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=prompt)
        ]
        response = self._llm.chat(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            stream=False
        )
        if not isinstance(response, str):
            raise RuntimeError("Streaming is disabled, so we should always get a stream."
                               "If this error happens, there is an error in the node implementation.")
        output.thought(f"Decision made: {response}")
        if self._field == "":
            return response
        if self._shared:
            shared[self._field] = response
            return None
        else:
            return {self._field: response}
