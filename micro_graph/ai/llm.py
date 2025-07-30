from typing import Generator, List
import requests
import openai
from openai import AzureOpenAI as _AzureAPI
from openai import OpenAI as _OpenAIAPI

from micro_graph.ai.types import ChatMessage


class LLMAPI(object):
    def embeddings(self, model: str, input: str | List[str]) -> List[List[float]]:
        raise NotImplementedError("This method should be implemented by subclasses.")
        
    def chat(self, model: str, messages: List[ChatMessage], max_tokens: int = -1, stream: bool = False) -> str | Generator[str, None, None]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_models(self) -> list[str]:
        raise NotImplementedError("This method should be implemented by subclasses.")


class LLM(LLMAPI):
    def __init__(self, api_endpoint: str, api_key: str, provider: str = "OpenAI", model: str = "AUTODETECT"):
        self._llms = {}
        if provider == "ollama":
            if model == "AUTODETECT":
                modelListEndpoint = api_endpoint + "/api/tags"
                response = requests.get(modelListEndpoint)
                response.raise_for_status()
                data = response.json()
                models: list[str] = [model["name"] for model in data["models"]]
            else:
                models = [model]
            for model in models:
                self._llms[model] = _OpenAIAPI(base_url=api_endpoint + "/v1", api_key=api_key)
        else:
            if model == "AUTODETECT":
                raise ValueError("Model must be specified when not using ollama provider.")
            if provider == "AzureOpenAI":
                self._llms[model] = _AzureAPI(api_version="2024-10-21", base_url=api_endpoint, api_key=api_key)
            else:
                self._llms[model] = _OpenAIAPI(base_url=api_endpoint, api_key=api_key)

    def embeddings(self, model: str, input: str | List[str]) -> List[List[float]]:
        response = self._llms[model].embeddings.create(input=input, model=model)
        return [d.embedding for d in response.data]

    def chat(self, model: str, messages: List[ChatMessage], max_tokens: int = -1, stream: bool = False) -> str | Generator[str, None, None]:
        try:            
            response = self._llms[model].chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, stream=stream)
        except openai.NotFoundError as e:
            raise RuntimeError(str(e))
        if stream:
            return LLM._stream_wrapper(response)
        else:
            if response.choices[0].finish_reason == "error":
                raise RuntimeError(response.choices[0].message.content)
            return response.choices[0].message.content or ""

    def get_models(self) -> list[str]:
        return list(self._llms.keys())

    @staticmethod
    def _stream_wrapper(stream):
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
