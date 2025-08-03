from typing import NamedTuple, List, Optional, Callable, Awaitable
from pydantic import BaseModel

from micro_graph.output_writer import OutputWriter


Agent = Callable[[OutputWriter, str, str], Awaitable[str | None]]


class ToolInfo(NamedTuple):
    name: str
    description: str
    arguments: str


class ImageURL(BaseModel):
    url: str


class ChatContent(BaseModel):
    type: str
    text: Optional[str] = ""
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: str
    content: str | List[ChatContent]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        else:
            for elem in self.content:
                if elem.type == "text" and elem.text is not None:
                    return elem.text
        return ""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = -1
    stream: Optional[bool] = False


class EmbeddingRequest(BaseModel):
    model: str
    input: str | List[str]
