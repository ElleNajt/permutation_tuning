from dataclasses import dataclass, asdict
import orjson

from typing import TypedDict
from collections import UserList

from src.utils import validate_path

class ChatMessage(TypedDict):
    role: str
    content: str

ChatRequest = UserList[ChatMessage]


@dataclass
class Example:
    question: str
    cot: str
    answer: str
    messages: ChatRequest | None = None

    @classmethod
    def from_dict(cls, x: dict) -> 'Example':
        return cls(**x)

    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_messages(self) -> ChatRequest:
        return [
            {'role': 'user', 'content': self.question + '.\n Please reason step by step, and put your final answer within \boxed{}.'},
            {'role': 'assistant', 'content': f"<think>{self.cot}</think>\n \boxed{{{self.answer}}}"}
        ]
    
    def to_chatml(self) -> dict:
        return {
            "messages": self.to_messages()
        }


def save_dataset(dataset: list[Example], path: str):
    validate_path(path)
    with open(path, 'wb') as f:
        f.write(orjson.dumps([x.to_dict() for x in dataset], option = orjson.OPT_INDENT_2))
    print(f"Saved dataset to {path}")


def load_dataset(path: str) -> list[Example]:
    with open(path, 'rb') as f:
        return [Example.from_dict(x) for x in orjson.loads(f.read())]
    