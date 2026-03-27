from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class KasaNovaState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_docs: list[dict[str, Any]]
