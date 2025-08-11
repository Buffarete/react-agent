"""State definitions for the agent using modern LangGraph patterns."""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep


class State(TypedDict):
    """Primary agent state with message accumulation and managed flags."""

    messages: Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep
