from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentResponse:
    message_content: str
    structured_response: Optional[str]
    human_messages: int
    ai_messages: int
    tool_messages: int
