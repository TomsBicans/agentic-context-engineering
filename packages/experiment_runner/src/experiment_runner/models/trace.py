from typing import Any, Optional
from pydantic import BaseModel


class TraceBlock(BaseModel):
    """One content block inside a trace message (text, tool_use, or tool_result)."""

    type: str

    # text block
    text: Optional[str] = None

    # tool_use block
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[str] = None  # JSON-encoded string as produced by most agent runtimes

    # tool_result block
    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    output: Optional[str] = None
    is_error: Optional[bool] = None


class TraceUsage(BaseModel):
    """Per-message token counts reported by the underlying LLM API."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None


class TraceMessage(BaseModel):
    role: str  # "user" | "assistant" | "tool"
    blocks: list[TraceBlock]
    usage: Optional[TraceUsage] = None


class SessionTrace(BaseModel):
    """Full conversation trace for one experiment run.

    Modelled after ClawCode's JSONL session format; fields are optional so that
    adapters for other systems can populate only what they have access to.
    """

    session_id: Optional[str] = None
    created_at_ms: Optional[int] = None
    updated_at_ms: Optional[int] = None
    model: Optional[str] = None
    workspace_root: Optional[str] = None
    messages: list[TraceMessage] = []
    # Unstructured fallback for system-specific metadata that does not fit above.
    extra: Optional[Any] = None
