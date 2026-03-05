from pydantic import BaseModel, Field

_DEFAULT_MAX_BLOCK_TEXT_PREVIEW_LENGTH = 100
_DEFAULT_MAX_FORMATTER_TEXT_LENGTH = 2000

# Unique marker for truncated text
TRUNCATION_MARKER_START = "<<<TRUNCATED>>>"
TRUNCATION_MARKER_END = "<<<END_TRUNCATED>>>"


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length, keeping head and tail portions."""
    text = str(text) if text else ""
    if not text or len(text) <= max_length:
        return text
    half_length = max_length // 2
    truncated_chars = len(text) - max_length
    return (
        f"{text[:half_length]}\n\n{TRUNCATION_MARKER_START} "
        f"({truncated_chars} characters omitted) "
        f"{TRUNCATION_MARKER_END}\n\n{text[-half_length:]}"
    )


class AsBlockStat(BaseModel):
    block_type: str = Field(default=...)
    text: str = Field(default="", description="Text content of the block")
    token_count: int = Field(default=0, description="Token count of the block, including base64 data")

    # For tool_use and tool_result blocks
    tool_name: str = Field(default="", description="Tool name for tool_use/tool_result blocks")
    tool_input: str = Field(default="", description="Tool input arguments for tool_use blocks")
    tool_output: str = Field(default="", description="Tool output for tool_result blocks")

    # For media blocks
    media_url: str = Field(default="", description="URL for image/audio/video blocks")

    @property
    def preview(self) -> str:
        return self.format(_DEFAULT_MAX_BLOCK_TEXT_PREVIEW_LENGTH)

    def format(self, max_length: int = _DEFAULT_MAX_FORMATTER_TEXT_LENGTH, include_thinking: bool = True) -> str:
        """Format block content to string representation."""
        if self.block_type == "text":
            return _truncate_text(self.text, max_length) if self.text else ""
        if self.block_type == "thinking":
            if include_thinking and self.text:
                return f"<thinking>\n{_truncate_text(self.text, max_length)}\n</thinking>"
            return ""
        if self.block_type in ("image", "audio", "video"):
            return f"[{self.block_type}] {self.media_url}" if self.media_url else f"[{self.block_type}]"
        if self.block_type == "tool_use":
            return f" - tool_call={self.tool_name} params={_truncate_text(self.tool_input, max_length)}"
        if self.block_type == "tool_result":
            output = _truncate_text(self.tool_output, max_length)
            return f" - tool_result={self.tool_name} output={output}" if output else ""
        return ""


class AsMsgStat(BaseModel):
    name: str = Field(default=...)
    role: str = Field(default="")
    content: list[AsBlockStat] = Field(default_factory=list)
    timestamp: str = Field(default="")
    metadata: dict = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(block.token_count for block in self.content)

    @property
    def preview(self) -> str:
        return self.format(_DEFAULT_MAX_BLOCK_TEXT_PREVIEW_LENGTH)

    def format(self, max_length: int = _DEFAULT_MAX_FORMATTER_TEXT_LENGTH, include_thinking: bool = True) -> str:
        """Format message to string representation."""
        time_str = f"[{self.timestamp}] " if self.timestamp else ""
        header = f"{time_str}{self.name or self.role}:"
        blocks = [block.format(max_length, include_thinking) for block in self.content]
        return "\n".join([header] + [b for b in blocks if b])
