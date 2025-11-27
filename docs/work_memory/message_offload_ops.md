---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Message Offload Ops

## MessageOffloadOp

### Purpose

Manages context window limits by intelligently offloading message content through compaction and compression strategies to reduce token usage while preserving important information.

### Functionality

- Supports three working summary modes: `compact`, `compress`, and `auto`
- **Compact mode**: Stores full content of large tool messages in external files, keeping only previews in context
- **Compress mode**: Uses LLM to generate concise summaries of older message groups
- **Auto mode** (recommended): Applies compaction first, then compression if compaction ratio exceeds `compact_ratio_threshold`
- Automatically writes offloaded content to files via `BatchWriteFileOp`
- Preserves recent messages and system messages to maintain conversation coherence
- Configurable token thresholds for both compaction and compression operations

### Parameters

- `messages` (array, **required**):
  - List of conversation messages to process for working memory summarization
  - Messages are analyzed for token count and processed according to management mode

- `working_summary_mode` (string, optional, default: `"auto"`):
  - Working summary strategy to use
  - `"compact"`: Only applies compaction to large tool messages
  - `"compress"`: Only applies LLM-based compression
  - `"auto"`: Applies compaction first then compression if compaction ratio exceeds threshold
  - Allowed values: `["compact", "compress", "auto"]`

- `compact_ratio_threshold` (number, optional, default: `0.75`):
  - Only used in `"auto"` mode
  - Threshold for compaction ratio (tokens after compaction divided by original tokens)
  - When the ratio is greater than this value, an additional LLM-based compression pass is triggered
  - Example: If ratio is 0.76 (76%) and threshold is 0.75, compression will be applied

- `max_total_tokens` (integer, optional, default: `20000`):
  - Maximum token count threshold for triggering compression/compaction
  - For compaction mode: this is the total token count threshold
  - For compression mode: excludes `keep_recent_count` messages and system messages
  - Operation is skipped if token count is below this threshold

- `max_tool_message_tokens` (integer, optional, default: `2000`):
  - Maximum token count per individual tool message before compaction is applied
  - Tool messages exceeding this threshold will have full content stored in external files
  - Only a preview is kept in context with a reference to the stored file

- `group_token_threshold` (integer, optional):
  - Maximum token count per compression group when using LLM-based compression
  - If `None` or `0`, all messages are compressed in a single group
  - Messages exceeding this threshold individually will form their own group
  - Only used in `"compress"` or `"auto"` mode

- `keep_recent_count` (integer, optional, default: `1` for compaction, `2` for compression):
  - Number of recent messages to preserve without compression or compaction
  - These messages remain unchanged to maintain conversation context
  - Does not include system messages (which are always preserved)

- `store_dir` (string, optional):
  - Directory path for storing summarized message content
  - Full tool message content and compressed message groups are saved as files in this directory
  - Required for compaction and compression operations

- `chat_id` (string, optional):
  - Unique identifier for the chat session
  - Used for file naming when storing compressed message groups
  - If not provided, a UUID will be generated automatically

### Usage Pattern
For complete working examples of how to use MessageOffloadOp in practice, please refer to:
[test_message_offload_op.py](../../test_op/test_message_offload_op.py)

This test file demonstrates:
- **Compact mode**: How to configure and use compaction-only strategy
- **Compress mode**: How to apply LLM-based compression strategy
- **Auto mode**: How to combine compaction and compression intelligently
- Proper parameter settings for different scenarios
- Integration with `BatchWriteFileOp` for file writing
- Real-world message sequences with various token sizes

