"""Tests for fs (full-session) agents including compactor and summarizer.

This module contains test functions for FsCompactor and FsSummarizer operations.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from reme import ReMe
from reme.agent.fs.fs_compactor import FsCompactor
from reme.agent.fs.fs_summarizer import FsSummarizer
from reme.core.enumeration import Role
from reme.core.schema import Message
from reme.tool.fs import ReadTool, WriteTool, EditTool


def create_test_messages(num_messages: int = 10) -> list[Message]:
    """Create a list of test messages for testing.

    Args:
        num_messages: Number of messages to create

    Returns:
        List of Message objects alternating between user and assistant
    """
    messages = []
    for i in range(num_messages):
        if i % 2 == 0:
            # User messages
            messages.append(
                Message(
                    role=Role.USER,
                    content=f"User message {i}: Can you help me with task {i}?",
                ),
            )
        else:
            # Assistant messages
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=f"Assistant message {i}: Sure, I'd be happy to help you with task {i - 1}. "
                    f"Let me explain the solution in detail. " * 10,  # Make it longer
                ),
            )
    return messages


def create_long_conversation() -> list[Message]:
    """Create a long conversation that exceeds token thresholds."""
    messages = [
        Message(
            role=Role.USER,
            content="I need help building a complete web application with authentication, database, and API endpoints.",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""I'll help you build a complete web application. Here's what we'll do:

1. Set up the project structure
2. Implement authentication system
3. Design and create database schema
4. Build API endpoints
5. Add frontend components
6. Test and deploy

Let me start with the project structure...""",
        ),
    ]

    # Initial user request

    # Assistant response with detailed steps

    # Continue with multiple turns
    for i in range(15):
        messages.append(
            Message(
                role=Role.USER,
                content=f"What about step {i + 1}? Can you provide more details?",
            ),
        )
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=f"""For step {i + 1}, here's a detailed explanation:

First, we need to consider the architecture. """
                + "This is important context. " * 50
                + """

Then we implement the following:
- Component A
- Component B
- Component C

Let me show you the code for this part..."""
                + "\n\ncode_example = 'example'" * 20,
            ),
        )

    return messages


async def test_compactor_basic(reme: ReMe):
    """Test basic FsCompactor functionality without triggering compaction.

    Tests that the compactor correctly skips compaction when token count
    is below the threshold.
    """
    print("\n" + "=" * 60)
    print("Testing FsCompactor - Basic (Below Threshold)")
    print("=" * 60)

    # Create a small conversation that won't trigger compaction
    messages = create_test_messages(num_messages=6)

    # Create compactor with high threshold so it won't trigger
    compactor = FsCompactor(
        context_window_tokens=128000,
        reserve_tokens=10000,
        keep_recent_tokens=5000,
    )

    print(f"Number of messages: {len(messages)}")
    output = await compactor.call(messages=messages, service_context=reme.service_context)
    print(f"test_compactor_basic output: {output}")


async def test_compactor_with_compaction(reme: ReMe):
    """Test FsCompactor with a long conversation that triggers compaction.

    Tests that the compactor correctly summarizes old messages when
    the conversation exceeds the token threshold.
    """
    print("\n" + "=" * 60)
    print("Testing FsCompactor - With Compaction")
    print("=" * 60)

    # Create a long conversation
    messages = create_long_conversation()

    # Create compactor with low threshold to trigger compaction
    compactor = FsCompactor(
        context_window_tokens=10000,  # Low threshold
        reserve_tokens=2000,
        keep_recent_tokens=2000,
    )

    print(f"Number of messages: {len(messages)}")
    output = await compactor.call(messages=messages, service_context=reme.service_context)
    print(f"test_compactor_with_compaction output: {output}")


async def test_compactor_split_turn(reme: ReMe):
    """Test FsCompactor with a split turn scenario.

    Tests the scenario where the cut point falls in the middle of a turn,
    requiring special handling to maintain context.
    """
    print("\n" + "=" * 60)
    print("Testing FsCompactor - Split Turn Detection")
    print("=" * 60)

    messages = []

    # Add some initial conversation
    for i in range(5):
        messages.append(Message(role=Role.USER, content=f"Question {i}"))
        messages.append(Message(role=Role.ASSISTANT, content=f"Answer {i}. " * 30))

    # Add a very long assistant response that will be split
    messages.append(Message(role=Role.USER, content="Please explain this in great detail."))
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="This is the first part of a very long response. " * 100,
        ),
    )
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="This is the continuation of the response. " * 100,
        ),
    )
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="And here's the final part with the conclusion. " * 50,
        ),
    )

    compactor = FsCompactor(
        context_window_tokens=8000,
        reserve_tokens=1000,
        keep_recent_tokens=2000,
    )

    print(f"Number of messages: {len(messages)}")
    output = await compactor.call(messages=messages, service_context=reme.service_context)
    print(f"test_compactor_split_turn output: {output}")


async def test_summarizer_basic(reme: ReMe):
    """Test basic FsSummarizer functionality.

    Tests that the summarizer correctly skips when below threshold
    and executes when above threshold.
    """
    print("\n" + "=" * 60)
    print("Testing FsSummarizer - Basic")
    print("=" * 60)

    # Create a temporary directory for memory storage
    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = os.path.join(temp_dir, "memories")
        Path(memory_dir).mkdir(parents=True, exist_ok=True)

        # Create a small conversation (below threshold)
        messages = create_test_messages(num_messages=4)

        summarizer = FsSummarizer(
            tools=[ReadTool(), WriteTool(), EditTool()],
            memory_dir=memory_dir,
            context_window_tokens=128000,
            reserve_tokens=32000,
            soft_threshold_tokens=4000,
        )

        print(f"Memory directory: {memory_dir}")
        print(f"Number of messages: {len(messages)}")
        output = await summarizer.call(messages=messages, service_context=reme.service_context)
        print(f"test_summarizer_basic output: {output}")


async def test_summarizer_with_execution(reme: ReMe):
    """Test FsSummarizer with execution triggered.

    Tests that the summarizer executes when token count is within
    the soft threshold range before compaction.
    """
    print("\n" + "=" * 60)
    print("Testing FsSummarizer - With Execution")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        memory_dir = os.path.join(temp_dir, "memories")
        Path(memory_dir).mkdir(parents=True, exist_ok=True)

        # Create messages that will trigger summarizer but not compactor
        messages = create_test_messages(num_messages=10)

        # Set low thresholds to trigger execution
        summarizer = FsSummarizer(
            tools=[ReadTool(), WriteTool(), EditTool()],
            memory_dir=memory_dir,
            context_window_tokens=5000,
            reserve_tokens=1000,
            soft_threshold_tokens=500,
        )

        print(f"Memory directory: {memory_dir}")
        print(f"Number of messages: {len(messages)}")
        output = await summarizer.call(messages=messages, service_context=reme.service_context)
        print(f"test_summarizer_with_execution output: {output}")


def test_compactor_serialization():
    """Test message serialization in FsCompactor.

    Tests that messages are correctly serialized to text format
    for summarization.
    """
    print("\n" + "=" * 60)
    print("Testing FsCompactor - Message Serialization")
    print("=" * 60)

    messages = [
        Message(role=Role.USER, content="Hello, how are you?", name="Alice"),
        Message(role=Role.ASSISTANT, content="I'm doing great, thanks!"),
        Message(role=Role.USER, content="Can you help me?"),
    ]

    # Access static method for testing serialization
    serialized = FsCompactor._serialize_conversation(messages)  # pylint: disable=protected-access

    print("Serialized conversation:")
    print(serialized)
    print("\n✓ Serialization completed")

    # Check that it contains expected markers
    assert "[Alice]" in serialized
    assert "[assistant]" in serialized
    assert "Hello, how are you?" in serialized
    print("✓ Serialization format is correct")


async def main():
    """Run all tests."""
    # Run basic tests first
    reme = ReMe()
    await reme.start()
    test_compactor_serialization()
    await test_compactor_basic(reme)
    await test_summarizer_basic(reme)

    # Run tests that require LLM calls (commented out by default)
    # Uncomment these if you want to test with actual LLM calls
    # await test_compactor_with_compaction(reme)
    # await test_compactor_split_turn(reme)
    # await test_summarizer_with_execution(reme)

    print("\n" + "=" * 60)
    print("All basic tests completed!")
    print("=" * 60)
    print("\nNote: Tests requiring LLM calls are commented out.")
    print("Uncomment them in the main() function to run with actual LLM.")
    await reme.close()


if __name__ == "__main__":
    asyncio.run(main())
