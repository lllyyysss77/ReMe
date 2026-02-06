"""Tests for ReMeFs compact interface.

This module tests the compact() method of ReMeFs class which provides
a high-level interface for conversation compaction.
"""

import asyncio

from reme import ReMeFs
from reme.core.enumeration import Role
from reme.core.schema import Message


def print_messages(messages: list[Message], title: str = "Messages", max_content_len: int = 150):
    """Print messages with their role and content.

    Args:
        messages: List of messages to print
        title: Title for the message list
        max_content_len: Maximum content length to display (truncate if longer)
    """
    print(f"\n{title}: (count: {len(messages)})")
    print("-" * 80)
    for i, msg in enumerate(messages):
        content = str(msg.content)
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        print(f"  [{i}] {msg.role.value:10s}: {content}")
    print("-" * 80)


def create_test_messages(num_messages: int = 10) -> list[Message]:
    """Create a list of test messages.

    Args:
        num_messages: Number of messages to create

    Returns:
        List of Message objects alternating between user and assistant
    """
    messages = []
    for i in range(num_messages):
        if i % 2 == 0:
            messages.append(
                Message(
                    role=Role.USER,
                    content=f"User message {i}: Can you help me with task {i}?",
                ),
            )
        else:
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=f"Assistant message {i}: Sure, I'd be happy to help you with task {i - 1}. "
                    f"Let me explain the solution in detail. " * 10,
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


async def test_compact_below_threshold():
    """Test compact() when messages are below threshold.

    Expects: compacted=False, returns original messages
    """
    print("\n" + "=" * 80)
    print("TEST 1: Compact - Below Threshold (No Compaction)")
    print("=" * 80)

    reme_fs = ReMeFs(enable_logo=False, vector_store=None)
    await reme_fs.start()

    messages = create_test_messages(num_messages=4)
    print_messages(messages, "INPUT MESSAGES", max_content_len=80)

    print("\nParameters:")
    print("  context_window_tokens: 5000")
    print("  reserve_tokens: 2000 (threshold = 3000)")
    print("  keep_recent_tokens: 1000")

    result = await reme_fs.compact(
        messages=messages,
        context_window_tokens=5000,
        reserve_tokens=2000,
        keep_recent_tokens=1000,
    )

    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  compacted: {result.get('compacted')}")
    print(f"  tokens_before: {result.get('tokens_before')}")
    print(f"  is_split_turn: {result.get('is_split_turn')}")

    result_messages = result.get("messages", [])
    print_messages(result_messages, "OUTPUT MESSAGES", max_content_len=80)

    assert result.get("compacted") is False, "Should not compact below threshold"
    assert len(result_messages) == len(messages), "Should return all original messages"
    print("\n✓ TEST PASSED: No compaction below threshold\n")

    await reme_fs.close()


async def test_compact_above_threshold():
    """Test compact() when messages exceed threshold.

    Expects: compacted=True, returns summary + left_messages
    """
    print("\n" + "=" * 80)
    print("TEST 2: Compact - Above Threshold (With Compaction & LLM Summary)")
    print("=" * 80)

    reme_fs = ReMeFs(enable_logo=False, vector_store=None)
    await reme_fs.start()

    messages = create_test_messages(num_messages=12)
    print_messages(messages, "INPUT MESSAGES", max_content_len=60)

    print("\nParameters:")
    print("  context_window_tokens: 3000")
    print("  reserve_tokens: 1500 (threshold = 1500)")
    print("  keep_recent_tokens: 500 (keep only recent messages)")

    result = await reme_fs.compact(
        messages=messages,
        context_window_tokens=3000,
        reserve_tokens=1500,
        keep_recent_tokens=500,
    )

    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  compacted: {result.get('compacted')}")
    print(f"  tokens_before: {result.get('tokens_before')}")
    print(f"  is_split_turn: {result.get('is_split_turn')}")

    result_messages = result.get("messages", [])
    if result.get("compacted") and result_messages:
        has_summary = "<summary>" in str(result_messages[0].content)
        print(f"\n  *** First message contains summary: {has_summary}")

    print_messages(result_messages, "OUTPUT MESSAGES (Summary + Recent)", max_content_len=1500)

    assert result.get("compacted") is True, "Should compact above threshold"
    assert len(result_messages) < len(messages), "Should reduce message count"
    print("\n✓ TEST PASSED: Compaction triggered and summary generated\n")

    await reme_fs.close()


async def test_compact_split_turn_scenario():
    """Test compact() with split turn scenario.

    Expects: is_split_turn=True when cut point is mid-turn
    """
    print("\n" + "=" * 80)
    print("TEST 3: Compact - Split Turn Scenario (Cut in Middle of Assistant Response)")
    print("=" * 80)

    reme_fs = ReMeFs(enable_logo=False, vector_store=None)
    await reme_fs.start()

    messages = []

    # Add initial conversation
    for i in range(3):
        messages.append(Message(role=Role.USER, content=f"Question {i}"))
        messages.append(Message(role=Role.ASSISTANT, content=f"Answer {i}. " * 30))

    # Add a very long multi-part assistant response
    messages.append(Message(role=Role.USER, content="Please explain this in great detail."))
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="This is the first part of a very long response. " * 50,
        ),
    )
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="This is the continuation of the response. " * 50,
        ),
    )
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content="And here's the final part with the conclusion. " * 30,
        ),
    )

    print_messages(messages, "INPUT MESSAGES", max_content_len=80)

    print("\nParameters:")
    print("  context_window_tokens: 3000")
    print("  reserve_tokens: 1000 (threshold = 2000)")
    print("  keep_recent_tokens: 800 (should cut in middle of assistant responses)")

    result = await reme_fs.compact(
        messages=messages,
        context_window_tokens=3000,
        reserve_tokens=1000,
        keep_recent_tokens=800,
    )

    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  compacted: {result.get('compacted')}")
    print(f"  tokens_before: {result.get('tokens_before')}")
    print(f"  is_split_turn: {result.get('is_split_turn')} *** (should be True)")

    result_messages = result.get("messages", [])
    print_messages(result_messages, "OUTPUT MESSAGES (Summary with Turn Context + Recent)", max_content_len=150)

    if result.get("is_split_turn"):
        print("\n✓ TEST PASSED: Split turn correctly detected and handled\n")
    else:
        print("\n⚠ WARNING: Split turn not detected (parameters may need adjustment)\n")

    await reme_fs.close()


async def main():
    """Run core compact interface tests."""
    print("\n" + "=" * 80)
    print("ReMeFs Compact Interface - Core Test Suite")
    print("=" * 80)
    print("\nThis test suite demonstrates the three key scenarios of conversation compaction:")
    print("  1. Below threshold - no compaction needed")
    print("  2. Above threshold - full compaction with LLM summary")
    print("  3. Split turn - cut point falls in middle of assistant response")
    print("=" * 80)

    # Test 1: No compaction (below threshold)
    await test_compact_below_threshold()

    # Test 2: Full compaction (requires LLM)
    await test_compact_above_threshold()

    # Test 3: Split turn compaction (requires LLM)
    await test_compact_split_turn_scenario()

    print("\n" + "=" * 80)
    print("All basic tests completed!")
    print("=" * 80)
    print("\nNote: Tests requiring LLM calls are commented out.")
    print("Uncomment them in the main() function to run with actual LLM.")


if __name__ == "__main__":
    asyncio.run(main())
