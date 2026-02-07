"""Tests for ReMeFs summary interface.

This module tests the summary() method of ReMeFs class which provides
a high-level interface for storing user's personal information into memory files.

The summary function should enable the LLM to:
1. Extract personal information from user messages (name, preferences, requirements)
2. Call file system tools (WriteTool, EditTool) to store this information
3. Maintain personalized memory for future conversations
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


def print_result(result: dict, title: str = "RESULT"):
    """Print the result of summary() call.

    Args:
        result: Result dictionary from summary()
        title: Title for the result section
    """
    print(f"\n{'=' * 80}")
    print(f"{title}:")
    print(f"  success: {result.get('success')}")
    print(f"  skipped: {result.get('skipped', False)}")

    tools_used = result.get("tools", [])
    print(f"  tools_called: {len(tools_used)}")

    if tools_used:
        print("\n  Tool Usage Details:")
        for i, tool in enumerate(tools_used):
            print(f"    [{i}] Tool: {tool.name} Arguments: {tool.tool_call.arguments}")

    answer = result.get("answer", "")
    if answer:
        answer_preview = answer[:300] + "..." if len(answer) > 300 else answer
        print(f"\n  answer: {answer_preview}")

    print(f"{'=' * 80}")


def create_personal_info_introduction() -> list[Message]:
    """Create a conversation where user introduces personal information."""
    return [
        Message(
            role=Role.USER,
            content="Hi! My name is Alice, and I'm a software engineer.",
        ),
        Message(
            role=Role.ASSISTANT,
            content="Nice to meet you, Alice! How can I help you today?",
        ),
        Message(
            role=Role.USER,
            content=(
                "I love Python programming and working on AI projects. "
                "I also enjoy reading sci-fi novels in my free time."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="That's great! Python and AI are exciting fields. What kind of AI projects are you interested in?",
        ),
    ]


def create_detailed_profile_conversation() -> list[Message]:
    """Create a comprehensive conversation with multiple personal details."""
    return [
        Message(
            role=Role.USER,
            content=(
                "Let me tell you about myself. My name is Charlie Chen, " "I'm a data scientist based in San Francisco."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="Hello Charlie! It's nice to meet you. Tell me more about your work.",
        ),
        Message(
            role=Role.USER,
            content=(
                "I specialize in machine learning and natural language processing. "
                "My favorite tools are PyTorch and Hugging Face transformers."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="Those are excellent tools for NLP work. What kind of projects do you work on?",
        ),
        Message(
            role=Role.USER,
            content="I work on chatbots and sentiment analysis. Outside of work, I love hiking and photography.",
        ),
        Message(
            role=Role.ASSISTANT,
            content="That's a great combination of technical and creative interests!",
        ),
        Message(
            role=Role.USER,
            content=(
                "Oh, and one more thing - please be more assertive when you think "
                "I'm making a mistake. I want you to challenge my ideas."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="Absolutely, I'll make sure to provide critical feedback when needed.",
        ),
    ]


async def test_summary_personal_info_storage():
    """Test summary() stores user's personal information to memory.

    User introduces name and basic info.
    Expects: LLM should call WriteTool to store this information
    """
    print("\n" + "=" * 80)
    print("TEST 1: Summary - Personal Information Storage")
    print("=" * 80)

    reme_fs = ReMeFs(enable_logo=False)
    await reme_fs.start()

    messages = create_personal_info_introduction()
    print_messages(messages, "INPUT MESSAGES", max_content_len=200)

    print("\nParameters:")
    print("  version: default")
    print("  Expected: LLM should call WriteTool to save user's name and interests")

    result = await reme_fs.summary(
        messages=messages,
        date="2023-09-01",
    )

    print_result(result, "SUMMARY RESULT")
    await reme_fs.close()


async def test_summary_detailed_profile():
    """Test summary() with comprehensive user profile information.

    User provides detailed personal, professional, and preference information.
    Expects: LLM should organize and store all relevant information
    """
    print("\n" + "=" * 80)
    print("TEST 2: Summary - Detailed User Profile Storage")
    print("=" * 80)

    reme_fs = ReMeFs(enable_logo=False, vector_store=None)
    await reme_fs.start()

    messages = create_detailed_profile_conversation()
    print_messages(messages, "INPUT MESSAGES", max_content_len=150)

    print("\nParameters:")
    print("  version: default")
    print("  Expected: LLM should extract and store:")
    print("    - Name: Charlie Chen")
    print("    - Profession: Data Scientist")
    print("    - Location: San Francisco")
    print("    - Skills: ML, NLP, PyTorch, Hugging Face")
    print("    - Hobbies: Hiking, Photography")
    print("    - Assistant behavior: Be assertive and critical")

    result = await reme_fs.summary(
        messages=messages,
        date="2023-10-01",
    )

    print_result(result, "SUMMARY RESULT")
    await reme_fs.close()


async def main():
    """Run core summary interface tests for personal memory storage."""
    print("\n" + "=" * 80)
    print("ReMeFs Summary Interface - Personal Memory Storage Tests")
    print("=" * 80)
    print("\nThis test suite validates that the summary() function:")
    print("  1. Extracts personal information from user messages")
    print("  2. Calls file system tools (WriteTool/EditTool) to store the info")
    print("  3. Organizes information for future retrieval")
    print("\nTest Scenarios:")
    print("  1. Personal info - name, profession, interests")
    print("  2. Detailed profile - comprehensive user information")
    print("=" * 80)

    # Test 1: Basic personal information storage
    await test_summary_personal_info_storage()

    # Test 2: Comprehensive user profile
    await test_summary_detailed_profile()

    print("\n" + "=" * 80)
    print("All summary tests completed!")
    print("=" * 80)
    print("\nNote: These tests require LLM calls to:")
    print("  - Analyze user messages for personal information")
    print("  - Decide what information to store")
    print("  - Call appropriate tools (WriteTool/EditTool) to save to memory")
    print("\nMake sure your API keys are properly configured before running.")
    print("The LLM will autonomously decide what to store based on the conversation.")


if __name__ == "__main__":
    asyncio.run(main())
