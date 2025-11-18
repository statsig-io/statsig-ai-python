"""
Example: Using wrap_openai with StatsigAI

This example demonstrates four different usage patterns:
1. Synchronous non-streaming
2. Synchronous streaming
3. Asynchronous non-streaming
4. Asynchronous streaming

Prerequisites:
- Set STATSIG_API_KEY environment variable
- Set OPENAI_API_KEY environment variable
"""

import asyncio
import os
from openai import OpenAI, AsyncOpenAI
from statsig_ai import StatsigAI, StatsigCreateConfig, wrap_openai


def sync_non_streaming_example():
    """Example 1: Synchronous non-streaming chat completion"""
    print("\n" + "=" * 60)
    print("Example 1: Sync Non-Streaming")
    print("=" * 60)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    wrapped_client = wrap_openai(client)

    response = wrapped_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say hello!"},
        ],
    )

    print(f"Response: {response.choices[0].message.content}\n")


def sync_streaming_example():
    """Example 2: Synchronous streaming chat completion"""
    print("\n" + "=" * 60)
    print("Example 2: Sync Streaming")
    print("=" * 60)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    wrapped_client = wrap_openai(client)

    response = wrapped_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Count from 1 to 5."},
        ],
        stream=True,
    )

    print("Response: ", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


async def async_non_streaming_example():
    """Example 3: Asynchronous non-streaming chat completion"""
    print("\n" + "=" * 60)
    print("Example 3: Async Non-Streaming")
    print("=" * 60)

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    wrapped_client = wrap_openai(client)

    response = await wrapped_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is 2+2?"},
        ],
    )

    print(f"Response: {response.choices[0].message.content}\n")


async def async_streaming_example():
    """Example 4: Asynchronous streaming chat completion"""
    print("\n" + "=" * 60)
    print("Example 4: Async Streaming")
    print("=" * 60)

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    wrapped_client = wrap_openai(client)

    response = await wrapped_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "List 3 colors."},
        ],
        stream=True,
    )

    print("Response: ", end="", flush=True)
    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def main():
    # Check for required environment variables
    if not os.environ.get("STATSIG_API_KEY"):
        print("Error: STATSIG_API_KEY environment variable not set")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize StatsigAI
    StatsigAI.new_shared(
        statsig_source=StatsigCreateConfig(
            sdk_key=os.environ.get("STATSIG_API_KEY"),
        )
    )
    StatsigAI.shared().initialize()

    try:
        # Run all examples
        sync_non_streaming_example()
        sync_streaming_example()
        asyncio.run(async_non_streaming_example())
        asyncio.run(async_streaming_example())

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("Check your Statsig dashboard for telemetry.")
        print("=" * 60 + "\n")

    finally:
        StatsigAI.shared().shutdown()
        print("✓ Check your Statsig console for Gen AI events and traces")


if __name__ == "__main__":
    main()
