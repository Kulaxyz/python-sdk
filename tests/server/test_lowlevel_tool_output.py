"""Tests for tool output in low-level server."""

import json
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import pytest

from mcp.client.session import ClientSession
from mcp.server import Server
from mcp.server.lowlevel import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.session import ServerSession
from mcp.shared.message import SessionMessage
from mcp.shared.session import RequestResponder
from mcp.types import ClientResult, ServerNotification, ServerRequest, TextContent, Tool


class ToolTestServer:
    """Helper class to reduce boilerplate in tool tests."""

    def __init__(self, server_name: str = "test"):
        self.server = Server(server_name)
        self.result = None

    def set_tools(self, tools: list[Tool]):
        """Set the tools that the server will expose."""

        @self.server.list_tools()
        async def list_tools():
            return tools

    def set_handler(self, handler: Callable[[str, dict], Awaitable[Any]]):
        """Set the tool call handler."""

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            return await handler(name, arguments)

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Run the server and call a tool, returning the result."""
        server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream[SessionMessage](10)
        client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream[SessionMessage](10)

        async def message_handler(
            message: RequestResponder[ServerRequest, ClientResult] | ServerNotification | Exception,
        ) -> None:
            if isinstance(message, Exception):
                raise message

        async def run_server():
            async with ServerSession(
                client_to_server_receive,
                server_to_client_send,
                InitializationOptions(
                    server_name="test-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            ) as server_session:
                async with anyio.create_task_group() as tg:

                    async def handle_messages():
                        async for message in server_session.incoming_messages:
                            await self.server._handle_message(message, server_session, {}, False)

                    tg.start_soon(handle_messages)
                    await anyio.sleep_forever()

        async with anyio.create_task_group() as tg:
            tg.start_soon(run_server)

            async with ClientSession(
                server_to_client_receive,
                client_to_server_send,
                message_handler=message_handler,
            ) as client_session:
                await client_session.initialize()
                self.result = await client_session.call_tool(tool_name, arguments)
                tg.cancel_scope.cancel()

        return self.result


@pytest.mark.anyio
async def test_lowlevel_server_traditional_tool_output():
    """Test that traditional content block output still works."""
    test_server = ToolTestServer()

    test_server.set_tools(
        [
            Tool(
                name="echo",
                description="Echo a message back",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                    "required": ["message"],
                },
            )
        ]
    )

    async def handler(name: str, arguments: dict):
        if name == "echo":
            message = arguments.get("message", "")
            return [TextContent(type="text", text=f"Echo: {message}")]
        else:
            raise ValueError(f"Unknown tool: {name}")

    test_server.set_handler(handler)
    result = await test_server.call_tool("echo", {"message": "Hello World"})

    # Verify traditional output
    assert result is not None
    assert result.content is not None
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "Echo: Hello World"
    assert result.structuredContent is None
    assert result.isError is False


@pytest.mark.anyio
async def test_lowlevel_server_structured_tool_output():
    """Test that structured dict output works correctly."""
    test_server = ToolTestServer()

    test_server.set_tools(
        [
            Tool(
                name="get_user",
                description="Get user information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                    },
                    "required": ["user_id"],
                },
                outputSchema={
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["id", "name", "email"],
                },
            )
        ]
    )

    expected_output = {
        "id": 42,
        "name": "John Doe",
        "email": "john@example.com",
    }

    async def handler(name: str, arguments: dict):
        if name == "get_user":
            return expected_output
        else:
            raise ValueError(f"Unknown tool: {name}")

    test_server.set_handler(handler)
    result = await test_server.call_tool("get_user", {"user_id": 42})

    assert result is not None
    assert result.content is not None
    assert len(result.content) == 1
    assert result.content[0].type == "text"

    parsed_content = json.loads(result.content[0].text)
    assert parsed_content == expected_output

    assert result.structuredContent is not None
    assert result.structuredContent == expected_output
    assert result.isError is False


@pytest.mark.anyio
async def test_lowlevel_server_structured_tool_output_complex():
    """Test structured output with nested and complex data."""
    test_server = ToolTestServer()

    test_server.set_tools(
        [
            Tool(
                name="get_organization",
                description="Get organization information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "org_id": {"type": "string"},
                    },
                    "required": ["org_id"],
                },
            )
        ]
    )

    expected_output = {
        "id": "test-org",
        "name": "Acme Corp",
        "employees": [
            {"id": 1, "name": "Alice", "role": "CEO"},
            {"id": 2, "name": "Bob", "role": "CTO"},
        ],
        "metadata": {
            "founded": 2020,
            "public": True,
            "tags": ["tech", "startup"],
        },
    }

    async def handler(name: str, arguments: dict):
        if name == "get_organization":
            return expected_output
        else:
            raise ValueError(f"Unknown tool: {name}")

    test_server.set_handler(handler)
    result = await test_server.call_tool("get_organization", {"org_id": "test-org"})

    assert result is not None
    assert result.content is not None
    assert len(result.content) == 1
    assert result.content[0].type == "text"

    parsed_content = json.loads(result.content[0].text)
    assert parsed_content == expected_output

    assert result.structuredContent is not None
    assert result.structuredContent == expected_output
    assert result.isError is False


@pytest.mark.anyio
async def test_lowlevel_server_no_schema_validation():
    """Test that low-level server does NOT validate against schemas."""
    test_server = ToolTestServer()

    test_server.set_tools(
        [
            Tool(
                name="strict_tool",
                description="Tool with strict schema",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "required_field": {"type": "string"},
                    },
                    "required": ["required_field"],
                },
                outputSchema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "integer"},
                    },
                    "required": ["result"],
                },
            )
        ]
    )

    invalid_output = {
        "result": "not an integer",
        "extra_field": "should not be here",
        "another_extra": 123,
    }

    async def handler(name: str, arguments: dict):
        if name == "strict_tool":
            return invalid_output
        else:
            raise ValueError(f"Unknown tool: {name}")

    test_server.set_handler(handler)
    result = await test_server.call_tool("strict_tool", {"wrong_field": "value"})

    assert result is not None
    assert result.isError is False
    assert result.content is not None
    assert len(result.content) == 1
    assert result.content[0].type == "text"

    parsed_content = json.loads(result.content[0].text)
    assert parsed_content == invalid_output

    assert result.structuredContent is not None
    assert result.structuredContent == invalid_output


@pytest.mark.anyio
async def test_lowlevel_server_unstructured_multiple_content_blocks():
    """Test tool output with multiple content blocks."""
    test_server = ToolTestServer()

    test_server.set_tools(
        [
            Tool(
                name="multi_output",
                description="Tool that returns multiple content blocks",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                    },
                    "required": ["count"],
                },
            )
        ]
    )

    async def handler(name: str, arguments: dict):
        if name == "multi_output":
            count = arguments.get("count", 3)
            return [TextContent(type="text", text=f"Line {i+1}") for i in range(count)]
        else:
            raise ValueError(f"Unknown tool: {name}")

    test_server.set_handler(handler)
    result = await test_server.call_tool("multi_output", {"count": 5})

    # Verify multiple content blocks
    assert result is not None
    assert result.content is not None
    assert len(result.content) == 5
    for i, content in enumerate(result.content):
        assert content.type == "text"
        assert content.text == f"Line {i+1}"
    assert result.structuredContent is None
    assert result.isError is False


@pytest.mark.anyio
async def test_lowlevel_server_empty_tool_output():
    """Test tool that returns empty content."""
    test_server = ToolTestServer()

    test_server.set_tools(
        [
            Tool(
                name="noop",
                description="Tool that returns nothing",
                inputSchema={"type": "object"},
            )
        ]
    )

    async def handler(name: str, arguments: dict):
        if name == "noop":
            return []  # Empty iterable
        else:
            raise ValueError(f"Unknown tool: {name}")

    test_server.set_handler(handler)
    result = await test_server.call_tool("noop", {})

    # Verify empty output
    assert result is not None
    assert result.content is not None
    assert len(result.content) == 0
    assert result.structuredContent is None
    assert result.isError is False


@pytest.mark.anyio
async def test_lowlevel_server_mixed_content_types():
    """Test tool output with mixed content types."""
    from mcp.types import ImageContent

    test_server = ToolTestServer()

    test_server.set_tools(
        [
            Tool(
                name="mixed_content",
                description="Tool that returns mixed content types",
                inputSchema={"type": "object"},
            )
        ]
    )

    async def handler(name: str, arguments: dict):
        if name == "mixed_content":
            return [
                TextContent(type="text", text="Here's some text"),
                ImageContent(
                    type="image",
                    data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                    mimeType="image/png",
                ),
                TextContent(type="text", text="And more text after the image"),
            ]
        else:
            raise ValueError(f"Unknown tool: {name}")

    test_server.set_handler(handler)
    result = await test_server.call_tool("mixed_content", {})

    # Verify mixed content
    assert result is not None
    assert result.content is not None
    assert len(result.content) == 3

    assert result.content[0].type == "text"
    assert result.content[0].text == "Here's some text"

    assert result.content[1].type == "image"
    assert result.content[1].mimeType == "image/png"
    assert result.content[1].data.startswith("iVBORw0KGgo")

    assert result.content[2].type == "text"
    assert result.content[2].text == "And more text after the image"

    assert result.structuredContent is None
    assert result.isError is False
