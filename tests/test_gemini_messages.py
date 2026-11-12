from typing import Any, List

import pytest

from statsig_ai.wrappers.configs import GenAICaptureOptions
from statsig_ai.wrappers.gemini import (
    aggregate_gemini_stream_chunks,
    extract_gemini_request_attributes,
    extract_gemini_response_attributes,
)
from statsig_ai.wrappers.gemini_messages import (
    to_gemini_finish_reason,
    to_gemini_input_messages,
    to_gemini_output_messages,
    to_gemini_parts,
)

from mock_gemini import (
    Blob,
    Candidate,
    Content,
    FileData,
    FinishReason,
    FunctionCall,
    FunctionResponse,
    GenerateContentResponse,
    Part,
    UsageMetadata,
)

CAPTURE_ALL = GenAICaptureOptions(capture_all=True)


def user(*parts: Any) -> dict:
    return {"role": "user", "parts": list(parts)}


def text(content: str) -> dict:
    return {"type": "text", "content": content}


# contents


@pytest.mark.parametrize(
    "contents, expected",
    [
        ("Hi", [user(text("Hi"))]),
        (["Hi", "there"], [user(text("Hi"), text("there"))]),
        (Part(text="Hi"), [user(text("Hi"))]),
        ([Part(text="Hi"), {"text": "there"}], [user(text("Hi"), text("there"))]),
        (Content(role="user", parts=[Part(text="Hi")]), [user(text("Hi"))]),
        ({"role": "user", "parts": [{"text": "Hi"}]}, [user(text("Hi"))]),
        (None, [user()]),
        ([], [user()]),
    ],
    ids=[
        "string",
        "strings",
        "part",
        "parts",
        "content",
        "content dict",
        "none",
        "empty list",
    ],
)
def test_contents_are_converted_to_user_messages(contents, expected):
    assert to_gemini_input_messages(contents) == expected


def test_a_content_list_keeps_each_role():
    contents = [
        Content(role="user", parts=[Part(text="Is checkout on?")]),
        Content(role="model", parts=[Part(function_call=FunctionCall(name="list_gates"))]),
        Content(
            role="user",
            parts=[Part(function_response=FunctionResponse(name="list_gates", response={"n": 1}))],
        ),
        Content(role="model", parts=[Part(text="Yes.")]),
    ]

    assert to_gemini_input_messages(contents) == [
        user(text("Is checkout on?")),
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "list_gates", "name": "list_gates"}],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "list_gates", "response": {"n": 1}}],
        },
        {"role": "assistant", "parts": [text("Yes.")]},
    ]


def test_dict_contents_can_use_the_api_camel_case_names():
    contents = [
        {"role": "model", "parts": [{"functionCall": {"id": "call_1", "name": "get", "args": {}}}]},
        {"role": "user", "parts": [{"functionResponse": {"id": "call_1", "response": {"ok": 1}}}]},
        {"role": "user", "parts": [{"inlineData": {"mimeType": "audio/wav"}}]},
    ]

    assert to_gemini_input_messages(contents) == [
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "call_1", "name": "get", "arguments": {}}],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call_1", "response": {"ok": 1}}],
        },
        user({"type": "blob", "modality": "audio", "mime_type": "audio/wav"}),
    ]


def test_parts_outside_a_content_are_grouped_into_user_messages_as_the_sdk_sends_them():
    contents = [
        "Look at this",
        Part(file_data=FileData(file_uri="gs://b/cat.png", mime_type="image/png")),
        Content(role="model", parts=[Part(text="A cat.")]),
        ["And", "this?"],
        "Thanks",
    ]

    assert to_gemini_input_messages(contents) == [
        user(
            text("Look at this"),
            {"type": "uri", "modality": "image", "uri": "gs://b/cat.png", "mime_type": "image/png"},
        ),
        {"role": "assistant", "parts": [text("A cat.")]},
        user(text("And"), text("this?")),
        user(text("Thanks")),
    ]


def test_a_message_of_function_responses_alone_is_a_tool_message():
    responses = Content(
        role="user",
        parts=[
            Part(function_response=FunctionResponse(id="a", name="f", response={"x": 1})),
            Part(function_response=FunctionResponse(name="g", response={"y": 2})),
        ],
    )
    mixed = Content(
        role="user",
        parts=[Part(text="Here"), Part(function_response=FunctionResponse(name="g", response={}))],
    )

    assert to_gemini_input_messages([responses, mixed]) == [
        {
            "role": "tool",
            "parts": [
                {"type": "tool_call_response", "id": "a", "response": {"x": 1}},
                {"type": "tool_call_response", "id": "g", "response": {"y": 2}},
            ],
        },
        user(text("Here"), {"type": "tool_call_response", "id": "g", "response": {}}),
    ]


# Parts


def test_parts_are_converted_and_unsupported_parts_are_left_out():
    parts = [
        Part(text="Thinking", thought=True),
        Part(text=""),
        Part(inline_data=Blob(mime_type="image/jpeg", data=b"...")),
        Part(inline_data=Blob()),
        Part(file_data=FileData(file_uri="https://x/doc.pdf", mime_type="application/pdf")),
        Part(file_data=FileData(file_uri="https://x/file")),
        {"executable_code": {"code": "print(1)"}},
        Part(),
    ]

    assert to_gemini_parts(parts) == [
        {"type": "reasoning", "content": "Thinking"},
        text(""),
        {"type": "blob", "modality": "image", "mime_type": "image/jpeg"},
        {"type": "blob", "modality": "unknown"},
        {
            "type": "uri",
            "modality": "application",
            "uri": "https://x/doc.pdf",
            "mime_type": "application/pdf",
        },
        {"type": "uri", "modality": "unknown", "uri": "https://x/file"},
    ]


def test_a_function_call_id_falls_back_to_its_name():
    parts = [
        Part(function_call=FunctionCall(id="call_9", name="get_gate", args={"id": "a"})),
        Part(function_call=FunctionCall(name="list_gates")),
    ]

    assert to_gemini_parts(parts) == [
        {"type": "tool_call", "id": "call_9", "name": "get_gate", "arguments": {"id": "a"}},
        {"type": "tool_call", "id": "list_gates", "name": "list_gates"},
    ]


# System instructions


@pytest.mark.parametrize(
    "system_instruction",
    [
        "Be brief. Cite gates.",
        ["Be brief.", " Cite gates."],
        Part(text="Be brief. Cite gates."),
        Content(parts=[Part(text="Be brief."), Part(text=" Cite gates.")]),
        {"parts": [{"text": "Be brief."}, {"text": " Cite gates."}]},
    ],
    ids=["string", "strings", "part", "content", "content dict"],
)
def test_the_system_instruction_is_recorded_as_parts(system_instruction):
    attrs = extract_gemini_request_attributes(
        "google.models.generate_content",
        {"model": "m", "contents": "Hi", "config": {"system_instruction": system_instruction}},
        CAPTURE_ALL,
    )

    assert "".join(part["content"] for part in attrs["gen_ai.system_instructions"]) == (
        "Be brief. Cite gates."
    )
    assert {part["type"] for part in attrs["gen_ai.system_instructions"]} == {"text"}


def test_the_system_instruction_can_be_read_from_a_config_object():
    class Config:
        system_instruction = "Be brief."
        max_output_tokens = 32

    attrs = extract_gemini_request_attributes(
        "google.models.generate_content",
        {"model": "m", "contents": "Hi", "config": Config()},
        CAPTURE_ALL,
    )

    assert attrs["gen_ai.system_instructions"] == [text("Be brief.")]
    assert attrs["gen_ai.request.max_tokens"] == 32


# Finish reasons


@pytest.mark.parametrize(
    "finish_reason, expected",
    [
        (None, "stop"),
        (FinishReason.STOP, "stop"),
        ("STOP", "stop"),
        (FinishReason.MAX_TOKENS, "length"),
        (FinishReason.SAFETY, "content_filter"),
        (FinishReason.RECITATION, "content_filter"),
        (FinishReason.BLOCKLIST, "content_filter"),
        (FinishReason.PROHIBITED_CONTENT, "content_filter"),
        (FinishReason.SPII, "content_filter"),
        (FinishReason.OTHER, "other"),
        (FinishReason.MALFORMED_FUNCTION_CALL, "malformed_function_call"),
    ],
)
def test_finish_reasons_are_mapped_like_the_node_sdk(finish_reason, expected):
    assert to_gemini_finish_reason(finish_reason, [text("ok")]) == expected


def test_a_candidate_that_calls_a_function_finishes_with_tool_call():
    parts: List[Any] = [{"type": "tool_call", "name": "get_gate"}]

    assert to_gemini_finish_reason(FinishReason.STOP, parts) == "tool_call"
    assert to_gemini_finish_reason(FinishReason.MAX_TOKENS, parts) == "tool_call"


def test_each_candidate_is_an_output_message():
    candidates = [
        Candidate(content=Content(role="model", parts=[Part(text="A")]), finish_reason="STOP"),
        Candidate(content=None, finish_reason=FinishReason.SAFETY),
    ]

    assert to_gemini_output_messages(candidates) == [
        {"role": "assistant", "parts": [text("A")], "finish_reason": "stop"},
        {"role": "assistant", "parts": [], "finish_reason": "content_filter"},
    ]


# Responses and streams


def test_response_attributes_leave_out_finish_reasons_when_no_candidate_has_one():
    response = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text="A")]))]
    )

    attrs = extract_gemini_response_attributes(response, GenAICaptureOptions())

    assert "gen_ai.response.finish_reasons" not in attrs
    assert "gen_ai.output.messages" not in attrs


def test_stream_chunks_keep_the_first_id_and_the_last_usage_and_finish_reason():
    chunks = [
        GenerateContentResponse(
            response_id="resp_1",
            model_version="gemini-001",
            candidates=[Candidate(content=Content(parts=[Part(text="Hel")]))],
            usage_metadata=UsageMetadata(prompt_token_count=4),
        ),
        {"responseId": "resp_ignored", "candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
        GenerateContentResponse(
            response_id="resp_ignored",
            candidates=[Candidate(finish_reason=FinishReason.MAX_TOKENS)],
            usage_metadata=UsageMetadata(prompt_token_count=4, candidates_token_count=2),
        ),
    ]

    merged = aggregate_gemini_stream_chunks(chunks)
    attrs = extract_gemini_response_attributes(merged, CAPTURE_ALL)

    assert attrs["gen_ai.response.id"] == "resp_1"
    assert attrs["gen_ai.response.model"] == "gemini-001"
    assert attrs["gen_ai.usage.input_tokens"] == 4
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert attrs["gen_ai.response.finish_reasons"] == ["MAX_TOKENS"]
    assert attrs["gen_ai.output.messages"] == [
        {"role": "assistant", "parts": [text("Hello")], "finish_reason": "length"}
    ]


def test_an_empty_stream_has_no_output_messages():
    attrs = extract_gemini_response_attributes(aggregate_gemini_stream_chunks([]), CAPTURE_ALL)

    assert attrs == {
        "gen_ai.response.id": None,
        "gen_ai.response.model": None,
        "gen_ai.output.messages": [],
    }
