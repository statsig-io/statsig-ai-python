import pytest
from openai.types.chat import ChatCompletion
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)

from statsig_ai.wrappers.gen_ai_messages import parse_tool_arguments, text_part
from statsig_ai.wrappers.openai_messages import (
    to_openai_output_messages,
    to_openai_request_messages,
)


def text(content):
    return {"type": "text", "content": content}


def tool_call(call_id, name, arguments):
    return {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}


def tool_result(call_id, response):
    return {"type": "tool_call_response", "id": call_id, "response": response}


def chat_tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# ---------- gen_ai_messages ----------


def test_text_part():
    assert text_part("hi") == {"type": "text", "content": "hi"}


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ('{"id":"checkout"}', {"id": "checkout"}),
        ("{}", {}),
        ("[1, 2]", [1, 2]),
        ('{"id":', '{"id":'),
        ("not json", "not json"),
        ("", ""),
        # JSON.parse rejects NaN and Infinity, so they stay strings.
        ("NaN", "NaN"),
        ('{"x": Infinity}', '{"x": Infinity}'),
    ],
)
def test_parse_tool_arguments_parses_json_and_keeps_anything_else(arguments, expected):
    assert parse_tool_arguments(arguments) == expected


# ---------- Chat Completions requests ----------


def test_chat_request_maps_each_role():
    request = to_openai_request_messages(
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "developer", "content": "Answer in English."},
                {"role": "user", "content": "Is checkout on?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [chat_tool_call("call_1", "list_gates", "{}")],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '["checkout"]'},
                {"role": "assistant", "content": "Checkout is on."},
            ]
        }
    )

    # System and developer messages become system instructions, not input messages.
    assert request.system_instructions == [text("You are helpful."), text("Answer in English.")]
    assert request.input_messages == [
        {"role": "user", "parts": [text("Is checkout on?")]},
        {"role": "assistant", "parts": [tool_call("call_1", "list_gates", {})]},
        {"role": "tool", "parts": [tool_result("call_1", '["checkout"]')]},
        {"role": "assistant", "parts": [text("Checkout is on.")]},
    ]


def test_chat_request_converts_multi_part_content():
    request = to_openai_request_messages(
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are helpful."},
                        {"type": "text", "text": "Be brief."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in these?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                        {"type": "file", "file": {"file_id": "file-1"}},
                        {"type": "unknown", "value": 1},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I can't help with that."},
                        {"type": "refusal", "refusal": "Refused."},
                    ],
                },
            ]
        }
    )

    assert request.system_instructions == [text("You are helpful."), text("Be brief.")]
    assert request.input_messages == [
        {
            "role": "user",
            "parts": [
                text("What is in these?"),
                {"type": "uri", "modality": "image", "uri": "https://example.com/a.png"},
                {"type": "blob", "modality": "audio", "mime_type": "wav"},
                {"type": "blob", "modality": "file"},
            ],
        },
        {"role": "assistant", "parts": [text("I can't help with that."), text("Refused.")]},
    ]


def test_chat_request_keeps_assistant_text_and_parses_tool_call_arguments():
    request = to_openai_request_messages(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        chat_tool_call("call_1", "get_gate", '{"id":"checkout"}'),
                        chat_tool_call("call_2", "get_gate", '{"id":'),
                    ],
                }
            ]
        }
    )

    assert request.input_messages == [
        {
            "role": "assistant",
            "parts": [
                text("Let me check."),
                tool_call("call_1", "get_gate", {"id": "checkout"}),
                # Arguments that aren't valid JSON are kept as the raw string.
                tool_call("call_2", "get_gate", '{"id":'),
            ],
        }
    ]


def test_chat_request_uses_the_function_name_for_legacy_function_results():
    request = to_openai_request_messages(
        {"messages": [{"role": "function", "name": "get_gate", "content": "on"}]}
    )

    assert request.input_messages == [{"role": "tool", "parts": [tool_result("get_gate", "on")]}]


def test_chat_request_leaves_out_ids_it_does_not_have():
    request = to_openai_request_messages(
        {
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]},
                {"role": "tool", "content": "done"},
            ]
        }
    )

    assert request.input_messages == [
        {"role": "assistant", "parts": [{"type": "tool_call", "name": "f", "arguments": ""}]},
        {"role": "tool", "parts": [{"type": "tool_call_response", "response": "done"}]},
    ]


def test_chat_request_skips_messages_it_does_not_know():
    request = to_openai_request_messages(
        {
            "messages": [
                {"role": "critic", "content": "?"},
                42,
                None,
                {"role": "user", "content": "hi"},
            ]
        }
    )

    assert request.input_messages == [{"role": "user", "parts": [text("hi")]}]
    assert request.system_instructions == []


def test_request_without_messages_has_no_messages():
    assert to_openai_request_messages({"model": "gpt-4"}) == ([], [])
    assert to_openai_request_messages({"messages": []}) == ([], [])


# ---------- Responses requests ----------


def test_responses_string_input_is_a_user_message():
    request = to_openai_request_messages({"input": "Regular response test"})

    assert request.input_messages == [{"role": "user", "parts": [text("Regular response test")]}]
    assert request.system_instructions == []


def test_responses_request_maps_instructions_and_input_items():
    request = to_openai_request_messages(
        {
            "instructions": "Be brief.",
            "input": [
                {"role": "developer", "content": "Answer in English."},
                {"role": "user", "content": [{"type": "input_text", "text": "Is checkout on?"}]},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Look up the gate."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "list_gates",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": '["checkout"]'},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checkout is on."}],
                },
            ],
        }
    )

    # `instructions` comes first, then system and developer messages.
    assert request.system_instructions == [text("Be brief."), text("Answer in English.")]
    assert request.input_messages == [
        {"role": "user", "parts": [text("Is checkout on?")]},
        {"role": "assistant", "parts": [{"type": "reasoning", "content": "Look up the gate."}]},
        {"role": "assistant", "parts": [tool_call("call_1", "list_gates", {})]},
        {"role": "tool", "parts": [tool_result("call_1", '["checkout"]')]},
        {"role": "assistant", "parts": [text("Checkout is on.")]},
    ]


def test_responses_request_converts_images_and_files():
    request = to_openai_request_messages(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://example.com/a.png"},
                        {"type": "input_image", "file_id": "file-1"},
                        {"type": "input_file", "file_id": "file-2"},
                    ],
                }
            ]
        }
    )

    assert request.input_messages == [
        {
            "role": "user",
            "parts": [
                {"type": "uri", "modality": "image", "uri": "https://example.com/a.png"},
                {"type": "blob", "modality": "image"},
                {"type": "blob", "modality": "file"},
            ],
        }
    ]


def test_responses_request_reads_output_items_passed_back_as_input():
    # Callers often append a previous response's output items, which are SDK objects.
    reasoning = ResponseReasoningItem.model_validate(
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Hm."}]}
    )
    call = ResponseFunctionToolCall.model_validate(
        {"type": "function_call", "call_id": "call_1", "name": "get_gate", "arguments": '{"id":1}'}
    )
    message = ResponseOutputMessage.model_validate(
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "On.", "annotations": []}],
        }
    )

    request = to_openai_request_messages({"input": [reasoning, call, message]})

    assert request.input_messages == [
        {"role": "assistant", "parts": [{"type": "reasoning", "content": "Hm."}]},
        {"role": "assistant", "parts": [tool_call("call_1", "get_gate", {"id": 1})]},
        {"role": "assistant", "parts": [text("On.")]},
    ]


# ---------- Completions and embeddings requests ----------


def test_completions_prompts_are_user_messages():
    assert to_openai_request_messages({"prompt": "Say hi"}).input_messages == [
        {"role": "user", "parts": [text("Say hi")]}
    ]
    assert to_openai_request_messages({"prompt": ["a", "b"]}).input_messages == [
        {"role": "user", "parts": [text("a")]},
        {"role": "user", "parts": [text("b")]},
    ]


def test_embeddings_token_input_has_no_messages():
    assert to_openai_request_messages({"input": [1, 2, 3]}).input_messages == []


# ---------- Chat Completions responses ----------


def test_chat_response_maps_text_and_finish_reason():
    output = to_openai_output_messages(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Hel"},
                    "finish_reason": "length",
                },
            ]
        }
    )

    assert output == [
        {"role": "assistant", "parts": [text("Hello")], "finish_reason": "stop"},
        {"role": "assistant", "parts": [text("Hel")], "finish_reason": "length"},
    ]


def test_chat_response_maps_tool_calls():
    output = to_openai_output_messages(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            chat_tool_call("call_2", "get_gate", '{"id":"checkout"}'),
                            chat_tool_call("call_3", "get_gate", "{bad json"),
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    assert output == [
        {
            "role": "assistant",
            "parts": [
                tool_call("call_2", "get_gate", {"id": "checkout"}),
                tool_call("call_3", "get_gate", "{bad json"),
            ],
            "finish_reason": "tool_call",
        }
    ]


@pytest.mark.parametrize(
    "finish_reason, expected",
    [
        ("stop", "stop"),
        ("length", "length"),
        ("content_filter", "content_filter"),
        ("tool_calls", "tool_call"),
        ("function_call", "tool_call"),
        (None, "stop"),
    ],
)
def test_chat_response_finish_reasons(finish_reason, expected):
    output = to_openai_output_messages(
        {"choices": [{"message": {"content": "x"}, "finish_reason": finish_reason}]}
    )

    assert output is not None
    assert output[0]["finish_reason"] == expected


def test_chat_response_records_refusals_as_text():
    output = to_openai_output_messages(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None, "refusal": "I can't."},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert output == [{"role": "assistant", "parts": [text("I can't.")], "finish_reason": "stop"}]


def test_text_completion_choices_are_text_parts():
    output = to_openai_output_messages(
        {"choices": [{"text": "Hi there", "index": 0, "finish_reason": "stop"}]}
    )

    assert output == [{"role": "assistant", "parts": [text("Hi there")], "finish_reason": "stop"}]


def test_chat_response_sdk_object_is_converted():
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [chat_tool_call("call_1", "get_gate", '{"id":"checkout"}')],
                    },
                }
            ],
        }
    )

    assert to_openai_output_messages(completion) == [
        {
            "role": "assistant",
            "parts": [tool_call("call_1", "get_gate", {"id": "checkout"})],
            "finish_reason": "tool_call",
        }
    ]


# ---------- Responses responses ----------


def test_responses_output_is_one_assistant_message():
    output = to_openai_output_messages(
        {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Gate is listed."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Checkout is on."},
                        {"type": "refusal", "refusal": "But not that."},
                    ],
                },
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            ],
        }
    )

    assert output == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "Gate is listed."},
                text("Checkout is on."),
                text("But not that."),
            ],
            "finish_reason": "stop",
        }
    ]


def test_responses_output_with_a_function_call_finishes_with_tool_call():
    output = to_openai_output_messages(
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_gate",
                    "arguments": '{"id":"checkout"}',
                }
            ],
        }
    )

    assert output == [
        {
            "role": "assistant",
            "parts": [tool_call("call_1", "get_gate", {"id": "checkout"})],
            "finish_reason": "tool_call",
        }
    ]


@pytest.mark.parametrize(
    "reason, expected",
    [("max_output_tokens", "length"), ("content_filter", "content_filter"), (None, "length")],
)
def test_incomplete_responses_finish_with_length_or_content_filter(reason, expected):
    output = to_openai_output_messages(
        {"status": "incomplete", "incomplete_details": {"reason": reason}, "output": []}
    )

    assert output == [{"role": "assistant", "parts": [], "finish_reason": expected}]


def test_responses_without_messages_have_no_output_messages():
    assert to_openai_output_messages({"object": "list", "data": [{"embedding": [0.1]}]}) is None
    assert to_openai_output_messages({}) is None
    assert to_openai_output_messages(None) is None
