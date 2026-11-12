from typing import Any, Dict, List

import pytest
from opentelemetry import context as otel_context

from statsig_ai import PromptVersion
from statsig_ai._vendor.combustache import MissingClosingTagError
from statsig_ai.otel.conventions import STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES
from statsig_ai.otel.prompt_context import (
    get_prompt_span_attrs_from_context,
    set_prompt_version_to_context,
)


class FakeDynamicConfig:
    def __init__(self, name: str, values: Dict[str, Any]) -> None:
        self.name = name
        self._values = values

    def get_string(self, key: str, fallback: str) -> str:
        return self._values.get(key, fallback)

    def get_array(self, key: str, fallback: list) -> list:
        return self._values.get(key, fallback)


def make_version(prompts: List[Dict[str, str]]) -> PromptVersion:
    config = FakeDynamicConfig("prompt_config:version-a", {"name": "V", "prompts": prompts})
    return PromptVersion(config, False)  # type: ignore[arg-type]


def render(template: str, params: Dict[str, Any]) -> str:
    messages = make_version([{"role": "user", "content": template}]).get_prompt_messages(params)
    return messages[0]["content"]


def test_plain_variables():
    assert render("Hello {{name}}, you are {{ role }}.", {"name": "Ada", "role": "admin"}) == (
        "Hello Ada, you are admin."
    )


def test_nested_paths():
    params = {"user": {"profile": {"name": "Ada"}}, "items": [{"name": "first"}]}

    assert render("{{user.profile.name}}", params) == "Ada"
    # mustache.js reads array indexes as dotted names too.
    assert render("{{items.0.name}}", params) == "first"


def test_sections_iterate_lists():
    params = {"items": [{"name": "a"}, {"name": "b"}], "tags": ["x", "y"]}

    assert render("{{#items}}- {{name}}\n{{/items}}", params) == "- a\n- b\n"
    assert render("{{#tags}}[{{.}}]{{/tags}}", params) == "[x][y]"


def test_sections_use_objects_and_truthy_values_as_context():
    params = {"user": {"name": "Ada"}, "name": "outer", "flag": True, "count": 2}

    assert render("{{#user}}{{name}}{{/user}}", params) == "Ada"
    assert render("{{#flag}}{{name}}{{/flag}}", params) == "outer"
    assert render("{{#count}}{{.}}{{/count}}", params) == "2"


@pytest.mark.parametrize("value", [False, 0, "", None, []])
def test_sections_skip_falsy_values(value):
    assert render("{{#x}}shown{{/x}}", {"x": value}) == ""


@pytest.mark.parametrize("value", [False, 0, "", None, []])
def test_inverted_sections_render_for_falsy_or_empty_values(value):
    assert render("{{^x}}empty{{/x}}", {"x": value}) == "empty"


def test_inverted_sections_skip_truthy_values():
    assert render("{{^x}}empty{{/x}}", {"x": ["a"]}) == ""
    assert render("{{^x}}empty{{/x}}", {"x": {"a": 1}}) == ""


def test_missing_variables_render_as_empty():
    params = {"user": {"name": "Ada"}}

    assert render("[{{missing}}]", params) == "[]"
    assert render("[{{user.missing}}]", params) == "[]"
    assert render("[{{missing.deep.path}}]", params) == "[]"
    assert render("[{{#missing}}shown{{/missing}}]", params) == "[]"
    assert render("[{{^missing}}shown{{/missing}}]", params) == "[shown]"


def test_values_are_not_html_escaped():
    params = {"x": "<b>&</b> \"quoted\" 'single'"}

    assert render("{{x}}", params) == "<b>&</b> \"quoted\" 'single'"


def test_triple_mustache_and_ampersand_insert_the_raw_value():
    params = {"x": "<b>&</b>"}

    assert render("{{{x}}}|{{& x}}|{{x}}", params) == "<b>&</b>|<b>&</b>|<b>&</b>"


def test_numbers_and_booleans_render_like_mustache_js():
    params = {
        "int": 3,
        "zero": 0,
        "float": 2.5,
        "whole_float": 1.0,
        "yes": True,
        "no": False,
        "none": None,
        "list": ["a", 1, True, None],
    }
    template = "{{int}} {{zero}} {{float}} {{whole_float}} {{yes}} {{no}} [{{none}}] {{list}}"

    assert render(template, params) == "3 0 2.5 1 true false [] a,1,true,"


def test_renders_each_message_and_keeps_its_role():
    version = make_version(
        [
            {"role": "system", "content": "You help {{company}}."},
            {"role": "user", "content": "{{question}}"},
            {"role": "assistant", "content": "No variables here."},
        ]
    )

    assert version.get_prompt_messages({"company": "Acme", "question": "Hi?"}) == [
        {"role": "system", "content": "You help Acme."},
        {"role": "user", "content": "Hi?"},
        {"role": "assistant", "content": "No variables here."},
    ]


def test_records_the_variables_before_rendering():
    version = make_version([{"role": "user", "content": "{{#unclosed}}{{input}}"}])
    ctx = set_prompt_version_to_context(otel_context.get_current(), "p", "V", "version-a")
    token = otel_context.attach(ctx)
    try:
        # Like mustache.js, an unclosed section is an error.
        with pytest.raises(MissingClosingTagError):
            version.get_prompt_messages({"input": "hello"})

        attrs = get_prompt_span_attrs_from_context(include_template_variables=True)
        assert attrs[STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES] == {"input": "hello"}
    finally:
        otel_context.detach(token)
