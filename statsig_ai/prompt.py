from contextlib import contextmanager
from typing import Generator, List, Optional, TypedDict

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from statsig_python_core import Statsig, StatsigUser

from .otel.conventions import STATSIG_CTX_KEY_ACTIVE_PROMPT
from .otel.prompt_context import set_prompt_version_to_context
from .otel.user_context import set_user_to_otel_context
from .prompt_version import PromptVersion


class PromptEvaluationOptions(TypedDict):
    pass


class Prompt:
    def __init__(
        self,
        name: str,
        live: PromptVersion,
        candidates: List[PromptVersion],
        user: Optional[StatsigUser] = None,
    ):
        self.name = name
        self.live = live
        self.candidates = candidates
        self._user = user

    def get_name(self) -> str:
        return self.name

    def get_live(self) -> PromptVersion:
        return self.live

    def get_candidates(self) -> List[PromptVersion]:
        return self.candidates

    @contextmanager
    def with_live(self) -> Generator[PromptVersion, None, None]:
        """Runs the block with the live version active. See `with_version`."""
        with self.with_version(self.live) as version:
            yield version

    @contextmanager
    def with_version(self, version: PromptVersion) -> Generator[PromptVersion, None, None]:
        """
        Runs the block with the given version active, so wrapped model calls inside it
        are tagged with the prompt name, the version and the template variables passed
        to `version.get_prompt_messages()`. Spans are also tagged with the prompt's user.
        """
        ctx = set_prompt_version_to_context(
            otel_context.get_current(),
            self.name,
            version.get_name(),
            version.get_id(),
        )
        with self._attached(ctx):
            yield version

    @contextmanager
    def with_context(self) -> Generator["Prompt", None, None]:
        """
        Runs the block with this prompt's name and user active, without a version.
        """
        ctx = otel_context.set_value(STATSIG_CTX_KEY_ACTIVE_PROMPT, self.name)
        with self._attached(ctx):
            yield self

    @contextmanager
    def _attached(self, ctx: Context) -> Generator[None, None, None]:
        if self._user is not None:
            ctx = set_user_to_otel_context(ctx, self._user)
        token = otel_context.attach(ctx)
        try:
            yield
        finally:
            otel_context.detach(token)


def make_prompt(statsig: Statsig, name: str, paramStore, user: StatsigUser) -> Prompt:
    live_config_id = paramStore.get_string("live", "") or ""
    candidates_config_ids = paramStore.get_array("candidates", []) or []
    live_config = statsig.get_dynamic_config(user, live_config_id)
    candidates_configs = []
    for config_id in candidates_config_ids:
        config = statsig.get_dynamic_config(user, config_id)
        candidates_configs.append(config)
    return Prompt(
        name,
        PromptVersion(live_config, True),
        [PromptVersion(config, False) for config in candidates_configs],
        user,
    )
