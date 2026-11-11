STATSIG_ATTR_SPAN_TYPE = "statsig.span.type"
STATSIG_ATTR_GEN_AI_SPAN_TYPE = "statsig.span.gen_ai.type"
STATSIG_ATTR_SPAN_LLM_ROOT = "statsig.span.llm_root"
STATSIG_SPAN_LLM_ROOT_VALUE = "1"
STATSIG_ATTR_LLM_PROMPT_NAME = "statsig.llm.prompt_name"
STATSIG_ATTR_LLM_PROMPT_VERSION = "statsig.llm.prompt_version"
STATSIG_ATTR_LLM_PROMPT_VERSION_ID = "statsig.llm.prompt_version_id"
STATSIG_ATTR_LLM_PROMPT_TEMPLATE_VARIABLES = "statsig.llm.prompt_template_variables"
STATSIG_ATTR_USER_ID = "statsig.user_id"
STATSIG_ATTR_CUSTOM_IDS = "statsig.custom_ids"
STATSIG_ATTR_ACTIVITY_ID = "statsig.activity_id"
STATSIG_ATTR_ENVIRONMENT = "statsig.environment"
# Incubating in the OpenTelemetry semantic conventions, which recommend copying it.
ATTR_DEPLOYMENT_ENVIRONMENT_NAME = "deployment.environment.name"

STATSIG_CTX_KEY = "statsig.context"
STATSIG_CTX_KEY_ACTIVE_PROMPT = "statsig.active_prompt"
STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION = "statsig.active_prompt_version"
STATSIG_CTX_KEY_ACTIVE_PROMPT_VERSION_ID = "statsig.active_prompt_version_id"
STATSIG_CTX_KEY_ACTIVE_PROMPT_TEMPLATE = "statsig.active_prompt_template"
STATSIG_CTX_KEY_ACTIVE_USER = "statsig.active_user"
STATSIG_CTX_KEY_ACTIVE_ENVIRONMENT = "statsig.active_environment"
# Set by the outermost LLM span (a model call, or a workflow span) so the model
# calls under it aren't marked `statsig.span.llm_root`.
STATSIG_CTX_KEY_LLM_ROOT = "statsig.llm_root"


class StatsigSpanType:
    GEN_AI = "gen_ai"


class StatsigGenAISpanType:
    WORKFLOW = "workflow"
    TOOL = "tool"
    CHAT = "chat"
    GENERATE_CONTENT = "generate_content"
