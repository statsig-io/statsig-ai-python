STATSIG_ATTR_SPAN_TYPE = "statsig.span.type"
STATSIG_ATTR_USER_ID = "statsig.user_id"
STATSIG_ATTR_CUSTOM_IDS = "statsig.custom_ids"
STATSIG_ATTR_ACTIVITY_ID = "statsig.activity_id"
STATSIG_ATTR_ENVIRONMENT = "statsig.environment"
# Incubating in the OpenTelemetry semantic conventions, which recommend copying it.
ATTR_DEPLOYMENT_ENVIRONMENT_NAME = "deployment.environment.name"

STATSIG_CTX_KEY = "statsig.context"
STATSIG_CTX_KEY_ACTIVE_USER = "statsig.active_user"
STATSIG_CTX_KEY_ACTIVE_ENVIRONMENT = "statsig.active_environment"


class StatsigSpanType:
    GEN_AI = "gen_ai"
