from statsig_python_core import StatsigUser

from statsig_ai import StatsigContext, statsig_context
from statsig_ai.otel.conventions import STATSIG_ATTR_CUSTOM_IDS, STATSIG_ATTR_USER_ID
from statsig_ai.otel.statsig_context import get_statsig_span_attrs_from_context


def test_statsig_context_attrs_match_the_node_sdk():
    user = StatsigUser("user-1", custom_ids={"team": "red"})
    with statsig_context(StatsigContext(user=user, activity_id="activity-1")):
        assert get_statsig_span_attrs_from_context() == {
            STATSIG_ATTR_USER_ID: "user-1",
            STATSIG_ATTR_CUSTOM_IDS: '{"team":"red","statsig.activity_id":"activity-1"}',
        }


def test_user_without_an_id_is_recorded_as_unknown():
    with statsig_context(StatsigContext(user=StatsigUser(custom_ids={"team": "red"}))):
        attrs = get_statsig_span_attrs_from_context()
        assert attrs is not None
        assert attrs[STATSIG_ATTR_USER_ID] == "unknown"


def test_activity_id_alone_sets_custom_ids_but_no_user_id():
    with statsig_context(StatsigContext(activity_id="activity-1")):
        assert get_statsig_span_attrs_from_context() == {
            STATSIG_ATTR_CUSTOM_IDS: '{"statsig.activity_id":"activity-1"}',
        }


def test_non_ascii_custom_ids_are_written_as_is():
    user = StatsigUser("user-1", custom_ids={"team": "café"})
    with statsig_context(StatsigContext(user=user)):
        attrs = get_statsig_span_attrs_from_context()
        assert attrs is not None
        # Like Node's JSON.stringify, non-ASCII text isn't escaped.
        assert attrs[STATSIG_ATTR_CUSTOM_IDS] == '{"team":"café"}'
