from typing import Dict, Optional, TypedDict


class AIEvalGradeData(TypedDict, total=False):
    session_id: Optional[str]
    activity_id: Optional[str]
    # Defaults to the active OTel span. Set these when grading after the call's span has ended.
    trace_id: Optional[str]
    span_id: Optional[str]
    reasoning: Optional[str]
    metadata: Optional[Dict[str, str]]
