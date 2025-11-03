from typing import Optional, TypedDict

from dataclasses import dataclass


class AIEvalGradeData(TypedDict, total=False):
    session_id: Optional[str]
