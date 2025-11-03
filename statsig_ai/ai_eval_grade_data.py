from typing import Optional

from dataclasses import dataclass


@dataclass
class AIEvalGradeData:
    session_id: Optional[str]
