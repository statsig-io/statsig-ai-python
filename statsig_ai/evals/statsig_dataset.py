import dataclasses
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, cast

import requests

from .eval_types import EvalDataRecord, Input, Output, _DataclassOrDictEvalDataRecord

_STATSIG_EVAL_DATASETS_ENDPOINT = "https://api.statsig.com/console/v1/eval_datasets"
_PAGE_SIZE = 100


@dataclass
class StatsigDatasetRow:
    row_id: str
    # Cell values by column name
    columns: Dict[str, str]
    groups: List[str] = field(default_factory=list)


StatsigDatasetToRecord = Callable[
    [StatsigDatasetRow], _DataclassOrDictEvalDataRecord[Input, Output]
]


class StatsigDataset(Generic[Input, Output]):
    """
    Eval data loaded from a Statsig eval dataset. Each record keeps its dataset
    row id and the run names the dataset, so the console compares runs on the
    dataset row by row.
    """

    def __init__(self, name: str, to_record: StatsigDatasetToRecord[Input, Output]):
        self.name = name
        self._to_record = to_record

    def load(self, api_key: str) -> List[_DataclassOrDictEvalDataRecord[Input, Output]]:
        records: List[_DataclassOrDictEvalDataRecord[Input, Output]] = []
        page = 1
        while True:
            response = requests.get(
                f"{_STATSIG_EVAL_DATASETS_ENDPOINT}/{urllib.parse.quote(self.name, safe='')}"
                f"/rows?page={page}&limit={_PAGE_SIZE}",
                headers={"STATSIG-API-KEY": api_key},
                timeout=10,
            )
            if not response.ok:
                raise RuntimeError(
                    f"[Statsig] Failed to load eval dataset '{self.name}': "
                    f"{response.status_code} {response.reason}"
                )
            rows = response.json().get("data") or []
            for row in rows:
                dataset_row = StatsigDatasetRow(
                    row_id=row["rowID"],
                    columns={column["name"]: column["value"] for column in row["columns"]},
                    groups=row.get("groups") or [],
                )
                records.append(_with_row_id(self._to_record(dataset_row), dataset_row.row_id))
            if len(rows) < _PAGE_SIZE:
                return records
            page += 1


def _with_row_id(
    record: _DataclassOrDictEvalDataRecord[Input, Output], row_id: str
) -> _DataclassOrDictEvalDataRecord[Input, Output]:
    if isinstance(record, EvalDataRecord):
        return dataclasses.replace(record, row_id=row_id)
    if isinstance(record, dict):
        return cast(_DataclassOrDictEvalDataRecord[Input, Output], {**record, "row_id": row_id})
    # Eval() rejects anything else as an invalid data record.
    return record


def statsig_dataset(
    name: str, to_record: StatsigDatasetToRecord[Input, Output]
) -> StatsigDataset[Input, Output]:
    """
    Loads a Statsig eval dataset as `Eval()` data. `to_record` says which
    columns hold the input and expected value, e.g.
    `lambda row: {"input": row.columns["question"], "expected": row.columns["answer"]}`.
    """
    return StatsigDataset(name, to_record)
