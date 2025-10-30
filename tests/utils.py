import json
import os

import pytest
from pytest_httpserver import HTTPServer
from statsig_python_core import Statsig, StatsigOptions


def get_test_data_resource(filename: str) -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "../testdata", filename), "r") as file:
        file_content = file.read()

    return file_content
