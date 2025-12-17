from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase

from jsonschema import Draft202012Validator


class TestSystemJsonSchema(TestCase):
    def test_schema_validation(self):
        root = Path(__file__).resolve().parents[1]

        schema_path = root / "data" / "schema.json"
        example_path = root / "test" / "data" / "system.json"

        with schema_path.open() as f:
            schema = json.load(f)

        with example_path.open() as f:
            instance = json.load(f)

        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
