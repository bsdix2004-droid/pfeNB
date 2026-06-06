"""
Tests for the unflatten + coerce + export pipeline.

These tests cover the field-name -> nested-dict rebuild used by
ResultService.export_results() so that the API returns a clean nested
JSON file containing only the extracted fields.

Run with:
    pytest tests/test_export_unflatten.py
"""
import json

import pytest

from app.services.result_service import (
    _coerce_value,
    _parse_path,
    _prune_empty,
    _unflatten,
)


# ----- _parse_path -----

def test_parse_simple_path():
    assert _parse_path("personal_info.name") == ["personal_info", "name"]


def test_parse_list_index_path():
    assert _parse_path("education.formation[0].name") == [
        "education", "formation", 0, "name",
    ]


def test_parse_multiple_list_indices():
    assert _parse_path("education.formation[0].dates[1]") == [
        "education", "formation", 0, "dates", 1,
    ]


def test_parse_top_level_list():
    assert _parse_path("items[2]") == ["items", 2]


# ----- _unflatten -----

def test_unflatten_simple_nested():
    flat = {
        "personal_info.name": "Maxime Moreau",
        "personal_info.contact.email": "maxime.moreau@exemple.com",
        "personal_info.contact.phone": "06 12 34 56 78",
    }
    assert _unflatten(flat) == {
        "personal_info": {
            "name": "Maxime Moreau",
            "contact": {
                "email": "maxime.moreau@exemple.com",
                "phone": "06 12 34 56 78",
            },
        },
    }


def test_unflatten_list_items():
    flat = {
        "education.formation[0].name": "BTS SIO",
        "education.formation[0].university": "Lycée René Cassin",
        "education.formation[0].dates[0]": "06/2022",
        "education.formation[1].name": "Bac STMG",
        "education.formation[1].university": "Lycée Camille See",
    }
    assert _unflatten(flat) == {
        "education": {
            "formation": [
                {
                    "name": "BTS SIO",
                    "university": "Lycée René Cassin",
                    "dates": ["06/2022"],
                },
                {
                    "name": "Bac STMG",
                    "university": "Lycée Camille See",
                },
            ],
        },
    }


def test_unflatten_skills_bool_map():
    flat = {
        "skills.analyse des dysfonctionnements": "True",
        "skills.procédures d'entretien": "True",
    }
    flat_coerced = {k: _coerce_value(v, "boolean") for k, v in flat.items()}
    assert _unflatten(flat_coerced) == {
        "skills": {
            "analyse des dysfonctionnements": True,
            "procédures d'entretien": True,
        },
    }


def test_unflatten_invoice_totals_numbers():
    flat = {
        "items[0].description": "Web design",
        "items[0].price": "100.0",
        "items[0].quantity": "1",
        "items[0].total": "100.0",
        "items[1].description": "Hosting",
        "items[1].total": "600.0",
        "totals.subtotal": "700.0",
        "totals.tax": "140.0",
        "totals.total": "840.0",
    }
    coerced = {k: _coerce_value(v, "integer" if "quantity" in k else "number") for k, v in flat.items()}
    assert _unflatten(coerced) == {
        "items": [
            {"description": "Web design", "price": 100.0, "quantity": 1, "total": 100.0},
            {"description": "Hosting", "total": 600.0},
        ],
        "totals": {"subtotal": 700.0, "tax": 140.0, "total": 840.0},
    }


def test_unflatten_drops_empty_values():
    flat = {
        "personal_info.name": "Jane Doe",
        "personal_info.headline": "",
        "personal_info.contact.email": None,
    }
    assert _unflatten(flat) == {"personal_info": {"name": "Jane Doe"}}


def test_unflatten_drops_empty_list_items():
    flat = {
        "work_experience[0].company": "Acme",
        "work_experience[1].company": None,
        "work_experience[2].company": "Globex",
    }
    assert _unflatten(flat) == {
        "work_experience": [{"company": "Acme"}, {"company": "Globex"}],
    }


# ----- _coerce_value -----

def test_coerce_value_string_passthrough():
    assert _coerce_value("hello", "string") == "hello"
    assert _coerce_value("06 12 34 56 78", "string") == "06 12 34 56 78"


def test_coerce_value_boolean():
    assert _coerce_value("True", "boolean") is True
    assert _coerce_value("true", "boolean") is True
    assert _coerce_value("1", "boolean") is True
    assert _coerce_value("False", "boolean") is False
    assert _coerce_value("0", "boolean") is False


def test_coerce_value_integer():
    assert _coerce_value("42", "integer") == 42
    assert isinstance(_coerce_value("42", "integer"), int)


def test_coerce_value_number():
    assert _coerce_value("12.50", "number") == 12.5
    assert isinstance(_coerce_value("12.50", "number"), float)


def test_coerce_value_array():
    assert _coerce_value('["a", "b"]', "array") == ["a", "b"]


def test_coerce_value_object():
    assert _coerce_value('{"k": "v"}', "object") == {"k": "v"}


def test_coerce_value_falls_back_to_text():
    assert _coerce_value("not a number", "integer") == "not a number"


def test_coerce_value_none_returns_none():
    assert _coerce_value(None, "string") is None


# ----- _prune_empty -----

def test_prune_empty_drops_empty_dicts():
    data = {"a": {"b": {}, "c": 1}}
    assert _prune_empty(data) == {"a": {"c": 1}}


def test_prune_empty_drops_empty_lists():
    data = {"a": [None, "", [], {}, {"k": 1}]}
    assert _prune_empty(data) == {"a": [{"k": 1}]}


# ----- End-to-end: flat fields -> nested JSON string -----

def test_full_export_simulation_cv():
    """Simulate the worker's stored flat fields and the export's nested output."""
    flat_fields = {
        "personal_info.name": "Maxime Moreau",
        "personal_info.headline": "Technicien de maintenance informatique",
        "personal_info.contact.email": "maxime.moreau@exemple.com",
        "personal_info.contact.phone": "06 12 34 56 78",
        "personal_info.contact.urls": "[]",
        "education.formation[0].name": "BTS SIO",
        "education.formation[0].university": "Lycée René Cassin",
        "education.formation[0].dates[0]": "06/2022",
        "work_experience[0].company": "Nextinfo",
        "work_experience[0].position": "Technicien de maintenance",
        "work_experience[0].dates[0]": "07/2021 - 12/2021",
        "skills.analyse des dysfonctionnements": "True",
    }
    data_types = {
        "personal_info.contact.urls": "array",
    }

    coerced = {
        path: _coerce_value(value, data_types.get(path))
        for path, value in flat_fields.items()
    }
    nested = _unflatten(coerced)

    exported = json.dumps(nested, ensure_ascii=False, indent=2)
    parsed = json.loads(exported)

    assert parsed == {
        "personal_info": {
            "name": "Maxime Moreau",
            "headline": "Technicien de maintenance informatique",
            "contact": {
                "email": "maxime.moreau@exemple.com",
                "phone": "06 12 34 56 78",
            },
        },
        "education": {
            "formation": [
                {
                    "name": "BTS SIO",
                    "university": "Lycée René Cassin",
                    "dates": ["06/2022"],
                },
            ],
        },
        "work_experience": [
            {
                "company": "Nextinfo",
                "position": "Technicien de maintenance",
                "dates": ["07/2021 - 12/2021"],
            },
        ],
        "skills": {
            "analyse des dysfonctionnements": True,
        },
    }


def test_full_export_has_no_metadata_keys():
    """The export must NOT contain document_type, fields, warnings, evidence."""
    flat_fields = {
        "personal_info.name": "Maxime Moreau",
    }
    nested = _unflatten(flat_fields)
    parsed = json.loads(json.dumps(nested, ensure_ascii=False, indent=2))
    forbidden = {"document_type", "fields", "warnings", "evidence", "document_name"}
    assert forbidden.isdisjoint(parsed.keys())
    assert "data" not in parsed
