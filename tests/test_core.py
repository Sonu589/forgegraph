from pathlib import Path
import pytest

from forgegraph.graph import local_graph_context, start_nodes
from forgegraph.core import (
    build_diagnostic_prompt, build_retrieval_query, chunk_text, classify_sensor,
    demo_diagnosis, extract_model_text, finite_number, rank_documents,
)

DOCS = [
    {"title": "Hydraulic Seal Checklist", "source": "manual-a.md", "text": "Low hydraulic pressure may indicate a seal leak. Isolate energy before inspecting."},
    {"title": "Vibration Guide", "source": "manual-b.md", "text": "High vibration may relate to looseness or bearing wear."},
]


def test_normal_readings_are_not_flagged():
    result = classify_sensor(280, 44, 2.1)
    assert result["is_anomaly"] is False
    assert result["severity"] == "normal"
    assert result["flags"] == []


def test_pressure_and_vibration_flags_are_reported():
    result = classify_sensor(250, 45, 5.2)
    assert result["is_anomaly"] is True
    assert result["severity"] == "warning"
    assert any("pressure" in flag for flag in result["flags"])
    assert any("vibration" in flag for flag in result["flags"])


def test_critical_severity_threshold():
    assert classify_sensor(240, 44, 2)["severity"] == "critical"


def test_retrieval_ranks_relevant_maintenance_document():
    hits = rank_documents("low hydraulic pressure seal leak", DOCS)
    assert hits
    assert hits[0]["title"] == "Hydraulic Seal Checklist"
    assert 0 <= hits[0]["similarity"] <= 1


def test_chunk_text_empty_and_overlap():
    assert chunk_text("   ") == []
    chunks = chunk_text("word " * 500, max_chars=150, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 150 for chunk in chunks)


@pytest.mark.parametrize("payload, expected", [
    ({"response": "  pressure drop  "}, "pressure drop"),
    ({"choices": [{"message": {"content": " inspect seal"}}]}, "inspect seal"),
    ({"message": {"content": "vibration issue"}}, "vibration issue"),
    ({"response": ""}, None),
    ({"choices": []}, None),
    (None, None),
])
def test_extract_model_text_variants(payload, expected):
    assert extract_model_text(payload) == expected


def test_finite_number_rejects_nan_and_invalid_values():
    assert finite_number("280.2", "pressure") == pytest.approx(280.2)
    with pytest.raises(ValueError):
        finite_number("NaN", "pressure")
    with pytest.raises(ValueError):
        finite_number("bad", "pressure")


def test_prompt_distinguishes_evidence_and_warns_against_certainty():
    prompt = build_diagnostic_prompt(250, 55, 5.4, ["low pressure"], DOCS[:1])
    assert "Retrieved evidence" in prompt
    assert "Hydraulic Seal Checklist" in prompt
    assert "Do not claim certainty" in prompt
    assert "250.00 bar" in prompt


def test_demo_diagnosis_says_rule_based_not_gemma():
    result = demo_diagnosis(250, 55, 5.4, ["low pressure"], DOCS[:1])
    assert "Rule-based demo" in result
    assert "not Gemma" in result
    assert "Hydraulic Seal Checklist" in result


def test_retrieval_query_tracks_only_relevant_signals():
    query = build_retrieval_query(250, 55, 5.4)
    assert "low hydraulic pressure" in query
    assert "high oil temperature" in query
    assert "high vibration" in query


def test_graph_context_links_pressure_symptom_to_procedure_and_incident():
    graph = local_graph_context(["hydraulic pressure below 260 bar", "vibration above 4.5 mm/s"])
    labels = {node["label"] for node in graph["nodes"]}
    assert "Hydraulic pressure drop" in labels
    assert "Hydraulic inspection checklist" in labels
    assert any("recorded_with" in fact for fact in graph["facts"])


def test_graph_start_nodes_ignore_normal_readings():
    assert start_nodes([]) == []
    assert start_nodes(["hydraulic pressure below 260 bar"]) == ["symptom:pressure_drop"]


def test_dashboard_renders_external_text_without_inner_html():
    script = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text()
    assert "textContent" in script
    assert ".innerHTML" not in script
