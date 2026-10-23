"""Pure helpers for ForgeGraph; deliberately independent of Flask and databases."""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable


def finite_number(value: Any, field_name: str) -> float:
    """Parse a finite numeric sensor value or raise ValueError."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def classify_sensor(pressure: float, temperature: float, vibration: float) -> dict[str, Any]:
    """Simple, transparent thresholds; not a certified diagnosis."""
    flags: list[str] = []
    if pressure < 260:
        flags.append("hydraulic pressure below 260 bar")
    if temperature > 50:
        flags.append("oil temperature above 50 C")
    if vibration > 4.5:
        flags.append("vibration above 4.5 mm/s")
    severity = "critical" if pressure < 245 or vibration >= 7 or temperature >= 60 else (
        "warning" if flags else "normal"
    )
    return {"is_anomaly": bool(flags), "severity": severity, "flags": flags}


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 1}


def rank_documents(query: str, documents: Iterable[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    """Transparent lexical ranking used only by the no-database demo mode."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    ranked: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        content = f"{document.get('title', '')} {document.get('text', '')} {document.get('source', '')}"
        doc_tokens = tokenize(content)
        overlap = query_tokens & doc_tokens
        if not overlap:
            continue
        score = len(overlap) / max(1, len(query_tokens))
        row = dict(document)
        row["similarity"] = round(score, 3)
        ranked.append((score, row))
    ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("title", ""))))
    return [row for _, row in ranked[:max(1, limit)]]


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    """Split text into bounded, overlapping chunks without requiring NLP packages."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return []
    if max_chars < 100 or overlap < 0 or overlap >= max_chars:
        raise ValueError("max_chars must be >= 100 and 0 <= overlap < max_chars")
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        if end < len(normalized):
            boundary = normalized.rfind(". ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary + 1
        part = normalized[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_model_text(payload: Any) -> str | None:
    """Support Ollama generate/chat and OpenAI-compatible chat-completion shapes."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            cleaned = payload.strip()
            return cleaned or None
    if not isinstance(payload, dict):
        return None

    candidates: list[Any] = [payload.get("response")]
    message = payload.get("message")
    if isinstance(message, dict):
        candidates.append(message.get("content"))
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice_message = choices[0].get("message")
        if isinstance(choice_message, dict):
            candidates.append(choice_message.get("content"))
        candidates.append(choices[0].get("text"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def build_retrieval_query(pressure: float, temperature: float, vibration: float) -> str:
    traits: list[str] = []
    if pressure < 260:
        traits.append("low hydraulic pressure possible leak seal pump or relief valve")
    if temperature > 50:
        traits.append("high oil temperature overheating cooling lubrication viscosity")
    if vibration > 4.5:
        traits.append("high vibration bearing misalignment looseness mechanical fault")
    return "Forging press maintenance: " + ("; ".join(traits) if traits else "normal operation preventive maintenance")


def build_diagnostic_prompt(pressure: float, temperature: float, vibration: float,
                            flags: list[str], evidence: list[dict[str, Any]], graph_facts: list[str] | None = None) -> str:
    evidence_text = "\n\n".join(
        f"[{i}] title={item.get('title', 'Untitled')} | source={item.get('source', 'unknown')}\n"
        f"{item.get('text', item.get('chunk_text', ''))}"
        for i, item in enumerate(evidence, start=1)
    ) or "No matching maintenance documents were retrieved."
    graph_text = "\n".join(f"- {fact}" for fact in (graph_facts or [])) or "No graph relationships matched these symptoms."
    return (
        "You are an industrial maintenance assistant for a 1500-ton forging press. "
        "Do not claim certainty or invent manual references. Distinguish measured facts from hypotheses. "
        "Give: (1) brief assessment, (2) up to three safe inspection steps, (3) evidence citations as [1], [2]. "
        "If evidence is insufficient, say so. Do not instruct bypassing safety interlocks.\n\n"
        f"Measured telemetry: hydraulic pressure={pressure:.2f} bar; oil temperature={temperature:.2f} C; "
        f"vibration={vibration:.2f} mm/s.\nDetected threshold flags: {', '.join(flags) if flags else 'none'}.\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\nCurated symptom/component/incident graph relationships (hypotheses, not confirmed causes):\n{graph_text}"
    )


def demo_diagnosis(pressure: float, temperature: float, vibration: float,
                   flags: list[str], evidence: list[dict[str, Any]], graph_facts: list[str] | None = None) -> str:
    """Deterministic local-only fallback clearly labelled as a rule-based demo, not Gemma."""
    if not flags:
        return "Rule-based demo: readings are within configured thresholds. Continue routine monitoring; this is not an AI diagnosis."
    clauses = []
    if pressure < 260:
        clauses.append("low pressure can be consistent with a leak, seal wear, or a relief-valve issue")
    if temperature > 50:
        clauses.append("elevated oil temperature warrants checking cooling, oil condition, and load")
    if vibration > 4.5:
        clauses.append("high vibration warrants inspection for looseness, misalignment, or bearing wear")
    evidence_note = " Retrieved references: " + ", ".join(
        str(item.get("title", "untitled")) for item in evidence[:3]
    ) + "." if evidence else " No matching maintenance reference was found."
    return "Rule-based demo (not Gemma): " + "; ".join(clauses).capitalize() + ". Confirm with the approved procedure and a qualified engineer before maintenance." + evidence_note
