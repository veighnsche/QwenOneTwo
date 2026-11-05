"""Inspect unreviewed model-proposed development drafts without running models.

Validates DRAFT model proposals only. Never approves, reviews, trains,
fits, runs models, or weakens reviewed-release gates in
``qwen_one_two.training.release``. Drafts must remain rejected by the
reviewed-release loader (state draft, reviews empty).

F-DRAFT-01: labels.payload is null for every non-assessed status including
NoFeasibleChoice; Choice feasibility lives in labels.feasibility.
F-DRAFT-02: no keyword censorship (answer-blindness is structural, with
semantic review pending); declared source paths are constrained to the
drafts root and final validation binds actual file bytes (inline content
is transient lane-handoff only); Choice selections are unique strings and
feasibility values are exact booleans (or the string "unknown" outside
assessed cases).
F-DRAFT-03: sealed output_hash must equal the canonical record hash with
only output_hash itself excluded; null output_hash is lane-handoff only.
"""

import argparse
import json
from pathlib import Path, PurePosixPath

from qwen_one_two.training.inputs import (
    Binding,
    Field,
    Node,
    Provenance,
    Source,
    canonical,
    check,
    digest,
    reference_input,
    validate_record_manifest,
)
from qwen_one_two.training.release import DOMAINS, decode

DRAFT_SCHEMA_ID = "draft-schema-v1.1"
OPERATIONS = frozenset({"Noul", "Choice", "Score"})
STATUSES = frozenset({"Assessed", "Insufficient", "Conflicting", "Unresolved", "NoFeasibleChoice"})
SPLITS = frozenset({"development"})

def _nonempty_str(value, name):
    check(isinstance(value, str) and value.strip(), name)


# Answer-blindness is enforced structurally, never by keyword censorship: request
# semantics live in `semantics`, outcomes live in `labels`/`reason`, and the two
# objects are validated separately with provenance-bound role builders. Legitimate
# supplied facts and prose may contain outcome-related words; whether a wording
# smuggles a verdict is a semantic judgment for human review, not a lexical gate.
# No model classifier is used here.


def load_jsonl(path):
    data = Path(path).read_bytes()
    check(b"\x00" not in data, "NUL byte in JSONL")
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
    records = []
    for index, line in enumerate(lines):
        try:
            record = decode(line.encode("utf-8"))
        except ValueError as exc:
            raise ValueError(f"line {index + 1}: {exc}") from exc
        check(isinstance(record, dict), f"line {index + 1}: not an object")
        records.append(record)
    return records


def _source_bytes(entry, drafts_root, *, allow_inline_content=False):
    # Declared paths are always constrained to the drafts root: absolute paths,
    # parent escapes and symlink escapes reject before any bytes are read.
    check(isinstance(entry.get("path"), str) and entry["path"].strip(), "source path")
    relative = PurePosixPath(entry["path"])
    check(not relative.is_absolute() and ".." not in relative.parts, "source path escapes drafts root")
    if "content" in entry:
        # Transient lane-handoff text only. Final records must bind the actual
        # declared file bytes; inline content never satisfies final validation.
        check(allow_inline_content, "inline source content is lane-handoff only")
        check(isinstance(entry["content"], str), "transient source content type")
        return entry["content"].encode("utf-8")
    root = Path(drafts_root).resolve()
    candidate = (root.joinpath(*relative.parts)).resolve()
    check(candidate.is_relative_to(root), "source path escapes drafts root")
    check(candidate.is_file(), f"missing source file {entry['path']}")
    return candidate.read_bytes()


def _exercise_role_builders(record, provenance):
    semantics = record["semantics"]
    exercised = []
    if "question" in semantics:
        fields = {k: Field(v, f"q-{k}") for k, v in semantics["question"].items()}
        reference_input("question", fields, provenance, permitted_nodes=set(provenance.nodes))
        exercised.append("question")
    if "target" in semantics:
        fields = {k: Field(v, f"t-{k}") for k, v in semantics["target"].items()}
        reference_input("target", fields, provenance, permitted_nodes=set(provenance.nodes))
        exercised.append("target")
    for index, alt in enumerate(semantics.get("alternatives", [])):
        fields = {k: Field(v, f"a{index}-{k}") for k, v in alt.items()}
        reference_input("alternative", fields, provenance, permitted_nodes=set(provenance.nodes))
        exercised.append(f"alternative[{index}]")
    if "rubric" in semantics:
        for field_name in ("dimension", "conditions", "zero", "one", "midpoint"):
            if field_name in semantics["rubric"]:
                reference_input(
                    field_name,
                    {"meaning": Field(semantics["rubric"][field_name], f"r-{field_name}")},
                    provenance,
                    permitted_nodes=set(provenance.nodes),
                )
                exercised.append(f"rubric/{field_name}")
    for index, key in enumerate(semantics.get("attribute_keys", [])):
        fields = {k: Field(v, f"k{index}-{k}") for k, v in key.items()}
        reference_input("key", fields, provenance, permitted_nodes=set(provenance.nodes))
        exercised.append(f"key[{index}]")
    reason = record.get("reason")
    if reason is not None:
        fields = {
            "status": Field(reason["status"], "reason-status"),
            "support": Field(reason["support"], "reason-support"),
        }
        reference_input(
            "reason",
            fields,
            provenance,
            permitted_nodes=set(provenance.nodes),
            operation=record["labels"]["operation"],
        )
        exercised.append("reason")
    return exercised


def validate_draft_record(record, drafts_root, parents, *, allow_inline_content=False):
    check(isinstance(record, dict), "record type")
    for key in ("id", "family"):
        _nonempty_str(record.get(key), f"identity {key}")
    check(type(record.get("version")) is int and record["version"] > 0, "version")
    check(record.get("split") in SPLITS, "split must be development")
    check(isinstance(record.get("parents"), list) and record["parents"] == [], "parents must be []")
    check(record.get("origin") == "model_proposal", "origin must be model_proposal")
    check(record.get("fictional") is True, "fictional must be true")
    check(record.get("domain") in DOMAINS, "domain")
    _nonempty_str(record.get("author"), "author")
    check(record.get("effort_proposal") is None, "effort_proposal must stay null")
    check(record.get("reviews") == [], "reviews must stay []")
    release = record.get("release", {})
    check(release.get("state") == "draft", "release.state must be draft")

    construction = record.get("construction", {})
    for key in ("scenario_spec", "template_version", "transformation"):
        _nonempty_str(construction.get(key), f"construction {key}")
    check("seed" in construction, "construction seed key")
    check(construction.get("transformation") == "original", "transformation must be original")

    human = record.get("human_input", {})
    _nonempty_str(human.get("prompt"), "human_input.prompt")
    for key in ("supplied_facts", "supplied_rules"):
        values = record.get(key)
        check(isinstance(values, list) and len(values) >= 1, f"{key} presence")
        for value in values:
            _nonempty_str(value, key)

    sources = record.get("sources")
    check(isinstance(sources, list) and len(sources) >= 1, "sources presence")
    seen_source_ids = set()
    nodes = {}
    provenance_sources = {}
    for entry in sources:
        check(set(entry) - {"id", "version", "path", "sha256", "byte_start", "byte_end", "content"} == set(),
              "source descriptor fields")
        _nonempty_str(entry.get("id"), "source id")
        check(entry["id"] not in seen_source_ids, "duplicate source identity in record")
        seen_source_ids.add(entry["id"])
        check(entry.get("version") == 1, "source version")
        _nonempty_str(entry.get("path"), "source path")
        raw = _source_bytes(entry, drafts_root, allow_inline_content=allow_inline_content)
        raw.decode("utf-8")
        check(digest(raw) == entry.get("sha256"), f"source hash mismatch {entry['id']}")
        start, end = entry.get("byte_start"), entry.get("byte_end")
        check(type(start) is int and type(end) is int and 0 <= start < end <= len(raw), "source byte coordinates")
        raw[:start].decode("utf-8")
        raw[:end].decode("utf-8")
        excerpt = raw[start:end]
        check(len(excerpt.strip()) > 0, "empty source binding")
        provenance_sources[entry["id"]] = Source(entry["id"], 1, raw, entry["sha256"])
        binding = Binding(entry["id"], 1, start, end)
        nodes[f"src-{entry['id']}"] = Node("source_fact", digest(excerpt), binding=binding)

    semantics = record.get("semantics")
    check(isinstance(semantics, dict) and "question" in semantics and "target" in semantics, "semantics question/target")
    question = semantics["question"]
    check(set(question) <= {"proposition", "criterion", "scope", "qualifications", "polarity", "constraints"},
          "question fields")
    _nonempty_str(question.get("proposition"), "question proposition")
    _nonempty_str(question.get("criterion"), "question criterion")
    target = semantics["target"]
    check(set(target) == {"referent"}, "target fields")
    _nonempty_str(target.get("referent"), "target referent")

    operation = record.get("labels", {}).get("operation")
    check(operation in OPERATIONS, "labels.operation")
    if operation == "Choice":
        alternatives = semantics.get("alternatives")
        check(isinstance(alternatives, list) and 2 <= len(alternatives) <= 16, "Choice alternatives count")
        identities = []
        for alt in alternatives:
            check(set(alt) == {"identity", "properties"}, "alternative fields")
            _nonempty_str(alt.get("identity"), "alternative identity")
            _nonempty_str(alt.get("properties"), "alternative properties")
            identities.append(alt["identity"])
        check(len(set(identities)) == len(identities), "duplicate alternative identity")
    else:
        check("alternatives" not in semantics, f"alternatives only for Choice, got {operation}")
    if operation == "Score":
        rubric = semantics.get("rubric")
        check(isinstance(rubric, dict), "Score rubric")
        for key in ("dimension", "conditions", "direction", "zero", "one"):
            _nonempty_str(rubric.get(key), f"rubric {key}")
        check(rubric.get("direction") in ("higher", "lower"), "rubric direction")
        check(set(rubric) <= {"dimension", "conditions", "direction", "zero", "one", "midpoint"}, "rubric fields")
    else:
        if operation == "Noul":
            check("rubric" not in semantics, "rubric only for Score")
    for key_entry in semantics.get("attribute_keys", []):
        check(set(key_entry) == {"property", "unit"}, "attribute key fields")
        _nonempty_str(key_entry.get("property"), "attribute property")
        _nonempty_str(key_entry.get("unit"), "attribute unit")

    labels = record.get("labels")
    check(labels.get("status") in STATUSES, "labels.status")
    status = labels["status"]
    check(not (status == "NoFeasibleChoice" and operation != "Choice"), "Choice-only NoFeasibleChoice")
    _nonempty_str(labels.get("justification"), "labels justification")
    payload = labels.get("payload")
    if status == "Assessed":
        check(isinstance(payload, dict), "assessed payload")
        if operation == "Noul":
            check(payload.get("p_true") in (0, 1), "Noul p_true determinate 0|1")
            check(set(payload) == {"p_true"}, "Noul payload fields")
        elif operation == "Choice":
            identities = [a["identity"] for a in semantics["alternatives"]]
            feasibility = labels.get("feasibility")
            check(isinstance(feasibility, dict) and set(feasibility) == set(identities), "Choice feasibility coverage")
            # An any-workable request can be assessed with one confirmed option
            # even when other candidates are unknown (result-statuses.md).
            check(all(type(v) is bool or (type(v) is str and v == "unknown")
                      for v in feasibility.values()), "Choice feasibility exact values")
            check(set(payload) == {"acceptable_selection_ids"}, "Choice payload holds selections only")
            selection = payload.get("acceptable_selection_ids")
            check(isinstance(selection, list) and len(selection) >= 1, "Choice acceptable selections")
            check(all(isinstance(s, str) for s in selection)
                  and len(set(selection)) == len(selection), "unique string selections")
            check(set(selection) <= set(identities), "Choice selection membership")
            check(all(feasibility[i] is True for i in selection), "selection must be feasible")
        else:
            interval = payload.get("score_interval")
            check(isinstance(interval, list) and len(interval) == 2, "Score interval")
            lo, hi = interval
            check(isinstance(lo, (int, float)) and isinstance(hi, (int, float))
                  and not isinstance(lo, bool) and not isinstance(hi, bool), "Score interval types")
            check(0 <= lo <= hi <= 1 and (hi - lo) <= 0.20 + 1e-12, "Score interval range/width")
            check(labels.get("unreviewed") is True, "Score proposal must be explicitly marked unreviewed")
            if "value_proposal" in payload:
                candidate = payload["value_proposal"]
                check(isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
                      and 0 <= candidate <= 1, "value_proposal range")
                check(lo <= candidate <= hi, "value_proposal inside interval")
    else:
        check(payload is None, "non-assessed payload must be null")
        if operation == "Choice":
            identities = [a["identity"] for a in semantics["alternatives"]]
            feasibility = labels.get("feasibility")
            check(isinstance(feasibility, dict) and set(feasibility) == set(identities), "Choice feasibility coverage")
            check(all(type(v) is bool or (type(v) is str and v == "unknown")
                      for v in feasibility.values()), "Choice feasibility exact values")
            if status == "NoFeasibleChoice":
                check(all(v is False for v in feasibility.values()), "NoFeasibleChoice needs every entry false")
    if operation != "Choice":
        check("feasibility" not in labels, "feasibility is Choice-only")

    reason = record.get("reason")
    if status == "Assessed":
        check(reason is None, "Reason only for supported non-assessed status")
    else:
        check(isinstance(reason, dict), "reason presence")
        check(reason.get("status") == status, "reason status mirrors labels.status")
        _nonempty_str(reason.get("support"), "reason support")

    guidance = record.get("guidance", {})
    check(isinstance(guidance.get("acceptable_continuations"), list)
          and len(guidance["acceptable_continuations"]) >= 1, "acceptable continuations")
    check(isinstance(guidance.get("acceptable_routes"), list)
          and len(guidance["acceptable_routes"]) >= 1, "acceptable routes")
    check(isinstance(guidance.get("direct_answer_allowed"), bool), "direct_answer_allowed")

    exposure = record.get("exposure", {})
    check(exposure.get("kind") == "offline_supervised_proposal", "exposure kind")
    check(exposure.get("claimed_model_rollout") is False, "never claim a model rollout")

    assistance = record.get("model_assistance", {})
    for key in ("checkpoint", "prompt_hash", "sampling", "software", "gpu", "output_hash"):
        check(key in assistance, f"missing model provenance {key}")
    _nonempty_str(assistance.get("run_id"), "model_assistance run_id")
    if assistance["gpu"] is None:
        _nonempty_str(assistance.get("gpu_reason"), "gpu_reason")
    else:
        _nonempty_str(assistance["gpu"], "gpu provenance")
    stated_output = assistance.get("output_hash")
    if stated_output is None:
        # Transient lane handoff only; sealed records carry a verified hash.
        check(allow_inline_content, "final output_hash must be sealed; null only in lane handoff")
    else:
        check(isinstance(stated_output, str) and len(stated_output) == 64
              and all(c in "0123456789abcdef" for c in stated_output), "output_hash hex")
        sealed = {k: v for k, v in record.items()}
        sealed_ma = dict(sealed["model_assistance"])
        sealed_ma.pop("output_hash", None)
        sealed["model_assistance"] = sealed_ma
        check(digest(canonical(sealed)) == stated_output, "output_hash mismatch with canonical record")

    validate_record_manifest(record, parents)

    for key, text in (
        ("q-proposition", question["proposition"]),
        ("q-criterion", question["criterion"]),
        *((f"q-{k}", v) for k, v in question.items() if k not in ("proposition", "criterion")),
        ("t-referent", target["referent"]),
    ):
        nodes[key] = Node("request_semantics", digest(text.encode()))
    if operation == "Choice":
        for index, alt in enumerate(semantics["alternatives"]):
            nodes[f"a{index}-identity"] = Node("request_semantics", digest(alt["identity"].encode()))
            nodes[f"a{index}-properties"] = Node("request_semantics", digest(alt["properties"].encode()))
    if operation == "Score":
        for field_name in ("dimension", "conditions", "zero", "one", "midpoint"):
            if field_name in semantics["rubric"]:
                text = semantics["rubric"][field_name]
                nodes[f"r-{field_name}"] = Node("request_semantics", digest(text.encode()))
    for index, key_entry in enumerate(semantics.get("attribute_keys", [])):
        nodes[f"k{index}-property"] = Node("request_semantics", digest(key_entry["property"].encode()))
        nodes[f"k{index}-unit"] = Node("request_semantics", digest(key_entry["unit"].encode()))
    if reason is not None:
        nodes["reason-status"] = Node("reason", digest(reason["status"].encode()))
        nodes["reason-support"] = Node(
            "reason", digest(reason["support"].encode()),
            parents=tuple(f"src-{entry['id']}" for entry in sources),
        )
    provenance = Provenance(nodes, provenance_sources)
    exercised = _exercise_role_builders(record, provenance)
    return exercised


def validate_batch(records, drafts_root, *, allow_inline_content=False):
    check(bool(records), "empty draft batch")
    ids = [r.get("id") for r in records]
    check(len(set(ids)) == len(ids), "duplicate ids")
    parents = {}
    exercised_all = {}
    for record in records:
        exercised_all[record["id"]] = validate_draft_record(
            record, drafts_root, parents, allow_inline_content=allow_inline_content)
    return exercised_all


def report_for(scenarios_path, drafts_root=None):
    scenarios_path = Path(scenarios_path)
    records = load_jsonl(scenarios_path)
    root = Path(drafts_root) if drafts_root is not None else scenarios_path.parent
    exercised = validate_batch(records, root)
    from collections import Counter
    summary = {
        "format": "qot-draft-batch-v1",
        "schema": DRAFT_SCHEMA_ID,
        "state": "draft",
        "records": len(records),
        "domains": dict(Counter(r["domain"] for r in records)),
        "operations": dict(Counter(r["labels"]["operation"] for r in records)),
        "statuses": dict(Counter(r["labels"]["status"] for r in records)),
        "reviews": [],
        "approval": "absent",
        "semantic_authentication": False,
        "role_builders_exercised": exercised,
        "scenario_sha256": digest(scenarios_path.read_bytes()),
        "note": "Model assistance only; human review absent. Reviewed-release loader rejection must remain.",
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", help="scenarios.jsonl path")
    parser.add_argument("drafts_root", nargs="?", default=None,
                        help="drafts directory holding sources/ (defaults to the scenarios parent)")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    try:
        _, summary = report_for(args.scenarios, args.drafts_root)
    except (ValueError, KeyError, TypeError, OSError, UnicodeDecodeError) as exc:
        parser.exit(1, "REJECTED: " + str(exc) + "\n")
    text = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
