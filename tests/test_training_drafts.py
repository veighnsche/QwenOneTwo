"""Model-free draft validator checks using synthetic fixtures only."""
import copy
import tempfile
import unittest
from pathlib import Path

from qwen_one_two.training.drafts import load_jsonl, validate_batch, validate_draft_record


def base_record(key="pilot-01", domain="refund_eligibility", operation="Noul", status="Assessed"):
    content = f"Source for {key}: fictional facts and rules.\n"
    obal = {"acceptable_selection_ids": ["a", "b"]}
    feasibility = {"a": True, "b": True}
    if operation == "Noul":
        payload = {"p_true": 1} if status == "Assessed" else None
        semantics = {
            "question": {"proposition": "The customer qualifies under the stated rule", "criterion": "Supplied refund rule"},
            "target": {"referent": "The customer in case alpha"},
        }
    elif operation == "Choice":
        payload = obal if status == "Assessed" else None
        semantics = {
            "question": {"proposition": "Which option satisfies the stated rule", "criterion": "Supplied selection rule"},
            "target": {"referent": "The requester in case beta"},
            "alternatives": [
                {"identity": "a", "properties": "Option a uses bay 1 on day 1"},
                {"identity": "b", "properties": "Option b uses bay 2 on day 2"},
            ],
        }
    else:
        payload = {"score_interval": [0.6, 0.75]} if status == "Assessed" else None
        semantics = {
            "question": {"proposition": "How large is the described delay", "criterion": "Supplied timing rubric"},
            "target": {"referent": "The delivery in case gamma"},
            "rubric": {
                "dimension": "Delay size",
                "conditions": "Delay occurs during the stated window",
                "direction": "higher",
                "zero": "No delay",
                "one": "Full window missed",
            },
        }
    record = {
        "id": key,
        "version": 1,
        "family": "family-" + key,
        "split": "development",
        "parents": [],
        "origin": "model_proposal",
        "fictional": True,
        "pilot": True,
        "domain": domain,
        "author": "muse-spark-1.3-contributor lane TEST proposal",
        "construction": {"scenario_spec": "synthetic fixture", "template_version": "draft-schema-v1",
                         "seed": None, "seed_reason": "sampling seed unexposed", "transformation": "original"},
        "human_input": {"prompt": f"Fictional request {key}: advise on the supplied situation."},
        "supplied_facts": ["Fictional fact one."],
        "supplied_rules": ["Fictional rule one."],
        "sources": [{"id": "s", "version": 1, "path": "sources/x.txt", "sha256": "x",
                      "byte_start": 0, "byte_end": 1, "content": content}],
        "semantics": semantics,
        "labels": {"operation": operation, "status": status, "payload": payload,
                   "justification": "Unreviewed proposal justified by supplied facts; unreviewed" if operation == "Score" and status == "Assessed"
                   else "Justified by supplied facts."} | ({"feasibility": dict(feasibility)} if operation == "Choice" else {}),
        "reason": None if status == "Assessed" else {"status": status, "support": "Missing material date for the target."},
        "guidance": {"acceptable_continuations": ["Ask for the missing date."],
                     "acceptable_routes": [operation, "direct"], "direct_answer_allowed": True},
        "exposure": {"kind": "offline_supervised_proposal", "claimed_model_rollout": False},
        "effort_proposal": None,
        "reviews": [],
        "release": {"state": "draft", "id": "muse-pilot-20261105T191615Z", "version": 1},
        "model_assistance": {"producer": "muse-spark-1.3-contributor", "run_id": "muse-pilot-20261105T191615Z",
                             "lane": "TEST", "checkpoint": None, "checkpoint_reason": "remote weights unexposed",
                             "prompt_hash": "0" * 64, "sampling": None, "sampling_reason": "remote sampling unexposed",
                             "software": "muse-code-1.4.14", "gpu": None,
                             "gpu_reason": "remote provider GPU unexposed; never local RTX3090",
                             "output_hash": None, "output_reason": "coordinator recomputes"},
    }
    if operation == "Score" and status == "Assessed":
        record["labels"]["unreviewed"] = True
    if operation == "Choice" and status not in ("Assessed", "NoFeasibleChoice"):
        record["labels"]["feasibility"] = {"a": "unknown", "b": "unknown"}
    return record, content


def fix_source(record, content):
    import hashlib
    raw = content.encode()
    record["sources"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    record["sources"][0]["byte_end"] = len(raw)
    return record


def seal(record):
    from qwen_one_two.training.inputs import canonical, digest
    ma = record["model_assistance"]
    ma.pop("output_hash", None)
    ma["output_hash"] = digest(canonical({**record, "model_assistance": ma}))
    return record


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def check_one(self, record):
        # Synthetic fixtures use transient inline handoff content; final
        # file-byte binding has its own dedicated tests below.
        return validate_draft_record(record, self.root, {}, allow_inline_content=True)

    def test_assessed_noul_choice_score_pass_and_builders_exercised(self):
        for operation in ("Noul", "Choice", "Score"):
            record, content = base_record("pilot-01", "scheduling", operation, "Assessed")
            fix_source(record, content)
            exercised = self.check_one(record)
            self.assertTrue(exercised)

    def test_nonassessed_requires_reason_and_assessed_rejects_reason(self):
        record, content = base_record("pilot-01", "scheduling", "Noul", "Insufficient")
        fix_source(record, content)
        self.check_one(record)
        record["reason"] = None
        with self.assertRaises(ValueError):
            self.check_one(record)
        record, content = base_record("pilot-01", "scheduling", "Noul", "Assessed")
        fix_source(record, content)
        record["reason"] = {"status": "Insufficient", "support": "x"}
        with self.assertRaises(ValueError):
            self.check_one(record)

    def test_choice_only_no_feasible_and_score_width(self):
        record, content = base_record("pilot-01", "scheduling", "Noul", "NoFeasibleChoice")
        fix_source(record, content)
        record["reason"] = {"status": "NoFeasibleChoice", "support": "x"}
        with self.assertRaises(ValueError):
            self.check_one(record)
        record, content = base_record("pilot-01", "scheduling", "Score", "Assessed")
        fix_source(record, content)
        record["labels"]["payload"] = {"score_interval": [0.0, 0.9]}
        with self.assertRaises(ValueError):
            self.check_one(record)

    def test_score_unreviewed_explicit_and_point_membership(self):
        # F-DRAFT-05: unreviewed must be an explicit flag, never inferred from
        # prose; an optional point must sit inside the proposed interval.
        record, content = base_record("pilot-01", "scheduling", "Score", "Assessed")
        fix_source(record, content)
        del record["labels"]["unreviewed"]
        record["labels"]["justification"] = "Unreviewed proposal justified by supplied facts."
        with self.assertRaisesRegex(ValueError, "explicitly marked unreviewed"):
            self.check_one(record)
        record["labels"]["unreviewed"] = True
        self.check_one(record)
        for bad_point in (0.1, 0.9, "high", True):
            mutated = copy.deepcopy(record)
            mutated["labels"]["payload"] = {"score_interval": [0.4, 0.6],
                                            "value_proposal": bad_point}
            with self.assertRaises(ValueError):
                self.check_one(mutated)
        mutated = copy.deepcopy(record)
        mutated["labels"]["payload"] = {"score_interval": [0.4, 0.6], "value_proposal": 0.5}
        self.check_one(mutated)

    def test_no_feasible_choice_has_no_payload_value(self):
        # F-DRAFT-01: NoFeasibleChoice is non-assessed; labels.payload stays null.
        record, content = base_record("pilot-01", "scheduling", "Choice", "NoFeasibleChoice")
        record["labels"]["feasibility"] = {"a": False, "b": False}
        record["reason"] = {"status": "NoFeasibleChoice",
                            "support": "Both options violate the stated rule for the target."}
        record["labels"]["justification"] = "Both options violate the stated rule."
        fix_source(record, content)
        self.check_one(record)
        record["labels"]["payload"] = {"acceptable_selection_ids": [], "feasibility": {"a": False, "b": False}}
        with self.assertRaises(ValueError):
            self.check_one(record)

    def test_choice_feasibility_lives_outside_payload(self):
        record, content = base_record("pilot-01", "scheduling", "Choice", "Assessed")
        fix_source(record, content)
        self.check_one(record)
        old_shape = copy.deepcopy(record)
        old_shape["labels"]["payload"] = {"acceptable_selection_ids": ["a"],
                                          "feasibility": {"a": True, "b": False}}
        del old_shape["labels"]["feasibility"]
        with self.assertRaises(ValueError):
            self.check_one(old_shape)
        missing = copy.deepcopy(record)
        missing["labels"]["status"] = "Insufficient"
        missing["labels"]["payload"] = None
        del missing["labels"]["feasibility"]
        missing["reason"] = {"status": "Insufficient", "support": "Availability unknown for the target."}
        with self.assertRaises(ValueError):
            self.check_one(missing)
        record, content = base_record("pilot-01", "scheduling", "Noul", "Assessed")
        fix_source(record, content)
        record["labels"]["feasibility"] = {"a": True}
        with self.assertRaises(ValueError):
            self.check_one(record)

    def test_noul_determinate_and_effort_reviews_gpu_gates(self):
        record, content = base_record("pilot-01", "scheduling", "Noul", "Assessed")
        fix_source(record, content)
        record["labels"]["payload"] = {"p_true": 0.8}
        with self.assertRaises(ValueError):
            self.check_one(record)
        for mutate in (
            lambda r: r.update(effort_proposal=0.05),
            lambda r: r.update(reviews=[{"reviewer": "x"}]),
            lambda r: r["model_assistance"].update(gpu="GPU-0"),
        ):
            record, content = base_record("pilot-01", "scheduling", "Noul", "Assessed")
            fix_source(record, content)
            mutate(record)
            with self.assertRaises(ValueError):
                self.check_one(record)

    def test_source_hash_bytes_without_keyword_censorship(self):
        # F-DRAFT-02: no keyword filter. Outcome-related prose passes mechanical
        # validation; whether wording smuggles a verdict is a human-review
        # judgment, not a lexical gate.
        record, content = base_record("pilot-01", "scheduling", "Noul", "Assessed")
        fix_source(record, content)
        record["sources"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check_one(record)
        record, content = base_record("pilot-01", "scheduling", "Noul", "Assessed")
        fix_source(record, content)
        record["semantics"]["target"] = {"referent": "The customer ruled ineligible under the stated rule"}
        record["human_input"]["prompt"] = "Score my request and choose wisely: is a refund due?"
        self.check_one(record)
        record, content = base_record("pilot-01", "scheduling", "Choice", "Assessed")
        fix_source(record, content)
        record["semantics"]["alternatives"][0]["properties"] = "The winning bidder's warehouse on day 1"
        self.check_one(record)

    def test_final_validation_binds_declared_file_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sources").mkdir()
            content = "Declared source bytes.\n"
            (root / "sources" / "x.txt").write_text(content, encoding="utf-8")
            raw = content.encode()
            record, _ = base_record("pilot-01", "scheduling", "Noul", "Assessed")
            record["sources"] = [{"id": "s", "version": 1, "path": "sources/x.txt",
                                  "sha256": hashlib.sha256(raw).hexdigest(),
                                  "byte_start": 0, "byte_end": len(raw)}]
            seal(record)
            validate_draft_record(record, root, {})
            record["sources"][0]["content"] = content
            with self.assertRaisesRegex(ValueError, "lane-handoff"):
                validate_draft_record(record, root, {})
            lane_record, lane_content = base_record("pilot-01", "scheduling", "Noul", "Assessed")
            fix_source(lane_record, lane_content)
            validate_draft_record(lane_record, root, {}, allow_inline_content=True)

    def test_corrupted_final_output_hash_rejected(self):
        # F-DRAFT-03: sealed output_hash must equal the canonical record hash
        # with only output_hash itself excluded.
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sources").mkdir()
            content = "Sealed source bytes.\n"
            (root / "sources" / "x.txt").write_text(content, encoding="utf-8")
            raw = content.encode()
            record, _ = base_record("pilot-01", "scheduling", "Noul", "Assessed")
            record["sources"] = [{"id": "s", "version": 1, "path": "sources/x.txt",
                                  "sha256": hashlib.sha256(raw).hexdigest(),
                                  "byte_start": 0, "byte_end": len(raw)}]
            seal(record)
            validate_draft_record(record, root, {})
            record["model_assistance"]["output_hash"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "output_hash mismatch"):
                validate_draft_record(record, root, {})
            record["model_assistance"]["output_hash"] = None
            with self.assertRaisesRegex(ValueError, "must be sealed"):
                validate_draft_record(record, root, {})

    def test_source_path_escape_rejected(self):
        for bad in ("/absolute/x.txt", "../outside.txt",
                    "sources/../../outside.txt", "sources/sub/../../x.txt"):
            record, content = base_record("pilot-01", "scheduling", "Noul", "Assessed")
            fix_source(record, content)
            record["sources"][0]["path"] = bad
            with self.assertRaisesRegex(ValueError, "escapes drafts root"):
                validate_draft_record(record, self.root, {}, allow_inline_content=True)

    def test_symlink_escape_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sources").mkdir()
            with tempfile.TemporaryDirectory() as other:
                secret = Path(other) / "secret.txt"
                secret.write_text("outside bytes\n", encoding="utf-8")
                (root / "sources" / "link.txt").symlink_to(secret)
                record, _ = base_record("pilot-01", "scheduling", "Noul", "Assessed")
                raw = b"outside bytes\n"
                record["sources"] = [{"id": "s", "version": 1, "path": "sources/link.txt",
                                      "sha256": hashlib.sha256(raw).hexdigest(),
                                      "byte_start": 0, "byte_end": len(raw)}]
                with self.assertRaisesRegex(ValueError, "escapes drafts root"):
                    validate_draft_record(record, root, {})

    def test_choice_selection_and_feasibility_exactness(self):
        record, content = base_record("pilot-01", "scheduling", "Choice", "Assessed")
        fix_source(record, content)
        record["labels"]["payload"] = {"acceptable_selection_ids": ["a", "a"]}
        with self.assertRaisesRegex(ValueError, "unique string"):
            self.check_one(record)
        record["labels"]["payload"] = {"acceptable_selection_ids": ["a", "b", "c"]}
        with self.assertRaises(ValueError):
            self.check_one(record)
        for bad_value in (1, 0):
            mutated = copy.deepcopy(record)
            mutated["labels"]["payload"] = {"acceptable_selection_ids": ["a"]}
            mutated["labels"]["feasibility"] = {"a": bad_value, "b": False}
            with self.assertRaisesRegex(ValueError, "exact boolean"):
                self.check_one(mutated)
        mutated = copy.deepcopy(record)
        mutated["labels"]["payload"] = {"acceptable_selection_ids": ["a"]}
        mutated["labels"]["feasibility"] = {"a": "unknown", "b": False}
        with self.assertRaisesRegex(ValueError, "exact boolean"):
            self.check_one(mutated)
        mutated = copy.deepcopy(record)
        mutated["labels"]["status"] = "NoFeasibleChoice"
        mutated["labels"]["payload"] = None
        mutated["labels"]["feasibility"] = {"a": False, "b": "unknown"}
        mutated["reason"] = {"status": "NoFeasibleChoice",
                             "support": "Both options violate the stated rule for the target."}
        with self.assertRaises(ValueError):
            self.check_one(mutated)

    def test_batch_counts_and_uniqueness(self):
        import hashlib
        import json
        domains = ["refund_eligibility"] * 2 + ["scheduling"] * 2 + ["demo_preparation"] * 2 + ["access_rules"] * 2 + ["resource_allocation"] * 2 + ["issue_prioritization"] * 2
        records = []
        for i in range(1, 13):
            operation = ["Noul", "Choice", "Score"][i % 3]
            status = "Assessed" if i % 2 else "Insufficient"
            key = f"pilot-{i:02d}"
            record, content = base_record(key, domains[i - 1], operation, status)
            if operation == "Choice" and status != "Assessed":
                record["labels"]["payload"] = None
                record["labels"]["feasibility"] = {"a": "unknown", "b": "unknown"}
                record["reason"] = {"status": status, "support": "Availability unknown for the target."}
                record["labels"]["justification"] = "Supported abstention."
            if operation == "Score" and status != "Assessed":
                record["labels"]["payload"] = None
                record["reason"] = {"status": status, "support": "Effect evidence missing for the target."}
                record["labels"]["justification"] = "Supported abstention."
            if operation == "Noul" and status != "Assessed":
                record["labels"]["payload"] = None
            fix_source(record, content)
            record["human_input"]["prompt"] = f"Fictional request {key} distinct prompt {i}."
            records.append(record)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
            exercised = validate_batch(records, tmp, allow_inline_content=True)
            self.assertEqual(len(exercised), 12)
            bad = copy.deepcopy(records)
            bad[0]["id"] = bad[1]["id"]
            with self.assertRaises(ValueError):
                validate_batch(bad, tmp, allow_inline_content=True)
            bad = copy.deepcopy(records)
            bad.pop()
            with self.assertRaises(ValueError):
                validate_batch(bad, tmp, allow_inline_content=True)


if __name__ == "__main__":
    unittest.main()
