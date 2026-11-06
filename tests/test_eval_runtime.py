from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from eval_runtime import require_finite, validate_patch


def example():
    return {"d": np.array([1., 0.]), "alpha": -1., "modules": ["w"],
            "cols": {"w": np.array([0, 2])}, "proj": {"w": np.array([2., 3.])}}


def test_valid_patch_counts_selected_scalars():
    assert validate_patch(example()) == 4


@pytest.mark.parametrize("columns", [[0, 0], [2, 0], [-1, 2], [0., 2.]])
def test_duplicate_unsorted_negative_or_float_columns_are_rejected(columns):
    patch = example()
    patch["cols"]["w"] = np.array(columns)
    with pytest.raises(ValueError, match="Columns"):
        validate_patch(patch)


@pytest.mark.parametrize("field", ["d", "alpha", "proj"])
def test_nonfinite_patch_fails_before_application(field):
    patch = copy.deepcopy(example())
    if field == "alpha":
        patch[field] = float("nan")
    elif field == "proj":
        patch[field]["w"][0] = float("inf")
    else:
        patch[field][0] = float("nan")
    with pytest.raises(ValueError):
        validate_patch(patch)


def test_measurement_guard_reports_location():
    with pytest.raises(ValueError, match=r"\$\.arms\[0\]\.score"):
        require_finite({"arms": [{"score": np.float64("nan")}]})


def test_non_numeric_metadata_is_allowed():
    require_finite({"stderr": "N/A", "count": 2, "score": 0.5, "metadata": None})


def test_named_chat_template_is_resolved_through_tokenizer_api():
    from eval_runtime import resolved_chat_template

    class Tokenizer:
        chat_template = {"default": "{{ messages }}", "tool_use": "{{ tools }}"}

        def get_chat_template(self):
            return self.chat_template["default"]

    assert resolved_chat_template(Tokenizer()) == "{{ messages }}"


def sample_records():
    return [{"doc_id": i, "filter": f, "resps": [[f"answer {i}"]],
             "prompt_hash": f"prompt-{i}", "doc_hash": f"doc-{i}", "target_hash": f"target-{i}"}
            for f in ["strict-match", "flexible-extract"] for i in range(2)]


def test_harness_retains_one_record_per_document_and_filter():
    from eval_runtime import validate_harness_samples
    check = validate_harness_samples(sample_records(), 2, ["strict-match", "flexible-extract"])
    assert check["unique_documents"] == 2
    assert check["sample_records"] == 4


@pytest.mark.parametrize("defect", ["missing_filter", "missing_doc", "changed_response", "changed_prompt", "duplicate_filter"])
def test_harness_coverage_rejects_missing_or_misaligned_records(defect):
    from eval_runtime import validate_harness_samples
    records = sample_records()
    if defect == "missing_filter":
        records.pop()
    elif defect == "missing_doc":
        records = [r for r in records if r["doc_id"] != 1]
    elif defect == "changed_response":
        records[-1]["resps"] = [["different generation"]]
    elif defect == "changed_prompt":
        records[-1]["prompt_hash"] = "different-prompt"
    else:
        records[-1]["filter"] = "strict-match"
    with pytest.raises(ValueError):
        validate_harness_samples(records, 2, ["strict-match", "flexible-extract"])


def test_failed_nonfinite_payload_retains_text_and_explicit_error_marker():
    from eval_runtime import preserve_invalid_numbers
    failed = preserve_invalid_numbers({"score": np.float64("nan"), "resps": [["full text"]]})
    assert failed == {"score": {"invalid_nonfinite_number": "nan"}, "resps": [["full text"]]}
    require_finite(failed)


def test_json_transport_removes_remote_metadata_classes():
    import json
    from eval_runtime import encode_result

    class RemoteLibraryVersion(str):
        pass

    decoded = json.loads(encode_result({"version": RemoteLibraryVersion("2.9.0"),
                                        "score": np.float64(.5), "count": np.int64(2),
                                        "array": np.array([0., 1.])}))
    assert type(decoded["version"]) is str
    assert decoded == {"version": "2.9.0", "score": .5, "count": 2, "array": [0., 1.]}
    with pytest.raises(TypeError, match="Unsupported result object"):
        encode_result({"score": object()})


def test_json_transport_records_library_metadata_objects():
    import json
    from eval_runtime import encode_result

    Dtype = type("dtype", (), {"__str__": lambda self: "torch.bfloat16"})
    Dtype.__module__ = "torch"
    decoded = json.loads(encode_result({"config": {"dtype": Dtype()}, "score": 0.5}))
    assert decoded["config"]["dtype"] == {"library_object": "torch.bfloat16",
                                         "python_type": "torch.dtype"}
    assert decoded["score"] == 0.5
