"""Read the existing published evaluation partition without changing its IDs."""
import hashlib


def evaluation_rows(rows: list[dict], split: dict, per_role: int | None = None) -> list[dict]:
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(split["by_id"]):
        raise ValueError("Corpus and frozen partition IDs must match exactly")
    output = []
    for role, label in (("eval_malicious", "harmful"), ("eval_benign", "safe")):
        selected = [r for r in rows if split["by_id"][r["id"]] == role]
        if len(selected) != split["counts"][role]:
            raise ValueError(f"Partition count mismatch: {role}")
        if per_role is not None:
            if per_role < 1 or per_role > len(selected):
                raise ValueError("Smoke size outside partition")
            selected = selected[:per_role]
        for row in selected:
            if row["label"] != label or not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"Invalid evaluation row: {row['id']}")
            output.append({"id": row["id"], "role": role, "label": label, "prompt": row["text"],
                           "prompt_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                           "source_tag": row["source_tag"], "row_index": row["row_index"]})
    return output
