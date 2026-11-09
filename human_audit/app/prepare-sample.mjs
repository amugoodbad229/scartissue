import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

const sample = JSON.parse(await readFile(new URL("../../results/human_audit_sample_20261106.json", import.meta.url), "utf8"));
const gate = JSON.parse(await readFile(new URL("../../results/calibration_gate_20261106.json", import.meta.url), "utf8"));
assert.equal(sample.sample_sha256, gate.sample_sha256, "Sample differs from the registered gate");
assert.equal(sample.items.length, gate.sample_items);
assert.equal(new Set(sample.items.map((item) => item.item_id)).size, sample.items.length);
const hash = (value) => createHash("sha256").update(value).digest("hex");
// Match Python json.dumps separators used by the frozen sampler.
const identity = `[${sample.items.map((i) => `[${JSON.stringify(i.item_id)}, ${JSON.stringify(i.answer_sha256)}]`).join(", ")}]`;
assert.equal(hash(identity), sample.sample_sha256);
const fields = ["item_id", "prompt", "answer", "answer_sha256"];
const items = sample.items.map((item) => {
  for (const field of fields) assert.equal(typeof item[field], "string");
  assert.equal(hash(item.answer), item.answer_sha256);
  return Object.fromEntries(fields.map((key) => [key, item[key]]));
});
const blind = { sample_sha256: sample.sample_sha256, items, gate: {
  minimum_labelled_items: gate.minimum_labelled_items,
  minimum_class_items: gate.minimum_class_items,
  thresholds: gate.thresholds,
} };
await writeFile(new URL("src/sample.json", import.meta.url), JSON.stringify(blind, null, 2) + "\n");
console.log(`Prepared ${items.length} blind items; identity ${sample.sample_sha256}`);
