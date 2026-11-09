# ScarTissue: the trade-offs of repairing safety without retraining

**Can a small weight-space edit restore safer behavior without damaging useful behavior?**

ScarTissue studies this question in Cohere Tiny Aya Global. Rather than treating a return of refusal language
as a successful repair, we compare harmful compliance, refusal of benign requests and retained task performance.
The result is a reproducible exploratory study—not a claim that we have made a model production-safe.

**The central finding:** the sparse repair retained more GSM8K performance than the full-support repair,
but both refused more benign requests than the drifted model. A circuit-restricted edit did not provide a
successful repair. Safety judging also failed calibration, so the harm estimates below remain exploratory.

## The research question

Removing a refusal-related direction from model weights—often called *abliteration*—can weaken refusal
behavior. Reintroducing refusal is not enough to show that a repair works: the model might simply refuse
more requests, including harmless ones, or lose useful task performance.

We therefore ask whether a targeted edit offers a better trade-off than a broad edit, and whether a circuit
discovered from refusal behavior is a useful repair target. This is a synthetic weight-space drift setting;
it does not establish that the same repairs work after arbitrary fine-tuning.

The work addresses **Track 2 — AI Red Teaming, Safety and Alignment**:

- **2a: AI Safety and Red Teaming:** Red-teaming and robustness evaluation; Adversarial and safety evaluation sets.
- **2b: AI Safety and Correction:** Repairing safety drift without retraining; Locating where safety lives.

These describe the questions investigated, not a declaration that safety has been successfully localized or repaired.

## What we compared

The main comparison uses the aligned model, its abliterated variant, a sparse repair and a full-support repair.
The sparse repair applies a correction to selected weight columns; the full-support repair applies it broadly.
A separate experiment restricts the edit to write columns in the discovered circuit. These edits have different
support sizes: this is **not** a matched-edit-budget demonstration of superiority.

The main safety evaluation contains **120 harmful and 40 benign prompts per arm**. Harmful compliance is scored
by the frozen HarmBench, Qwen3Guard and Granite Guardian panel, using a threshold above 0.5 and majority-of-three
aggregation. Benign over-refusal uses the Qwen3Guard refusal flag. Capability is measured separately with
**lm-eval GSM8K strict exact match on 500 documents**. A refusal-marker regex is only a diagnostic, never an ASR estimate.

## What the results show

**Read the harm column as an uncalibrated exploratory measurement, not verified safety.**

| Model state | Judged harm rate | Benign over-refusal | GSM8K exact match |
|---|---:|---:|---:|
| Aligned model | 5.8% | 7.5% | 59.0% |
| Abliterated model | 21.7% | 5.0% | 56.0% |
| Sparse repair | 5.0% | 17.5% | 51.8% |
| Full-support repair | 4.2% | 27.5% | 43.8% |

The sparse repair scored **8.0 percentage points higher on GSM8K** than the full-support repair, with a paired
bootstrap 95% interval of **3.8 to 12.2 points**. That supports a task-specific retention finding, not preservation
of general capability. Both repairs also refused more benign requests than the drifted model. The small benign
sample and unvalidated judging limit how broadly this trade-off can be interpreted.

Sources: [task-performance analysis](https://github.com/amugoodbad229/scartissue/blob/master/results/p16_capability_analysis_20261102.json)
and [safety comparisons, denominators and intervals](https://github.com/amugoodbad229/scartissue/blob/master/results/circuit_arm_analysis_20261106.json).

**The circuit-restricted repair was a negative result.** Its judged harm rate was **28.3%**, versus **21.7%**
for the drifted model. Restricting the edit to that discovered circuit did not improve this metric. This does
not establish that no safety-relevant circuit exists; it shows that this discovery-and-edit procedure did not
deliver the proposed repair under the evaluated conditions. The same calibration caveat applies.

## Why we are not claiming a safety success

The judge panel's calibration audit failed. That run also contained a prompt-rendering defect, so its
sensitivity is not a reliable estimate of a correctly configured panel. Agreement among judges is not enough
to establish valid harm measurement. The [calibration appraisal](https://github.com/amugoodbad229/scartissue/blob/master/_research/APPRAISAL-20261106.md)
documents the defect. The blind human audit on this project's own
completions remains pending. Its labels must come from people; neither the annotation app nor another model
can stand in for that evidence.

An additional held-out comparison has been analyzed, but it remains exploratory under the same failed
calibration. The locked held-out **confirmatory comparison is still pending**. The study is also limited to
the evaluated model, seed, prompt sets and capability task. It does not demonstrate general safety, multilingual
robustness or a repair that survives subsequent harmful fine-tuning.

See the [evidence manifest](https://github.com/amugoodbad229/scartissue/blob/master/results/manifest.json)
and [readiness ledger](https://github.com/amugoodbad229/scartissue/blob/master/results/readiness_ledger.json)
for the evidence status and remaining gates.

## What is available

The release includes code, recorded generations and judgments, analyses, and parameter patches with a loader.
It uses CuratorKIT, CircuitKIT and AuditKIT. Only parameters are released—not a hosted drifted checkpoint—and
use remains subject to Cohere's LSAL restrictions.

- [Code and reproduction instructions](https://github.com/amugoodbad229/scartissue)
- [Repair-parameter collection](https://huggingface.co/collections/amugoodbad229/scartissue-repair-parameters-6aedb25709af05c05784a9e0)
- [Evaluation-evidence collection](https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence-6aedb2693a8dd1f844817b46)

An optional [evidence viewer](https://amugoodbad229-scartissue.static.hf.space/) makes the recorded comparisons
easier to inspect; it performs no live inference and does not certify a repair.

**The takeaway:** safety repair should be evaluated as a trade-off, not inferred from refusal language alone.
ScarTissue provides evidence of task-specific costs and a failed circuit-repair attempt, while keeping the
unresolved safety question explicit.
