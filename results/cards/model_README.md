---
library_name: pytorch
license: other
license_name: lsal-1.2-research-use-parameters-only
tags:
  - safety
  - abliteration
  - weight-space-editing
  - cohere
  - parameter-delta
---

# ScarTissue — Safety Drift and Repair Artifacts

**Research patches for studying the trade-off between restored refusal and useful behavior.**

ScarTissue investigates whether weight-space edits can repair synthetic safety drift in
[Cohere Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) without retraining.
This repository contains the drift and repair parameters used in the study—not a standalone model,
a LoRA adapter or a safety-certified checkpoint.

[Project and findings](https://github.com/amugoodbad229/scartissue) ·
[Reproduction guide](https://github.com/amugoodbad229/scartissue/blob/master/REPRODUCE.md) ·
[Evaluation data](https://huggingface.co/collections/amugoodbad229/scartissue-evaluation-evidence-6aedb2693a8dd1f844817b46)

## What is included

| File | Purpose |
|---|---|
| `abliteration_tiny-aya-global_layer15.npz` | Parameters for the synthetic drift intervention. |
| `repair_harm2_a4.npz` | Sparse repair with the smallest published support. |
| `repair_harm4_a4.npz` | Intermediate sparse repair. |
| `repair_harm10_a4.npz` | Broader sparse repair. |
| `repair_global_a4.npz` | Full-support repair comparator. |
| `load_patch.py` | The original patch-loading helper. |
| `manifest.json` | Technical provenance and artifact verification details. |

The patch files store factorized weight updates. They do not include the base model weights.
The older patch-only dataset repositories duplicate these parameters and are retained for existing links;
this model repository is the main download location.

## What the study found

The sparse repair retained more GSM8K performance and refused fewer benign requests than the
full-support repair. Both repairs refused more benign requests than the drifted model.
A separate circuit-restricted intervention was a negative result, not a demonstrated localization
of safety. Its analysis is in the project repository; no circuit patch is included in this download.

**Restoring refusal is not proof of restoring safety.** Judge calibration failed, including a
prompt-rendering defect in the calibration procedure. Harm estimates remain exploratory.
The blind human audit and locked held-out confirmatory comparison are pending.

These findings are limited to the evaluated model, settings and capability task.
They do not establish general capability preservation or production safety.

## Reproducing the study

Start with the [reproduction guide](https://github.com/amugoodbad229/scartissue/blob/master/REPRODUCE.md),
which describes the base model, dependencies and intervention sequence. Apply each repair to the
intended drifted starting state, not directly to an arbitrary model.

Keep clean saved weights between comparisons. Subtracting a patch is not a reliable way to restore
the original floating-point weights. The bundled helper is the original implementation; it does not
provide automatic base-revision validation or transactional rollback. The project repository contains
the [validated loader](https://github.com/amugoodbad229/scartissue/blob/master/artifacts/load_patch.py)
and its tests.

Exact identifiers and verification records remain in the technical manifests rather than the
reader-facing summary.

## Research use

Use is subject to the [base model's license](https://huggingface.co/CohereLabs/tiny-aya-global).
The project follows its LSAL research-use restrictions: no product hosting and no publication or
deployment of a drifted checkpoint. Publishing parameters does not waive the base model's terms.

Recorded outputs can contain harmful or otherwise sensitive material. These artifacts are for
controlled research and evaluation, not deployment.
