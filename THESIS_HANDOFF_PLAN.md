# Thesis Handoff Plan

## Goal
Finish the practical work report by:
- integrating the final comparison figure into `main.tex`,
- writing the comparison-study subsection, discussion, limitations, and conclusion,
- documenting the JAXTOuNN modifications relative to the upstream repository,
- citing the original TOuNN paper and the JAXTOuNN GitHub project,
- keeping the final story consistent with the actual experiment artifacts already collected.

## Key Files
- Report source:
  - `/Users/dominik/Documents/Shared/ai-bachelor/thesis/practical-work-report/main.tex`
- Main experiment bundle:
  - `/Users/dominik/Documents/Shared/ai-bachelor/thesis/practical-work/experiments/11/summary.csv`
  - `/Users/dominik/Documents/Shared/ai-bachelor/thesis/practical-work/experiments/11/comparison_overview.png`
  - `/Users/dominik/Documents/Shared/ai-bachelor/thesis/practical-work/experiments/11/comparison_overview.pdf`
- Figure generator:
  - `/Users/dominik/Documents/Shared/ai-bachelor/thesis/practical-work/experiments/11/generate_comparison_figure.py`
- Main SIREN config:
  - `/Users/dominik/Documents/Shared/ai-bachelor/thesis/practical-work/train_configs/train_simple-Tounn_omega_sweep_l5_h64.yaml`
- Small SIREN config:
  - `/Users/dominik/Documents/Shared/ai-bachelor/thesis/practical-work/train_configs/train_simple-Tounn_omega_sweep_l3_32.yaml`
- Main JAXTOuNN config:
  - `/Users/dominik/Documents/Shared/ai-bachelor/thesis/JAXTOuNN/config.txt`

## Final Comparison Story To Preserve

### Main-capacity comparison
- Main JAXTOuNN and main SIREN are matched at `5 layers / 64 hidden units`.
- Same-resolution comparison is on the `180 x 90` mesh (`60 x 30` physical domain, `scale=3`).
- Refined evaluation comparison is on the `900 x 450` mesh (`scale=15`).
- MMA is only available as the `180 x 90` same-resolution reference in the current figure bundle.

### Small-capacity comparison
- Small JAXTOuNN and small SIREN are matched at `3 layers / 32 hidden units`.
- Small JAXTOuNN refined result is available at `900 x 450`.
- Small SIREN is intentionally shown and evaluated on the **unprojected** field because projection materially changed the visible topology and otherwise the figure would not match the shown image. What we want to show with the small capacity comparison is that it SIREN does not behave well with smaller models unlike the JaxTounn implmentation with that fourier and swish activation function. (Tounn method from 2 years ago, correct siting TBD)

## Current Experiment Facts
These come from `experiments/11/summary.csv` and should be treated as the current source of truth unless the user reruns experiments.

### Same-resolution (`180 x 90`)
- JAXTOuNN main (`5x64`):
  - internal compliance: `68.3019`
  - FEAX compliance: `68.610404`
  - volume fraction: `0.499639`
- SIREN main (`5x64`):
  - FEAX compliance: `68.0147`
  - volume fraction: `0.4977`
- MMA:
  - projected compliance: `69.786010`
  - projected volume fraction: `0.500000`

### Refined evaluation (`900 x 450`)
- JAXTOuNN main (`5x64`):
  - internal compliance: `72.1021`
  - FEAX compliance: `71.105390`
  - volume fraction: `0.499100`
- SIREN main (`5x64`):
  - FEAX compliance: `71.2921`
  - volume fraction: `0.4952`

### Small-capacity refined evaluation (`900 x 450`)
- JAXTOuNN small (`3x32`):
  - internal compliance: `70.2936`
  - FEAX compliance: `71.773066`
  - volume fraction: `0.499784`
- SIREN small (`3x32`, **unprojected**):
  - FEAX compliance: `90.536541`
  - volume fraction: `0.498715`

### Small-capacity same-resolution (`180 x 90`)
- SIREN small (`3x32`, **unprojected**):
  - FEAX compliance: `86.512651`
  - volume fraction: `0.500001`

## Important Interpretation Notes
- Lower compliance is better.
- JAXTOuNN and SIREN are very close at the main `5x64` capacity.
- MMA is not dominating the same-resolution comparison in the current numbers; it is slightly worse than the two neural methods in the stored `180 x 90` bundle.
- But MMA can be generally considered as good using the same resultion. Problem is with MMA is that you cannot upscale the resultion. And training MMA on higher resolution is much more expensive.
- The small-capacity comparison is the clearest separation:
  - JAXTOuNN small stays competitive.
  - SIREN small degrades strongly, especially when evaluated on the raw unprojected field ( the other feax example (ours), was projected, but that doesn't matter since the neural network ALREADY was almost 0 and 1 so projection didn't do anything. We are at fault here but as this is a practical work I would just not mention that the smaller is not projected while the bigger one is.)
- The small SIREN case should not be mixed carelessly with projected results in text or figures.

## What The Stronger Agent Should Write In `main.tex`

### 1. Update the method naming in Section 5.3
In `main.tex`, the comparison section currently still describes TOuNN as if it were the original method:
- see around lines `497-540` in `main.tex`.

This should be updated to:
- `JAXTOuNN` or `a JAX implementation of TOuNN by the original authors`,
- plus an explicit statement that the code used here is not the untouched upstream state, because export/evaluation and solver-related modifications were added for this work.

Suggested framing:
- `MMA`: classical density method.
- `JAXTOuNN`: JAX implementation of TOuNN from the authors’ GitHub project, used here with local modifications for export, solver consistency, and FEAX-based comparison.
- `SIREN (ours)`: proposed method.

### 2. Insert the current final figure
Use:
- `experiments/11/comparison_overview.pdf` for LaTeX if possible.

The figure already shows:
- same-resolution row,
- refined-evaluation row,
- model-size row,
- with layer/hidden-size labels and compliance/volume printed directly.

### 3. Rewrite the claims in Section 5.3 to match the actual evidence
The current draft in `main.tex` assumes expected outcomes that may not match the actual final numbers.

The final text should say:
- Main `5x64` JAXTOuNN and SIREN are very close.
- The refined `900x450` evaluation also remains close.
- The small `3x32` comparison strongly favors JAXTOuNN over SIREN.
- MMA provides the classical density baseline on the training mesh, but the strongest thesis claim here is not universal superiority of one neural field over the other; it is that the methods are competitive at matched main capacity, while JAXTOuNN is more robust in the low-capacity regime.

Do **not** keep any wording that presumes SIREN must outperform JAXTOuNN on refined transfer unless the actual final values support it.

### 4. Write the missing sections
Sections still effectively missing or empty:
- Abstract body still placeholder earlier in the document.
- Discussion.
- Limitations and Conclusion.
- Conclusion.

Recommended content:

#### Discussion
- Compare same-resolution and refined results separately.
- Point out that at `5x64`, both neural methods are competitive and close.
- Point out that at `3x32`, JAXTOuNN is clearly stronger.
- Mention that density projection can substantially alter the visual appearance of some SIREN outputs, which matters for interpretation.

#### Limitations
- JAXTOuNN comparison is based on a locally modified version of the authors’ repository.
- Not all comparisons use the same projection convention.
- Results are on one benchmark family (`cantilever_corner`) and one material/load regime.
- JaxTounn uses different optimization stratagies ( look them up)

#### Conclusion
- SIREN is viable and competitive at matched main capacity.
- JAXTOuNN is more parameter-efficient in the small-model regime.
- The thesis contribution is the direct FEAX-based comparison pipeline and the clarified tradeoff between representation flexibility and low-capacity robustness.

## JAXTOuNN Modifications To Describe
These are the main code changes visible in the current `git diff`:

### Diff summary
Command already checked:
```bash
git -C /Users/dominik/Documents/Shared/ai-bachelor/thesis/JAXTOuNN diff --stat
```

Relevant edited files:
- `FE_Solver.py`
- `Mesher.py`
- `config.txt`
- `export_density_vtu.py`
- `main_TOuNN.py`
- `TOuNN.py`

### Changes that must be mentioned in the report

#### 1. Compliance interpolation cleanup
In `FE_Solver.py`:
- removed the artificial `+0.001` shift in the density-to-stiffness interpolation and its derivative.

Why it matters:
- makes the compliance definition more consistent with the FEAX-side evaluation and with the target SIMP form.

#### 2. Snapshot export support
In `Mesher.py`:
- added `saveFieldSnapshot(...)`.

Why it matters:
- supports non-interactive experiment logging and figure harvesting.

#### 3. JAXTOuNN configuration retuning
In `config.txt`:
- switched to `E=1.0`, `Emin=1e-6`, `plane-stress-compatible` material scale.
- set the main architecture to `5 layers / 64 hidden`.
- enabled Fourier features.
- switched activation to `swish`.
- enabled export and snapshot-oriented workflow.
- fixed optimization settings for the comparison runs (`penal=3`, `epochs=200`, etc.).

Why it matters:
- aligns JAXTOuNN with the same physical scaling and comparison protocol used for SIREN and FEAX.

#### 4. VTU and PNG export support
In `export_density_vtu.py`:
- added PNG export of density fields.
- returned/printed final compliance from the optimization path.

Why it matters:
- made it practical to include JAXTOuNN in the unified experiment bundle and paper figures.

#### 5. Main comparison workflow additions
In `main_TOuNN.py`:
- added PNG export.
- added refined-mesh exported-resolution compliance computation.
- added optional snapshot mode via `disableDisplay`.
- fixed the refined mesh construction so the physical domain stays the same when `exportRes` changes.

Why it matters:
- this is the core bridge that made direct FEAX-side comparison possible.

### One caveat
The `git diff` did not show `TOuNN.py` content in the limited file diff snippet above, but `git diff --stat` shows it changed. A stronger agent should inspect the full file diff before finalizing the report wording, so no code change is omitted.

## Citation Work Needed

### TOuNN paper
Already referenced in the report:
- `Chandrasekhar2020TOuNNTO`

### JAXTOuNN GitHub project
Need a bibliographic entry in `references.bib` for the GitHub repository.

The stronger agent should:
1. identify the exact repository URL and maintainers from the upstream project used by the user,
2. create a proper software/reference entry,
3. cite it in the comparison-study method description.

Suggested wording:
- `We use the JAX implementation of TOuNN released by the original authors and extend it with export and evaluation functionality required for the present comparison study.`

If web access is allowed for that agent, it should verify the exact repository metadata instead of guessing.

## Concrete Edits Needed In `main.tex`

### Comparison section edits
- Replace `TOuNN` with `JAXTOuNN` where the actual implementation being compared is the modified JAX codebase.
- Add one paragraph explaining the local modifications and why they were necessary for fair FEAX-side evaluation.
- Add one paragraph on same-resolution results using the actual numbers from `summary.csv`.
- Add one paragraph on refined evaluation results using the actual numbers from `summary.csv`.
- Add one paragraph on the `3x32` model-size comparison.

### Figure insertion
- Insert `experiments/11/comparison_overview.pdf` in Section 5.3.
- Caption should mention:
  - row 1: same-resolution comparison,
  - row 2: refined evaluation,
  - row 3: smaller matched model size,
  - all methods use the same `cantilever_corner` setup,
  - small SIREN row uses the unprojected field.

## Recommended Next Actions For The Stronger Agent
1. Open `main.tex` and replace the drafted Section 5.3 text with the actual final comparison narrative.
2. Add the final figure and likely a small numeric table derived from `experiments/11/summary.csv`.
3. Update `references.bib` with the JAXTOuNN GitHub citation.
4. Add a concise paragraph summarizing the JAXTOuNN code modifications.
5. Write `Discussion`, `Limitations`, and `Conclusion`.
6. Keep the prose faithful to the current experiment bundle; do not claim trends that are not supported by `summary.csv`.

## Do Not Re-derive Unless Necessary
Most expensive comparison work is already done. Prefer using:
- `experiments/11/summary.csv`
- `experiments/11/comparison_overview.pdf`

instead of rerunning experiments, unless the user explicitly wants the numbers changed.
