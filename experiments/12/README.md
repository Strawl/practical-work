# Experiment 12: 50-Iteration Comparison Scaffold

This directory collects evidence for a follow-up comparison figure focused on a baseline run that already converges within 50 iterations.

## Baseline reference

- Run: `outputs/06-23_17-56-03_baseline_50_iterations`
- Source: `history.csv`
- Iteration 50 objective: `69.9603830684385`
- Iteration 50 volume fraction: `0.49998229463650895`
- Same-resolution VTU evaluation
  - VTU: `outputs/06-23_17-56-03_baseline_50_iterations/final.vtu`
  - Image copied to: `mma_50_scale3_projected.png`
  - Compliance: `70.411946`
  - Volume fraction: `0.499982`
- Refined upscaled VTU evaluation
  - Upscale command:
    `uv run python -m topopt.upscale_vtu outputs/06-23_17-56-03_baseline_50_iterations/baseline_final_with_without_filter.npz --array-name rho_filtered --scale-factor 5`
  - VTU: `outputs/06-23_17-56-03_baseline_50_iterations/baseline_final_with_without_filter_upscaled_x5.vtu`
  - Image copied to: `mma_50_scale15_projected.png`
  - Compliance: `71.536154`
  - Volume fraction: `0.500796`

## JAXTOuNN candidates recorded so far

1. Same-resolution candidate
   - Image: `tounn_density_big_paper_fourier64_match_low_res.png`
   - VTU: `../JAXTOuNN/results/tounn_density_big_paper_fourier64_match_low_res.vtu`
   - Compliance: `69.725568`
   - Volume fraction: `0.500303`
   - Evaluation mesh: `180x90`
   - Load matching: enabled with `point_load=1.0`, `traction=3`
   - Config provenance: `../JAXTOuNN/config.txt`
   - Important: this config uses `numEpochs = 50` and `res = 1`

2. Refined-evaluation candidate
   - Image: `tounn_density_big_paper_fourier64_match.png`
   - VTU: `../JAXTOuNN/results/tounn_density_big_paper_fourier64_match.vtu`
   - Compliance: `72.757569`
   - Volume fraction: `0.500811`
   - Evaluation mesh: `900x450`
   - Load matching: enabled with `point_load=1.0`, `traction=15`
   - Config provenance: intentionally not tied to `config_good.txt`
   - Important: treat this as evidence from the exported VTU/PNG pair only until a cleaner provenance note is available

## Next step

## SIREN representative recorded

- Run: `outputs/06-23_19-04-30_neural_field_train_simple-Tounn_best`
- Chosen model: `model_3` (user-facing "model 4")
- Training iterations: `50`
- Architecture: `SIREN num_hidden_layers=5 num_hidden_units=64 omega=20.0`
- Same-resolution evaluation
  - Image: `siren_model_4_scale3.png`
  - Compliance: `71.3401`
  - Volume fraction: `0.4819`
  - Source eval copy: `experiments/12/siren_eval_06-23_19-04-30_best/model_evaluation.csv`
- Refined evaluation
  - Image: `siren_model_4_scale15.png`
  - Compliance: `74.4075`
  - Volume fraction: `0.4812`
  - Source eval file: `outputs/06-23_19-04-30_neural_field_train_simple-Tounn_best/model_evaluation.csv`

## Next step

Generate the 50-iteration comparison figure from this directory.
