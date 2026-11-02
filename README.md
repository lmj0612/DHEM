# DHEM

Diffusion High-latitude Electrodynamic Models: conditional diffusion models that generate 1-min maps (32 x 32, MLAT x MLT)
of the high-latitude electric potential (DHEM-EP) and of the particle energy flux and average energy (DHEM-AP) from
solar wind and geomagnetic drivers, trained on AMIE.

## Contents

| | |
|---|---|
| `dhem/` | model code (PyTorch), based on [improved-diffusion](https://github.com/openai/improved-diffusion) |
| `configs/` | model cards of the nine models |
| `weights/` | DHEM-EP 240k; DHEM-AP particle energy flux cat1-4 470k / 430k / 420k / 510k; DHEM-AP average energy cat1-4 375k / 385k / 405k / 390k; the `.pt` files are in the Zenodo archive |
| `assets/` | AMIE grid, z-score statistics of the conditioning inputs, normalisation ranges, oval mask of the training maps |
| `scripts/sample.py` | generate maps |
| `scripts/evaluate.py` | HP / CPCP statistics |
| `scripts/run_example_day.sh`, `scripts/run_example_event.sh` | examples: 2003-11-20 and 6-7 April 2000 |
| `scripts/training/` | training scripts `image_train_*.py` |
| `scripts/data_prep/` | AMIE to npy, conditioning vectors from OMNI / SuperMAG / SOPHIE |
| `scripts/figures/` | python scripts of the figures; data in [DHEM_data_for_figures](https://github.com/lmj0612/DHEM_data_for_figures) |
| `conditions/` | condition vectors of the 2003 test dataset (one minute every 10 min, 43,777 min) and of Figures 3-5 |
| `examples/` | conditions and outputs for 2003-11-20 and 6-7 April 2000 |

## Install

```bash
conda env create -f environment.yml -n dhem      # or: pip install -r requirements.txt
```

## Run

```bash
python scripts/sample.py --family dhem_ap --conditions examples/2003-11-20/conditions/conds_dhem_ap_2003-11-20.npz --out out/ap.npz
python scripts/evaluate.py --samples out/ap.npz --paper examples/2003-11-20/paper_outputs/dhem_ap_flux_maps_2003-11-20.npz
```

- `--family dhem_ap | dhem_ape | dhem_ep`, or `--config configs/<model>.json` for one model.
- `--mode timeseries` (default: AR(1) noise, 5-min driver average, blend at category switches; Figures 6-8) or `--mode independent` (Figures 2-5).
- Output: maps in physical units (keV | mW/m² | kV) and the raw model output, with timestamps and run metadata; HP / CPCP in `dhem/metrics.py`.

## License

MIT (code), CC BY 4.0 (weights). Citation: `CITATION.cff`.
