Training scripts: image_train_flux.py (particle energy flux, channel 1), image_train_energy.py (average energy, channel 0),
image_train_pot.py (electric potential, channel 2); options are read from TRAIN_* environment variables. Common settings:
lr 1e-4, 1000 linear diffusion steps, hybrid (learn_sigma) loss, EMA 0.9999, UNet 128 channels, 2 residual blocks, attention
at 16 and 8, 2 GPUs (global batch 256 potential, 512 precipitation).

DHEM-EP                                    56-d, C_spatial=2 to 210k -> MLAT channel added -> 240k
DHEM-AP particle energy flux, backbone     48-d model 340k -> 52-d (+Tilt, SYM-H, SML 30/120-min means) + MLAT -> 390k
DHEM-AP particle energy flux, cat1-4       per-category fine-tune, uniform loss: 440k / 410k / 410k / 485k;
                                           then cross-attention (TRAIN_CROSS_ATTN=2, TRAIN_XATTN_HEADS=4) + Eq. 4 oval loss
                                           (TRAIN_OVAL_*: 2 / 0.5 / taper 3 px, HP gate 10-20 GW, mask assets/oval_boundaries_training.npz): 470k / 430k / 420k / 510k
DHEM-AP average energy, backbone           same recipe on the average-energy channel, 400k
DHEM-AP average energy, cat1-4             from 350k: 360k / 380k / 385k / 365k; then as above without the HP gate: 375k / 385k / 405k / 390k

Checkpoints of the final stage were chosen on the 2003 validation dataset (odd 10-day blocks, scripts/split2003.py).
assets/oval_boundaries_training.npz: equatorward (eb) and poleward (pb) row of the k-means auroral oval in each MLT column of every
training map (255 = none), from scripts/data_prep/compute_oval_boundaries.py.
