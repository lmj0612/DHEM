Data preparation (research versions; data paths are set at the top of each file).

transfer_amie_parallel_new.py, generate_npz_1min.py, npy_pipeline.py   AMIE IDL .save files (24 MLAT x 25 MLT, 1 min) -> 32 x 32 npy per minute,
                                        3 channels (mean energy, energy flux, potential), min-max scaled to [-1, 1] (ranges_joint/*.json)
generate_condition_all.py               OMNI 1-min, F10.7 -> Newell coupling, sin/cos clock angle, SYM-H, Pdyn, F10.7, AE, dipole tilt, z-scored
generate_sampling_conds_SMU_SML.py, build_2003_cadence_base48.py   SuperMAG MLT-sector SML/SMU -> 48-d SML[0:24] + SMU[24:48], z-scored
generate_conditions_tsh52_sampling.py   + tilt, SYM-H, SML 30/120-min means -> 52-d
generate_conditions_4cat_sampling.py, generate_conditions_4cat_tsh52x.py   SOPHIE phase lists (Forsyth et al., 2015) -> category per minute
                                        and the category extras (Newell, T1, T2, dSML/dt) -> 55 / 53 / 53 / 54-d files
build_pot56_2003_stratified.py          8 drivers + 48-d SML/SMU -> 56-d DHEM-EP conditions
compute_oval_boundaries.py              auroral-oval boundaries of the training maps (assets/oval_boundaries_training.npz)
make_family_conditions.py               per-category files -> family file for scripts/sample.py (conditions/, examples/*/conditions/)
