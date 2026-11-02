2003-11-20 (super storm), 132 minutes of the 2003 test dataset at 10-min cadence.

conditions/conds_dhem_ap_2003-11-20.npz   family conditions for DHEM-AP (particle energy flux and average energy)
conditions/conds_pot56_2003-11-20.npz     DHEM-EP conditions
conditions/conds_cat{k}_2003-11-20.npz    the minutes of category k alone (22 / 77 / 12 / 21), for one category model with --config
paper_outputs/                            the paper's maps of these minutes (normalised)

bash scripts/run_example_day.sh out/day/ cuda   (~10 min on one GPU)

The AMIE maps are not part of this archive. With the npy files of these minutes in amie_npy/ (shape (3, 32, 32), scaled with
assets/amie_range_test20.json) run_example_day.sh also evaluates against AMIE.
