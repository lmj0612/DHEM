6-7 April 2000 storm (Figures 6-8): 2,772 consecutive minutes for DHEM-AP, 2,770 for DHEM-EP.

conditions/conds_dhem_ap_2000-04-06.npz        family conditions for DHEM-AP (particle energy flux and average energy)
conditions/conds_dhem_ep_2000-04-06.npz        DHEM-EP conditions
paper_outputs/dhem_ap_flux_maps_2000-04-06.npz  DHEM-AP maps of Figures 7 and 8c-d (time-series mode, one realization, normalised)
paper_outputs/dhem_ep_pot_maps_2000-04-06.npz   DHEM-EP maps of Figures 6 and 8a-b

bash scripts/run_example_event.sh out/event/ cuda              regenerates both series with the seeds of the paper and compares them with the paper maps
bash scripts/run_example_event.sh out/event/ cuda independent  also DHEM-AP with independent noise
