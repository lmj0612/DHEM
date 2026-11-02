"""
Train a diffusion model on images.
"""
#%%

import argparse

# ----- mj_20250804 -----
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ----- mj_20250804 -----



from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

import os
import numpy as np  # mj_20260225

from mj_spatial_basis import *


# python -c "import improved_diffusion; print(improved_diffusion.__file__)"


#%%
def main():



    # args = create_argparser().parse_args()


    # mpiexec -n 8 python image_train.py
    # CUDA_VISIBLE_DEVICES=3,4,5 mpiexec -n 3 python image_train.py

    # --- env-var overrides: set by SLURM scripts to run parallel experiments  # mj_20260225
    # TRAIN_C_SPATIAL=4      -> MLT-aligned injection (4 spatial channels)
    # TRAIN_C_SPATIAL=2      -> flat 48-dim baseline  (2 spatial channels, no aligned injection)
    # TRAIN_CIRC_CONV=1      -> use CircMLTEncoder instead of flat MLP for label_emb
    # TRAIN_HP_WEIGHT=1      -> enable HP-based sample weighting (needs hp_weights.npz)
    # TRAIN_AMP_TOPK=1       -> enable pixel_amp_weight + topk loss
    # TRAIN_RESUME_CKPT=path -> resume from checkpoint (overrides resume_checkpoint)
    _C_spatial        = int(os.environ.get("TRAIN_C_SPATIAL",  "4"))   # mj_20260225
    _use_circ_conv    = int(os.environ.get("TRAIN_CIRC_CONV",  "0"))   # mj_20260225
    _use_hp_weight    = int(os.environ.get("TRAIN_HP_WEIGHT",  "0"))   # mj_20260225
    _use_amp_topk     = int(os.environ.get("TRAIN_AMP_TOPK",   "0"))   # mj_20260225
    _resume_ckpt_env  = os.environ.get("TRAIN_RESUME_CKPT",    "")    # mj_20260225
    _use_scalar5      = int(os.environ.get("TRAIN_SCALAR5",    "0"))   # mj_20260301
    _use_oval_weight  = int(os.environ.get("TRAIN_OVAL_WEIGHT", "0"))   # mj_20260302
    # mj_20260927: oval-weighting parameters (paper Sect. 3, Eq. 4: 2 inside the k-means oval, 0.5 background, 3-px cosine
    # taper; the old code default w_bndry=3.0 adds an extra EB/PB emphasis that the paper does NOT describe -> the 2026-09-27
    # fine-tunes set TRAIN_OVAL_W_BNDRY=2.0). TRAIN_OVAL_HP_LO/HI [GW]: HP gate, weighting off below LO, full above HI
    # (0 = no gate = 2026-03 behaviour).
    _oval_w_oval   = float(os.environ.get("TRAIN_OVAL_W_OVAL",  "2.0"))  # mj_20260927
    _oval_w_bndry  = float(os.environ.get("TRAIN_OVAL_W_BNDRY", "3.0"))  # mj_20260927
    _oval_w_floor  = float(os.environ.get("TRAIN_OVAL_W_FLOOR", "0.5"))  # mj_20260927
    _oval_taper    = int(os.environ.get("TRAIN_OVAL_TAPER",     "3"))    # mj_20260927
    _oval_hp_lo    = float(os.environ.get("TRAIN_OVAL_HP_LO",   "0"))    # mj_20260927
    _oval_hp_hi    = float(os.environ.get("TRAIN_OVAL_HP_HI",   "0"))    # mj_20260927
    _use_t1t2         = int(os.environ.get("TRAIN_T1T2",       "0"))   # mj_20260305
    _4cat_id          = int(os.environ.get("TRAIN_4CAT",        "0"))   # mj_20260311
    _use_cross_attn   = int(os.environ.get("TRAIN_CROSS_ATTN",  "0"))   # mj_20260312
    # mj_20260901: TRAIN_EXCLUDE_YEARS="1999" (comma-separated) drops those years
    # from training (filename prefix filter); empty/unset = old behavior
    _exclude_years    = [y.strip() for y in os.environ.get("TRAIN_EXCLUDE_YEARS", "").split(",") if y.strip()]  # mj_20260901

    # mj_20260311: TRAIN_4CAT={1,2,3,4} overrides cond_dim, cond_file, data_dir for Forsyth-phase training
    _4CAT_CONFIG = {  # mj_20260311
        1: dict(cond_dim=51, cond_file="conditions_cat1_growth_zscore.npz",       cat_name="cat1_growth"),  # mj_20260311
        2: dict(cond_dim=49, cond_file="conditions_cat2_compound_zscore.npz",     cat_name="cat2_compound"),  # mj_20260311
        3: dict(cond_dim=49, cond_file="conditions_cat3_isolated_zscore.npz",     cat_name="cat3_isolated"),  # mj_20260311
        4: dict(cond_dim=50, cond_file="conditions_cat4_convection_zscore.npz",   cat_name="cat4_convection"),  # mj_20260311
        # mj_20260901: TRAIN_4CAT=5 = unified single model on ALL data:
        # 48d SML/SMU + Newell + dSML_filt/dt + T1 + T2 + Forsyth one-hot(4) = 56-dim.
        # Shares the backbone across categories; category enters only as conditioning.
        5: dict(cond_dim=56, cond_file="conditions_joint_9_zscore_4cat_unified.npz", cat_name="4cat_unified",  # mj_20260901
                data_dir='/home/lim2/data/amie_npy_joint_from105/training_npy_joint/'),  # mj_20260901
    }  # mj_20260311

    # mj_20260301: TRAIN_SCALAR5=1 overrides C_spatial, cond_dim, cond file, and column selection
    if _4cat_id in _4CAT_CONFIG:  # mj_20260311
        _cfg          = _4CAT_CONFIG[_4cat_id]  # mj_20260311
        _cond_dim     = int(os.environ.get("TRAIN_COND_DIM", str(_cfg['cond_dim'])))  # mj_20260322 allow override
        _cond_file    = os.environ.get("TRAIN_COND_FILE", _cfg['cond_file'])  # mj_20260322 allow override
        # mj_20260902: TRAIN_COND_DIM smaller than the cat file's width -> use the
        # first cond_dim columns (cat files are SML[0:24]+SMU[24:48]+extras, so
        # TRAIN_COND_DIM=48 gives a pure-partition model with no cat-specific extras)
        _cols_to_use  = list(range(_cond_dim)) if _cond_dim < _cfg['cond_dim'] else None  # mj_20260311, mj_20260902
        _data_dir_default = _cfg.get('data_dir', f'/home/lim2/data/amie_npy_joint_from105/training_{_cfg["cat_name"]}/')  # mj_20260311, mj_20260901 allow per-config data_dir override
    elif _use_scalar5:  # mj_20260301
        _C_spatial    = 2   # sinMLT, cosMLT only (no SML/SMU spatial channels)  # mj_20260301
        _cond_dim     = 5   # [Newell, sin_clock, cos_clock, Pdyn, tilt]  # mj_20260301
        _cond_file    = "conditions_joint_9_zscore.npz"  # 8-col file  # mj_20260301
        _cols_to_use  = [0, 1, 2, 4, 7]  # Newell, sin_clock, cos_clock, Pdyn, tilt  # mj_20260301
    elif _use_t1t2:  # mj_20260305
        _cond_dim     = 50  # 48-dim SMU/SML + 2 substorm phase (T1, T2)  # mj_20260305
        _cond_file    = "conditions_joint_9_zscore_SMU_SML_T1T2.npz"  # mj_20260305
        _cols_to_use  = None  # use all 50 columns  # mj_20260305
    else:  # mj_20260301
        _cond_dim     = 48  # mj_20260301
        _cond_file    = None  # use default (conditions_joint_9_zscore_SMU_SML.npz)  # mj_20260301
        _cols_to_use  = None  # use default (all 48 columns)  # mj_20260301
    if _4cat_id not in _4CAT_CONFIG:  # mj_20260311
        _data_dir_default = '/home/lim2/data/amie_npy_joint_from105/training_npy_joint/'  # mj_20260311

    args = argparse.Namespace(
        data_dir=os.environ.get("TRAIN_DATA_DIR", _data_dir_default),  # mj_20260311
        # data_dir='/home/lim2/data/amie_npy/training_npy_logflux/',
        use_structured_cond=True, # mj_20250530
        C_spatial=_C_spatial, # mj_20260225: 4=MLT-aligned (sinMLT,cosMLT,SML,SMU) | 2=flat baseline
        cond_dim=_cond_dim, # mj_20260301: 48 for SML/SMU, 5 for scalar5
        use_circ_conv_cond=bool(_use_circ_conv), # mj_20260225: circular conv over 24 MLT sectors
        use_cross_attn=int(_use_cross_attn), # mj_20260312: cross-attention to MLT condition tokens; mj_20260927: TRAIN_CROSS_ATTN=2 -> MLTCrossAttention v2
        xattn_heads=int(os.environ.get("TRAIN_XATTN_HEADS", "0")), # mj_20261027 ENERGY: fixed heads for MLTCrossAttention (4 in the final ovxM round), synced with flux mj_20261026
        sector_drop=float(os.environ.get("TRAIN_SECTOR_DROP", "0")), # mj_20261027 ENERGY: kept 0 (flux v3 sector-dropout round was negative)
        spatial_basis=spatial_basis[:3] if int(os.environ.get('TRAIN_MLAT', '0')) else spatial_basis[:2], # mj_20250911 (geometric channels; SML/SMU appended at dataset level when C_spatial=4), mj_20260904 TRAIN_MLAT=1 adds the normalised-MLAT channel (use with TRAIN_C_SPATIAL=3)
        # C_spatial=0, # mj_20250911
        # spatial_basis=None, # mj_20250911
        image_size=32,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=1,
        attention_resolutions = "16,8",
        learn_sigma=True, # mj
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        # diffusion_steps=2000,
        noise_schedule="linear",
        # diffusion_steps=250,
        # noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False, # mj
        rescale_learned_sigmas=False, # mj
        use_checkpoint = False,
        use_scale_shift_norm=True,
        # mj start !!!
        schedule_sampler="uniform",
        # schedule_sampler="loss-second-moment",
        lr=1e-4,
        # lr=2e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        # batch_size=128,  # mj
        batch_size=int(os.environ.get('TRAIN_BATCH_SIZE', '256')),  # mj; mj_20260912 env override (1-GPU runs use 512 to keep the global batch at 512)
        microbatch=int(os.environ.get('TRAIN_MICROBATCH', '-1')),  # mj_20260912 env override: 256 = gradient accumulation, same memory as 2x256
        ema_rate="0.9999",
        log_interval=10,
        save_interval=int(os.environ.get('TRAIN_SAVE_INTERVAL', '10000')),  # mj_20260912 env override (fine-tunes: 5k)
        # resume_checkpoint="/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_default_NewellCoupling/model100000.pt",
        resume_checkpoint=_resume_ckpt_env,  # mj_20260225: set via TRAIN_RESUME_CKPT env var (or "" for fresh start)
        use_fp16=False, 
        fp16_scale_growth=1e-3)   # mj_20250527



    # mj_20260927: constants for the on-the-fly HP of the training target (oval HP gate). Same definitions as
    # scripts/cat_candidate_metrics.py: train80 range of channel 1 (energy flux) and AMIE cell areas on the
    # 32-row grid resampled linearly from the 24 AMIE latitudes 90..44 deg (b20000101n.save 'lats').
    # mj_20260927 ENERGY repo note: the oval HP gate below is a FLUX-model device (HP from channel 1); it is inactive unless
    # TRAIN_OVAL_WEIGHT=1, which the energy experiments do not use. Kept verbatim so the two repos stay diff-able.
    _hp_vmin, _hp_vmax = -1.9955321550369263, 2.4705755710601807  # mj_20260927 fallback = ranges_joint/amie_range_train80.json
    try:  # mj_20260927
        import json as _json  # mj_20260927
        with open('/home/lim2/data/amie_npy_joint_from105/ranges_joint/amie_range_train80.json') as _f:  # mj_20260927
            _rt = _json.load(_f); _hp_vmin, _hp_vmax = float(_rt['vmins'][1]), float(_rt['vmaxs'][1])  # mj_20260927
    except Exception as _e:  # mj_20260927
        print(f"[oval HP gate] using hard-coded train80 range ({_e})")  # mj_20260927
    _li = np.linspace(90.0, 43.999996185302734, 32)  # [H] MLAT of each image row (row 0 = pole)  # mj_20260927
    _cr = np.deg2rad(90.0 - _li); _dc = abs(np.deg2rad(_li[1] - _li[0]))  # mj_20260927
    _hp_area = (np.sin(_cr) * _dc * (2 * np.pi / 32) * 6.371e6 ** 2)[:, None] * np.ones((1, 32))  # [H, W] m^2  # mj_20260927

    # --------------minjing start 2025/09/03--------------
    # mj_20260302: oval_weight disables pixel_amp_weight and topk (cleaner comparison)
    _effective_amp_topk = _use_amp_topk and not _use_oval_weight  # mj_20260302

    args.loss_cfg = dict(
        # (1) Pixel-wise amplitude weighting (by x0 target)
        # mj_20260225: TRAIN_AMP_TOPK=1 enables this (amp_thresh in normalized [-1,1] space)
        # mj_20260302: disabled when TRAIN_OVAL_WEIGHT=1 (oval weighting replaces this)
        pixel_amp_weight=bool(_effective_amp_topk),  # mj_20260302
        # Training data is log10(flux) normalized to [-1,1]:
        # amp_thresh=0.3 roughly targets upper-quartile high-flux (storm aurora) pixels
        amp_thresh=0.3 if _effective_amp_topk else 5.0,  # mj_20260302
        amp_scale=2.0,
        amp_gamma=1.0,
        amp_base=1.0,

        # (2) Sample-wise weighting by AE (rebalance high-AE events)
        sample_ae_weight=False,
        ae_pivot=500.0,
        ae_slope=100.0,
        ae_max_boost=3.0,
        ae_gamma=1.0,

        # (3) Top-k x0 loss (explicitly push peaks)
        # mj_20260225: TRAIN_AMP_TOPK=1 enables topk (top 15% aurora pixels)
        # mj_20260302: disabled when TRAIN_OVAL_WEIGHT=1
        topk_x0_frac=0.15 if _effective_amp_topk else -0.30,   # positive=enabled  # mj_20260302
        if_normalize_topk_error=True,
        topk_norm_min=0.20,
        topk_norm_max=0.55,
        topk_by="x0",
        topk_weight=0.05 if _effective_amp_topk else 0.0002,   # mj_20260302
        base_weight=1.0,

        # (1b) K-means oval boundary loss weighting (mj_20260302)
        # Replaces pixel_amp_weight with spatially-structured weights based on
        # pre-computed auroral oval boundaries (EB/PB from k-means clustering)
        oval_weight=bool(_use_oval_weight),  # mj_20260302
        oval_w_oval=_oval_w_oval,    # weight inside oval  # mj_20260302, mj_20260927 env
        oval_w_bndry=_oval_w_bndry,  # weight at EB/PB boundary pixels  # mj_20260302, mj_20260927 env
        oval_w_floor=_oval_w_floor,  # weight for background  # mj_20260302, mj_20260927 env
        oval_taper_width=_oval_taper, # cosine taper width (pixels) from boundary into background  # mj_20260302, mj_20260927 env
        # mj_20260927: HP gate for the oval weighting (see _hp_from_x0 in gaussian_diffusion.py)
        oval_hp_gate_lo=_oval_hp_lo,  # mj_20260927
        oval_hp_gate_hi=_oval_hp_hi,  # mj_20260927
        oval_hp_vmin=_hp_vmin, oval_hp_vmax=_hp_vmax, oval_hp_eps=1e-2,  # train80 log10 range of channel 1  # mj_20260927
        oval_hp_area=_hp_area,  # [32, 32] AMIE cell areas [m^2]  # mj_20260927

        # (4) supervised loss
        sup_weight=-0.005,
        sup_type="l1",

        loss_cfg = {
            # (5) smoothing loss
            "smooth_weight": -0.005,
            "smooth_type": "laplacian",
            "smooth_loss_type": "l2",
            "smooth_time_weight": "sqrt_snr",

            # (6) gradient matching loss
            "grad_match_weight": -0.01,
            "grad_match_type": "l1",
            "grad_time_weight": "sqrt_snr",

            # (7) multiscale loss
            "multiscale_weight": -0.01,
            "ms_lambda_0": 1.0,
            "ms_lambda_lp": 0.5,
            "ms_lambda_hp": 0.1,
            "ms_kernel_size": 5,
            "ms_sigma": 1.5,
            "ms_time_weight": "sqrt_snr",

            # (8) Hemispheric Power (HP) based sample weighting
            # mj_20260225: TRAIN_HP_WEIGHT=1 enables this (requires hp_weights.npz)
            "hp_sample_weight": bool(_use_hp_weight),   # mj_20260225
            "hp_weight_scale": 1.0,
            "hp_weight_clip_min": 0.1,
            "hp_weight_clip_max": 10.0,
        }
    )
    # --------------minjing end 2025/09/03--------------



    defaults = model_and_diffusion_defaults()
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)


    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_GradientLoss/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_MultiscaleLoss/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_Steps250Cosine_LossSecondMoment/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_HPSampleWeight/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_default_steps2000_lr2e-4_bs256_LossSecondMoment/"
    # mj_20260225: TRAIN_LOGDIR env var lets SLURM scripts set the log dir per-experiment;
    # falls back to the aligned experiment dir if not set.
    if _use_scalar5:  # mj_20260301
        _default_logdir = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_scalar5/"  # mj_20260301
    else:  # mj_20260301
        _default_logdir = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_MLT_SML_SMU_aligned/"
    os.environ["OPENAI_LOGDIR"] = os.environ.get("TRAIN_LOGDIR", _default_logdir)  # mj_20260225
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_default_AE/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_energy/logs/train_default_NewellCoupling/"

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, defaults.keys())
    )




    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # Check if HP-based sample weighting is enabled (mj_20260205)
    use_hp_weights = args.loss_cfg.get("loss_cfg", {}).get("hp_sample_weight", False)
    # MLT grid only needed when C_spatial=4 (aligned injection); None disables injection  # mj_20260225
    mlt_grid = np.linspace(0.0, 24.0, args.image_size, endpoint=False) if args.C_spatial == 4 else None  # mj_20260904: SML/SMU spatial channels only for the 4-channel aligned design (C_spatial=3 = sin,cos,MLAT)
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        use_structured_cond=args.use_structured_cond, # mj_20250530
        class_cond=args.class_cond,
        spatial_basis=args.spatial_basis, # mj_20250911
        use_hp_weights=use_hp_weights, # mj_20260205
        mlt_grid=mlt_grid, # mj_20260225
        cond_file=_cond_file, # mj_20260301
        cols_to_use=_cols_to_use, # mj_20260301
        use_oval_boundaries=bool(_use_oval_weight), # mj_20260302
        exclude_years=_exclude_years, # mj_20260901
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        loss_cfg=args.loss_cfg, # minjing 2025/09/03
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=int(os.environ.get('TRAIN_SAVE_INTERVAL', '10000')),  # mj_20260912 env override (fine-tunes: 5k)
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
