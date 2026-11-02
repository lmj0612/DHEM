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

from mj_spatial_basis import *


# python -c "import improved_diffusion; print(improved_diffusion.__file__)"


#%%
def main():



    # args = create_argparser().parse_args()


    # mpiexec -n 8 python image_train.py
    # CUDA_VISIBLE_DEVICES=3,4,5 mpiexec -n 3 python image_train.py

    _data_dir_default = os.environ.get("TRAIN_DATA_DIR",
                                       '/home/lim2/data/amie_npy_joint/training_npy_joint/')
    args = argparse.Namespace(
        data_dir=_data_dir_default,
        # data_dir='/home/lim2/data/amie_npy/training_npy_logflux/',
        use_structured_cond=True, # mj_20250530
        C_spatial=int(os.environ.get('TRAIN_C_SPATIAL', '2')), # mj_20250911, mj_20260904 env override (3 = +MLAT)
        spatial_basis=spatial_basis[:3] if int(os.environ.get('TRAIN_MLAT', '0')) else spatial_basis[:2], # mj_20250911, mj_20260904 TRAIN_MLAT=1 adds normalised-MLAT channel
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
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False, # mj
        rescale_learned_sigmas=False, # mj
        use_checkpoint = False,
        use_scale_shift_norm=True,
        # mj start !!!
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        # batch_size=64,  # mj
        # batch_size=128,  # 2026/01/21 minjing
        batch_size=256,  # 2026/02/20 minjing
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=int(os.environ.get("TRAIN_SAVE_INTERVAL", "10000")),  # mj_20260927 env override (fine-tunes)
        resume_checkpoint="",
        use_fp16=False, 
        fp16_scale_growth=1e-3,   # mj_20250527
        # cfg_dropout_prob=0.1,     # CFG: 10% unconditional dropout for classifier-free guidance
        cfg_dropout_prob=float(os.environ.get("TRAIN_CFG_DROP", "0")),     # CFG: 10% unconditional dropout for classifier-free guidance; mj_20260927 env TRAIN_CFG_DROP (per-sample, cond only)
    )



    # --------------minjing start 2025/09/03--------------
    args.loss_cfg = dict(
        # (1) Pixel-wise amplitude weighting (by x0 target)
        pixel_amp_weight=False,
        # Pick amp_thresh in YOUR x0 units:
        # - if training directly in mW/m^2: use ~5.0
        # - if using log10(flux): use ~math.log10(5) ≈ 0.699
        # - if normalized to [-1,1], set the equivalent threshold in that space
        amp_thresh=5.0,
        amp_scale=3.0,
        amp_gamma=2.0,
        amp_base=1.0,

        # (2) Sample-wise weighting by AE (rebalance high-AE events)
        sample_ae_weight=False,
        ae_pivot=500.0,     # start boosting beyond ~500 nT
        ae_slope=100.0,     # smoothness of transition
        ae_max_boost=3.0,   # cap on multiplier
        ae_gamma=1.0,

        # (3) Top-k x0 loss (explicitly push peaks)
        topk_x0_frac=-0.30,  # top 15% pixels per-sample
        if_normalize_topk_error=True,
        topk_norm_min=0.20,
        topk_norm_max=0.55,
        topk_by="x0",       # or "err"
        # topk_weight=0.2,
        topk_weight=0.0002,
        base_weight=1.0,

        # (4) supervised loss
        # sup_weight=-0.005,
        sup_weight=-0.01,
        sup_type="l1",
        # sup_type="smooth_l1",


        loss_cfg = {
            # (5) smoothing loss (regularization on pred_x0)
            "smooth_weight": -0.005,           # Weight for smoothing loss (0 = disabled)
            "smooth_type": "laplacian",     # "laplacian" (2nd order) or "gradient" (1st order)
            "smooth_loss_type": "l2",       # "l1" or "l2"
            "smooth_time_weight": "sqrt_snr",  # Time-dependent weight: "none", "sqrt_snr", "snr", "linear"
            
            # (6) gradient matching loss (supervised: compares gradients of pred_x0 vs x_start)
            "grad_match_weight": -0.01,       # Weight for gradient matching loss (0 = disabled)
            "grad_match_type": "l1",         # "l1" or "l2"
            "grad_time_weight": "sqrt_snr",  # Time-dependent weight: "none", "sqrt_snr", "snr", "linear"
            
            # (7) multiscale loss: L = λ₀‖x̂₀-x₀‖₁ + λ₁‖LP(x̂₀)-LP(x₀)‖₁ + λ₂‖HP(x̂₀)-HP(x₀)‖₁
            "multiscale_weight": -0.01,       # Overall weight for multiscale loss (0 = disabled)
            "ms_lambda_0": 1.0,              # Weight for original scale (λ₀)
            "ms_lambda_lp": 0.5,             # Weight for low-pass component (λ₁)
            "ms_lambda_hp": 0.1,             # Weight for high-pass component (λ₂)
            "ms_kernel_size": 5,             # Gaussian kernel size for LP filter
            "ms_sigma": 1.5,                 # Gaussian sigma for LP filter
            "ms_time_weight": "sqrt_snr",    # Time-dependent weight: "none", "sqrt_snr", "snr", "linear"
        }
    )
    # --------------minjing end 2025/09/03--------------



    defaults = model_and_diffusion_defaults()
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)


    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_pot/logs/train_LapSmooth/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_pot/logs/train_default/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_pot/logs/train_CFG/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_pot/logs/train_GradientLoss/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_pot/logs/train_MultiscaleLoss/"
    # os.environ["OPENAI_LOGDIR"] = "/home/lim2/code/improved-diffusion_mj_1D_pot/logs/train_default_newell/"
    # --- Cell boundary weighting (mj_20260317) ---
    _use_cell_weight = int(os.environ.get("TRAIN_CELL_WEIGHT", "0"))
    if _use_cell_weight:
        args.loss_cfg["cell_weight"] = True
        args.loss_cfg["cell_w_cell"] = 2.0     # inside convection cells
        args.loss_cfg["cell_w_bndry"] = 3.0    # cell boundary pixels
        args.loss_cfg["cell_w_floor"] = 0.5    # background outside cells
        args.loss_cfg["cell_taper_width"] = 3  # cosine taper width (pixels)
        args.loss_cfg["pixel_amp_weight"] = False  # cell weight replaces pixel amp

    _logdir = os.environ.get("TRAIN_LOGDIR",
                             "/home/lim2/code/improved-diffusion_mj_1D_pot/logs/train_default_AL/")
    os.environ["OPENAI_LOGDIR"] = _logdir

    _resume = os.environ.get("TRAIN_RESUME_CKPT", "")
    if _resume:
        args.resume_checkpoint = _resume

    # Per-rank batch size override (mj_20260901): train_default used 2 GPUs x 128 = 256 global;
    # to reproduce global batch 256 on 4 GPUs set TRAIN_BATCH_SIZE=64.
    _bs = os.environ.get("TRAIN_BATCH_SIZE", "")  # mj_20260901
    if _bs:  # mj_20260901
        args.batch_size = int(_bs)  # mj_20260901
        print(f"[mj_20260901] per-rank batch_size overridden to {args.batch_size}")  # mj_20260901

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, defaults.keys())
    )




    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        use_structured_cond=args.use_structured_cond, # mj_20250530
        class_cond=args.class_cond,
        spatial_basis=args.spatial_basis, # mj_20250911
        use_cell_boundaries=bool(_use_cell_weight),  # mj_20260317
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
        cfg_dropout_prob=args.cfg_dropout_prob,  # CFG unconditional dropout
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
        save_interval=int(os.environ.get("TRAIN_SAVE_INTERVAL", "10000")),  # mj_20260927 env override (fine-tunes)
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
