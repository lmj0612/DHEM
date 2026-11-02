import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        loss_cfg=None, # minjing 2025/09/03
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.loss_cfg = loss_cfg # minjing 2025/09/03
        self.cond_drop_p = float(os.environ.get("TRAIN_COND_DROP", "0"))  # mj_20260927 CFG: per-sample probability of replacing the condition embedding by the model's null_emb (model must be created with cond_drop=True)
        if self.cond_drop_p > 0:  # mj_20260927
            assert hasattr(self.model, "null_emb"), "TRAIN_COND_DROP>0 needs a model built with cond_drop=True (TRAIN_COND_DROP sets it in image_train.py)"  # mj_20260927
            logger.log(f"TRAIN_COND_DROP={self.cond_drop_p}: classifier-free-guidance condition dropout (per sample, learned null embedding)")  # mj_20260927

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        # mj_20260903: TRAIN_PARTIAL_KEEP_STATE=1 -> a shape-mismatched (widened cond_dim) load behaves like a
        # TRUE resume: new input columns are ZERO-initialised (model == old model at step 0), step/opt/EMA are kept,
        # with the optimizer moments and EMA tensors zero-padded to the new shapes. Removes the fresh-Adam
        # protocol loss when only the label_emb input width changes.
        self._partial_pad = {}  # mj_20260903 key -> (old_shape, new_shape)
        self._keep_state = bool(int(os.environ.get("TRAIN_PARTIAL_KEEP_STATE", "0")))  # mj_20260903
        self._new_param_names = set()  # mj_20260927 parameters absent from the checkpoint (new layers, e.g. MLTCrossAttention) when KEEP_STATE keeps the step

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            saved_state = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev()
            )
            model_state = self.model.state_dict()  # mj_20260305
            # Partial load: handle shape mismatches (e.g. cond_dim 48->50)  # mj_20260305
            mismatched = []  # mj_20260305
            for key in saved_state:  # mj_20260305
                if key in model_state and saved_state[key].shape != model_state[key].shape:  # mj_20260305
                    mismatched.append(key)  # mj_20260305
            if mismatched:  # mj_20260305
                logger.log(f"Shape-mismatched keys (partial load): {mismatched}")  # mj_20260305
                for key in mismatched:  # mj_20260305
                    old_w = saved_state[key]  # mj_20260305
                    new_w = (th.zeros_like(model_state[key]) if self._keep_state else model_state[key].clone())  # mj_20260305, mj_20260903 zero-init new dims when keeping state
                    self._partial_pad[key] = (tuple(old_w.shape), tuple(new_w.shape))  # mj_20260903
                    # Copy overlapping region, leave new dims at init  # mj_20260305
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_w.shape, new_w.shape))  # mj_20260305
                    new_w[slices] = old_w[slices]  # mj_20260305
                    saved_state[key] = new_w  # mj_20260305
                    logger.log(f"  {key}: {old_w.shape} -> {new_w.shape} (partial copy)")  # mj_20260305
                # Reset step to 0 for partial load (fresh optimizer, no EMA load)  # mj_20260305
                if self._keep_state:  # mj_20260903
                    logger.log(f"TRAIN_PARTIAL_KEEP_STATE=1: keeping resume_step {self.resume_step}; opt/EMA will be loaded with zero-padded tensors for {list(self._partial_pad)}")  # mj_20260903
                else:  # mj_20260903
                    self.resume_step = 0  # mj_20260305
                    logger.log("Partial load: resume_step reset to 0 (fresh optimizer + EMA)")  # mj_20260305
            # Handle missing keys (new layers) and unexpected keys (removed layers)  # mj_20260312
            missing = set(model_state.keys()) - set(saved_state.keys())  # mj_20260312
            unexpected = set(saved_state.keys()) - set(model_state.keys())  # mj_20260312
            if missing or unexpected:  # mj_20260312
                if missing:  # mj_20260312
                    logger.log(f"Missing keys (new layers, keep init): {sorted(missing)[:5]}... ({len(missing)} total)")  # mj_20260312
                if unexpected:  # mj_20260312
                    logger.log(f"Unexpected keys (removed, ignoring): {sorted(unexpected)[:5]}... ({len(unexpected)} total)")  # mj_20260312
                # Copy only matching keys from saved_state into model_state  # mj_20260312
                for key in saved_state:  # mj_20260312
                    if key in model_state:  # mj_20260312
                        model_state[key] = saved_state[key]  # mj_20260312
                saved_state = model_state  # mj_20260312
                if self._keep_state and missing and not unexpected:  # mj_20260927 new layers only (nothing dropped) -> TRUE resume: keep step, remap opt/EMA by name
                    self._new_param_names = {n for n, _ in self.model.named_parameters() if n in missing}  # mj_20260927
                    logger.log(f"TRAIN_PARTIAL_KEEP_STATE=1: {len(missing)} new keys keep their init ({len(self._new_param_names)} params, e.g. {sorted(self._new_param_names)[:2]}); resume_step {self.resume_step} kept; EMA/optimizer state remapped by parameter name, new params start with fresh Adam moments")  # mj_20260927
                else:  # mj_20260927
                    self.resume_step = 0  # mj_20260312
                    logger.log("Partial load (missing/unexpected): resume_step reset to 0")  # mj_20260312
            self.model.load_state_dict(saved_state)
            # mj_20260902: TRAIN_RESET_STEP=1 -> weights-only warm start: step 0,
            # fresh optimizer, EMA re-initialised from the loaded weights. Makes a
            # same-shape control resume identical in protocol to the partial-load
            # fine-tunes (which already reset step/opt/EMA on shape mismatch).
            if int(os.environ.get("TRAIN_RESET_STEP", "0")) and self.resume_step != 0:  # mj_20260902
                logger.log(f"TRAIN_RESET_STEP=1: weights-only load, resume_step {self.resume_step} -> 0 (fresh opt/EMA)")  # mj_20260902
                self.resume_step = 0  # mj_20260902

        dist_util.sync_params(self.model.parameters())

    def _pad_to_model(self, key, t):  # mj_20260903 zero-pad a saved tensor to the current model shape (overlap copy)
        target = self.model.state_dict()[key] if key in self.model.state_dict() else None  # mj_20260903
        new_shape = self._partial_pad[key][1]  # mj_20260903
        out = th.zeros(new_shape, dtype=t.dtype, device=t.device)  # mj_20260903
        sl = tuple(slice(0, min(o, n)) for o, n in zip(t.shape, new_shape))  # mj_20260903
        out[sl] = t[sl]  # mj_20260903
        return out  # mj_20260903

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            for key in self._partial_pad:  # mj_20260903 widened params: zero-pad EMA tensors
                if key in state_dict and tuple(state_dict[key].shape) != self._partial_pad[key][1]:  # mj_20260903
                    state_dict[key] = self._pad_to_model(key, state_dict[key])  # mj_20260903
                    logger.log(f"  EMA {key}: padded to {self._partial_pad[key][1]}")  # mj_20260903
            ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            if self._new_param_names and not self.use_fp16:  # mj_20260927 remap the saved per-parameter state (indexed in the OLD parameter order) onto the new order
                names = [n for n, _ in self.model.named_parameters()]  # mj_20260927
                old_names = [n for n in names if n not in self._new_param_names]  # mj_20260927 old order == new order minus the new params (module registration order unchanged)
                new_index = {n: i for i, n in enumerate(names)}  # mj_20260927
                groups = state_dict['param_groups']  # mj_20260927
                assert len(groups) == 1, f"expected 1 param group, got {len(groups)}"  # mj_20260927
                assert len(groups[0]['params']) == len(old_names), (len(groups[0]['params']), len(old_names))  # mj_20260927 sanity: checkpoint was written by the old-layout model
                old_state = state_dict['state']  # mj_20260927
                new_state = {}  # mj_20260927
                for old_idx, n in enumerate(old_names):  # mj_20260927
                    st = old_state.get(old_idx, old_state.get(str(old_idx)))  # mj_20260927
                    if st is not None:  # mj_20260927
                        new_state[new_index[n]] = st  # mj_20260927
                state_dict['state'] = new_state  # mj_20260927
                groups[0]['params'] = list(range(len(names)))  # mj_20260927
                logger.log(f"  optimizer state remapped by name: {len(new_state)} carried, {len(self._new_param_names)} new params fresh")  # mj_20260927
            if self._partial_pad and not self.use_fp16:  # mj_20260903 pad Adam moments of widened params (param order == named_parameters order)
                names = [n for n, _ in self.model.named_parameters()]  # mj_20260903
                for idx, name in enumerate(names):  # mj_20260903
                    if name not in self._partial_pad:  # mj_20260903
                        continue  # mj_20260903
                    st = state_dict['state'].get(idx, state_dict['state'].get(str(idx)))  # mj_20260903
                    if st is None:  # mj_20260903
                        continue  # mj_20260903
                    for mkey in ('exp_avg', 'exp_avg_sq'):  # mj_20260903
                        if mkey in st and tuple(st[mkey].shape) != self._partial_pad[name][1]:  # mj_20260903
                            st[mkey] = self._pad_to_model(name, st[mkey])  # mj_20260903
                    logger.log(f"  optimizer state for {name}: moments zero-padded to {self._partial_pad[name][1]}")  # mj_20260903
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        _max_step = int(os.environ.get("TRAIN_MAX_STEP", "0"))  # mj_20260918 0 = run forever (default)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if _max_step and self.step + self.resume_step >= _max_step:  # mj_20260918 TRAIN_MAX_STEP: stop after the checkpoint at this global step (fine-tune scans)
                logger.log(f"TRAIN_MAX_STEP={_max_step} reached at step {self.step + self.resume_step}; stopping")  # mj_20260918
                return  # mj_20260918
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            if self.cond_drop_p > 0:  # mj_20260927 CFG dropout mask (independent per sample)
                micro_cond["cond_null"] = th.rand(micro.shape[0], device=dist_util.dev()) < self.cond_drop_p  # mj_20260927
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                loss_cfg=self.loss_cfg,   # minjing 2025/09/03
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        _cur = None  # mj_20260927
        _filled = []  # mj_20260927
        params = []  # mj_20260927
        for name, _ in self.model.named_parameters():  # mj_20260927 (was a one-line comprehension)
            if name in state_dict:  # mj_20260927
                params.append(state_dict[name])  # mj_20260927
            else:  # mj_20260927 new layer absent from the EMA checkpoint -> start its EMA from the current (init) weights
                if _cur is None:  # mj_20260927
                    _cur = self.model.state_dict()  # mj_20260927
                params.append(_cur[name].detach().clone())  # mj_20260927
                _filled.append(name)  # mj_20260927
        if _filled:  # mj_20260927
            logger.log(f"  EMA: {len(_filled)} new params initialised from the model (e.g. {_filled[:2]})")  # mj_20260927
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
