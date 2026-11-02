"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

import torch.nn.functional as F  # minjing 2025/09/03


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


    # --------------minjing start 2025/09/03--------------
    @staticmethod
    def _weighted_mean_flat(x, w):
        """
        Per-sample weighted mean over all non-batch dims.
        x, w: [B, C, ...] with identical shape.
        returns: [B]
        """
        B = x.shape[0]
        xw = (x * w).reshape(B, -1).sum(dim=1)
        wsum = w.reshape(B, -1).sum(dim=1) + 1e-8
        return xw / wsum

    @staticmethod
    def _compute_amp_weights(x0, amp_thresh=5.0, amp_scale=1.0, amp_gamma=1.0, amp_base=1.0):
        """
        Pixel-wise weights from target amplitude (ground-truth x0).
        x0 can be in your working units (normalized or physical) — set amp_thresh accordingly.
        w = amp_base + amp_scale * ReLU((x0 - amp_thresh)/max(amp_thresh,eps)) ** amp_gamma
        """
        rel = F.relu(x0 - amp_thresh) / (amp_thresh + 1e-8)
        w = amp_base + amp_scale * (rel ** amp_gamma)
        return w

    @staticmethod
    def _topk_mask_per_sample(ref, topk_frac=0.02):
        """
        ref: [B, C, H, W] (or any shape after [B, C, ...])
        Returns a float mask of the same shape where the top-k fraction
        (per sample) are marked as 1.0 and the rest 0.0.
        """
        B = ref.shape[0]
        flat = ref.view(B, -1)
        k = max(1, int(topk_frac * flat.shape[1]))
        thresh = th.topk(flat, k, dim=1).values[:, -1]  # [B]
        thresh = thresh.view(B, 1, 1, 1)
        return (ref >= thresh).float()

    @staticmethod  # mj_20260302
    def _build_oval_weight_mask(eb, pb, H, W, w_oval=2.0, w_bndry=3.0, w_floor=0.5, taper_width=3):  # mj_20260302
        """
        Build a per-pixel weight mask [B, 1, H, W] from oval boundaries.  # mj_20260302

        eb: [B, W] int16 — equatorward boundary row index (255=no boundary)  # mj_20260302
        pb: [B, W] int16 — poleward boundary row index (255=no boundary)  # mj_20260302
        H, W: spatial dimensions  # mj_20260302

        Weight profile per MLAT column:  # mj_20260302
          - inside oval [PB..EB]:  w_oval  # mj_20260302
          - at EB and PB pixels:   w_bndry (extra boundary emphasis)  # mj_20260302
          - background:            w_floor, with cosine taper over taper_width into oval  # mj_20260302
          - no boundary detected:  1.0 (uniform)  # mj_20260302
        """
        B = eb.shape[0]  # mj_20260302
        device = eb.device  # mj_20260302
        SENTINEL = 255  # mj_20260302

        # row indices [1, 1, H, 1]  # mj_20260302
        rows = th.arange(H, device=device, dtype=th.float32).view(1, 1, H, 1)  # mj_20260302

        # Reshape boundaries to [B, 1, 1, W] for broadcasting  # mj_20260302
        eb_f = eb.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, W]  # mj_20260302
        pb_f = pb.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, W]  # mj_20260302

        # Mask for columns with valid boundaries (not sentinel)  # mj_20260302
        valid = (eb < SENTINEL).unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, W]  # mj_20260302

        # Inside oval: pb <= row <= eb  # mj_20260302
        inside = ((rows >= pb_f) & (rows <= eb_f)).float()  # [B, 1, H, W]  # mj_20260302

        # Boundary pixels: row == eb or row == pb  # mj_20260302
        at_eb = (rows == eb_f).float()  # mj_20260302
        at_pb = (rows == pb_f).float()  # mj_20260302
        at_bndry = ((at_eb + at_pb) > 0).float()  # mj_20260302

        # Distance from oval (for taper): positive outside, 0 inside  # mj_20260302
        dist_eq = (rows - eb_f).clamp(min=0)   # distance equatorward of EB  # mj_20260302
        dist_po = (pb_f - rows).clamp(min=0)   # distance poleward of PB  # mj_20260302
        dist_outside = dist_eq + dist_po        # one of them is always 0 inside oval  # mj_20260302

        # Cosine taper: 1 at boundary → 0 at taper_width pixels away  # mj_20260302
        if taper_width > 0:  # mj_20260302
            taper = 0.5 * (1.0 + th.cos(th.pi * (dist_outside / taper_width).clamp(max=1.0)))  # mj_20260302
        else:  # mj_20260302
            taper = th.zeros_like(dist_outside)  # mj_20260302

        # Build weight:  # mj_20260302
        #   inside oval (not boundary): w_oval  # mj_20260302
        #   at boundary: w_bndry  # mj_20260302
        #   outside with taper: w_floor + (w_oval - w_floor) * taper  # mj_20260302
        #   outside beyond taper: w_floor  # mj_20260302
        w_inside = inside * (1.0 - at_bndry) * w_oval + at_bndry * w_bndry  # mj_20260302
        w_outside = (1.0 - inside) * (w_floor + (w_oval - w_floor) * taper)  # mj_20260302
        w = w_inside + w_outside  # mj_20260302

        # For columns with no boundary, use uniform weight=1.0  # mj_20260302
        w = valid * w + (1.0 - valid) * 1.0  # mj_20260302

        return w  # [B, 1, H, W]  # mj_20260302

    def _hp_from_x0(self, x0, loss_cfg):  # mj_20260927
        """
        Hemispheric power [GW] of each normalised training target.  # mj_20260927
        x0: [B, 1, H, W] = log10(energy flux [mW/m^2]) rescaled to [-1, 1] with the train80 range  # mj_20260927
        (oval_hp_vmin / oval_hp_vmax, log10 offset oval_hp_eps); oval_hp_area = [H, W] cell areas [m^2].  # mj_20260927
        HP = sum(flux * area) * 1e-12  (mW -> GW), identical to scripts/cat_candidate_metrics.py.  # mj_20260927
        """
        return self._fluxarea_from_x0(x0, loss_cfg).sum(dim=(1, 2, 3)) * 1e-12  # [B] GW  # mj_20260927, mj_20261026 via _fluxarea_from_x0

    def _fluxarea_from_x0(self, x0, loss_cfg):  # mj_20261026
        """flux [mW/m^2] * cell area [m^2] per pixel, [B, 1, H, W]; sum * 1e-12 = HP [GW]."""  # mj_20261026
        vmin = float(loss_cfg["oval_hp_vmin"]); vmax = float(loss_cfg["oval_hp_vmax"]); eps = float(loss_cfg.get("oval_hp_eps", 1e-2))  # mj_20260927
        area = loss_cfg["oval_hp_area"]  # numpy [H, W]  # mj_20260927
        if getattr(self, "_oval_area_t", None) is None or self._oval_area_t.device != x0.device:  # mj_20260927
            self._oval_area_t = th.as_tensor(np.asarray(area, dtype=np.float32), device=x0.device)  # mj_20260927
        x = x0[:, :1].float()  # [B, 1, H, W]  # mj_20260927, mj_20261026 keep channel dim
        flux = th.clamp(th.pow(10.0, (x + 1.0) / 2.0 * (vmax - vmin) + vmin) - eps, min=0.0)  # mW/m^2  # mj_20260927
        return flux * self._oval_area_t  # [B, 1, H, W]  # mj_20261026

    @staticmethod  # mj_20261026
    def _oval_inside_mask(eb, pb, H, W):  # mj_20261026
        """[B, 1, H, W] float: 1 for PB <= row <= EB in columns with a detected boundary (same convention as _build_oval_weight_mask)."""  # mj_20261026
        rows = th.arange(H, device=eb.device, dtype=th.float32).view(1, 1, H, 1)  # mj_20261026
        eb_f = eb.float().unsqueeze(1).unsqueeze(2); pb_f = pb.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, W]  # mj_20261026
        valid = (eb < 255).unsqueeze(1).unsqueeze(2).float()  # mj_20261026
        return ((rows >= pb_f) & (rows <= eb_f)).float() * valid  # mj_20261026

    #  --------------minjing end 2025/09/03--------------





    # def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, loss_cfg=None): # minjing 2025/09/03

        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        loss_cfg = loss_cfg or {} # minjing 2025/09/03
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]


            assert model_output.shape == target.shape == x_start.shape
            # --------------minjing start 2025/09/03--------------

            # -------------------------------------------
            # (1) Pixel-wise amplitude weighting (by x0)
            # -------------------------------------------
            use_pixel_amp = bool(loss_cfg.get("pixel_amp_weight", False))
            use_oval_weight = bool(loss_cfg.get("oval_weight", False))  # mj_20260302
            if use_oval_weight and model_kwargs is not None and "oval_eb" in model_kwargs:  # mj_20260302
                # Oval boundary weighting — replaces pixel_amp_weight  # mj_20260302
                per_pix_mse = (target - model_output) ** 2  # mj_20260302
                oval_eb = model_kwargs["oval_eb"]  # [B, W] int16  # mj_20260302
                oval_pb = model_kwargs["oval_pb"]  # [B, W] int16  # mj_20260302
                w_oval_mask = self._build_oval_weight_mask(  # mj_20260302
                    oval_eb, oval_pb,  # mj_20260302
                    H=x_start.shape[2], W=x_start.shape[3],  # mj_20260302
                    w_oval=float(loss_cfg.get("oval_w_oval", 2.0)),  # mj_20260302
                    w_bndry=float(loss_cfg.get("oval_w_bndry", 3.0)),  # mj_20260302
                    w_floor=float(loss_cfg.get("oval_w_floor", 0.5)),  # mj_20260302
                    taper_width=int(loss_cfg.get("oval_taper_width", 3)),  # mj_20260302
                )  # mj_20260302
                # mj_20261026: HP-SHARE scaling INSIDE the k-means oval. Diagnosis of the 2026-09-27 null round: the oval
                # (PB..EB) covers 34 % of the map at median and 46 % for storm maps and already holds ~95 % of the HP, and the
                # top 10 % of its pixels carry ~1/3 of the HP -- yet the 2026-03 mask weights every inside pixel equally, so the
                # paper weighting only de-emphasises polar cap / sub-auroral background. Here every inside pixel is additionally
                # scaled by g_ij = clip(N_in * f_ij A_ij / sum_in f A, oval_share_min, oval_share_max) (mean 1 over the oval, so
                # the oval's total weight is unchanged): the pixel weight becomes proportional to the pixel's share of HP, i.e.
                # the loss is a first-order proxy of the HP error inside the oval. oval_share_max <= 0 disables (paper mask).
                _smax = float(loss_cfg.get("oval_share_max", 0.0))  # mj_20261026
                if _smax > 0:  # mj_20261026
                    _smin = float(loss_cfg.get("oval_share_min", 0.5))  # mj_20261026
                    _inside = self._oval_inside_mask(oval_eb, oval_pb, x_start.shape[2], x_start.shape[3])  # [B,1,H,W]  # mj_20261026
                    _fa_in = self._fluxarea_from_x0(x_start, loss_cfg) * _inside  # mj_20261026
                    _n_in = _inside.sum(dim=(1, 2, 3), keepdim=True)  # mj_20261026
                    _g = (_n_in * _fa_in / (_fa_in.sum(dim=(1, 2, 3), keepdim=True) + 1e-12)).clamp(_smin, _smax)  # mj_20261026
                    w_oval_mask = th.where(_inside > 0, w_oval_mask * _g, w_oval_mask)  # mj_20261026
                    terms["oval_w_in_max"] = (w_oval_mask * _inside).amax(dim=(1, 2, 3)).detach()  # logged only  # mj_20261026
                # mj_20260927: HP gate -- quiet maps (small hemispheric power) have no meaningful oval boundary,
                # so the spatial weighting is faded out below oval_hp_gate_hi GW and switched off below
                # oval_hp_gate_lo GW (gate 0 -> uniform weight 1). HP is computed on the fly from the
                # normalised log10 target x_start (train80 range) with the AMIE cell areas (same formula
                # as the analysis scripts). oval_hp_gate_hi <= 0 disables the gate (old behaviour).
                _hp_hi = float(loss_cfg.get("oval_hp_gate_hi", 0.0))  # mj_20260927
                if _hp_hi > 0:  # mj_20260927
                    _hp_lo = float(loss_cfg.get("oval_hp_gate_lo", 0.0))  # mj_20260927
                    _hp = self._hp_from_x0(x_start, loss_cfg)  # [B] GW  # mj_20260927
                    _gate = ((_hp - _hp_lo) / max(_hp_hi - _hp_lo, 1e-6)).clamp(0.0, 1.0)  # mj_20260927
                    w_oval_mask = 1.0 + _gate.view(-1, 1, 1, 1) * (w_oval_mask - 1.0)  # mj_20260927
                    terms["oval_gate"] = _gate.detach()  # logged only (not part of the loss)  # mj_20260927
                    terms["oval_hp"] = _hp.detach()  # logged only  # mj_20260927
                terms["mse"] = self._weighted_mean_flat(per_pix_mse, w_oval_mask)  # mj_20260302
                # mj_20261026: optional per-SAMPLE storm re-weighting m = clip((HP/ref)^gamma, min, max), normalised to batch
                # mean 1 (lr scale unchanged). Addresses the storm-tail (HP>60 GW) under-prediction / storm rarity directly;
                # oval_hpsamp_gamma <= 0 disables.
                _sg = float(loss_cfg.get("oval_hpsamp_gamma", 0.0))  # mj_20261026
                if _sg > 0:  # mj_20261026
                    _hp_s = self._hp_from_x0(x_start, loss_cfg)  # [B] GW  # mj_20261026
                    _m = ((_hp_s.clamp(min=1e-3) / float(loss_cfg.get("oval_hpsamp_ref", 30.0))) ** _sg)  # mj_20261026
                    _m = _m.clamp(float(loss_cfg.get("oval_hpsamp_min", 0.7)), float(loss_cfg.get("oval_hpsamp_max", 2.0)))  # mj_20261026
                    _m = _m / (_m.mean() + 1e-12)  # mj_20261026
                    terms["mse"] = terms["mse"] * _m  # mj_20261026
                    terms["oval_msamp"] = _m.detach()  # logged only  # mj_20261026
            elif use_pixel_amp:
                # weights are computed from ground-truth x0 amplitudes
                # If your training uses normalized x0, set amp_thresh in that space.
                amp_thresh = float(loss_cfg.get("amp_thresh", 5.0))
                amp_scale  = float(loss_cfg.get("amp_scale", 1.0))
                amp_gamma  = float(loss_cfg.get("amp_gamma", 1.0))
                amp_base   = float(loss_cfg.get("amp_base", 1.0))

                w_pix = self._compute_amp_weights(
                    x0=x_start,
                    amp_thresh=amp_thresh,
                    amp_scale=amp_scale,
                    amp_gamma=amp_gamma,
                    amp_base=amp_base,
                )
                per_pix_mse = (target - model_output) ** 2
                terms["mse"] = self._weighted_mean_flat(per_pix_mse, w_pix)
            else:
                terms["mse"] = mean_flat((target - model_output) ** 2)

            # ------------------------------------------------
            # (2) Sample-wise weighting by AE (rebalance high-AE)
            # ------------------------------------------------
            if bool(loss_cfg.get("sample_ae_weight", False)):
                # from model_kwargs get cond matrix
                if model_kwargs is None or "cond" not in model_kwargs:
                    raise KeyError("sample_ae_weight=True but 'cond' not found in model_kwargs.")

                # cond is [B, D], column order is [sin(UT), cos(UT), sin(DOY), cos(DOY), AE]
                cond_mat = model_kwargs["cond"]
                if cond_mat.dim() != 2 or cond_mat.size(1) < 5:
                    raise ValueError("model_kwargs['cond'] shape is not expected, should be [B, >=5] to get AE.")

                # get AE and normalize to float, [B]
                ae = cond_mat[:, 4].float()
                if ae.dim() == 0:  # defense for extreme cases
                    ae = ae.unsqueeze(0)

                ae_pivot     = float(loss_cfg.get("ae_pivot", 500.0))
                ae_slope     = max(1e-6, float(loss_cfg.get("ae_slope", 100.0)))  # prevent division by zero
                ae_max_boost = float(loss_cfg.get("ae_max_boost", 3.0))
                ae_gamma     = float(loss_cfg.get("ae_gamma", 1.0))

                # smooth boost the weight of high AE samples
                w_samp = 1.0 + (ae_max_boost - 1.0) * th.sigmoid((ae - ae_pivot) / ae_slope)
                if ae_gamma != 1.0:
                    w_samp = w_samp ** ae_gamma

                # terms["mse"] is [B], scale per-sample
                terms["mse"] = terms["mse"] * w_samp

            # ----------------------------------------------------------
            # (3) Top-k x0 loss (explicitly push highest-energy pixels)
            # ----------------------------------------------------------
            topk_frac = float(loss_cfg.get("topk_x0_frac", 0.0))
            if topk_frac > 0.0:
            
                # 1) AE (per-sample gating / weighting)
                cond_mat = model_kwargs["cond"]
                if cond_mat.dim() != 2 or cond_mat.size(1) < 1:
                    raise ValueError("model_kwargs['cond'] shape is not expected, should be [B, >=5] to get AE.")
                z_al = cond_mat[:, -1].float() # mj_20250912
                if z_al.dim() == 0:   
                    z_al = z_al.unsqueeze(0)
                al_mean, al_std =   -1.23843925e2,1.6298245e2

                al_raw = z_al * al_std + al_mean
                A = (-al_raw).clamp_min(0.0)               # |AL| >= 0
                logA = th.log10(A + 1e-3)  # avoid log(0)

                # thresholds (log10)
                t1, t2, t3 = 1.95, 2.60, 3.00              # ~90, 400, 1000 nT
                w = th.ones_like(logA)  # baseline 1.0

                def ramp(x, a, b):  # linear 0 -> 1 between a and b
                    return ((x - a) / (b - a)).clamp(0, 1)

                w = w + 40.0  * ramp(logA, t1, t1+0.05)  # up to +4, total ~5
                w = w + 60.0 * ramp(logA, t2, t2+0.05)  # up to +15, total ~20
                w = w + 100.0 * ramp(logA, t3, t3+0.05)  # up to +30, total ~50
                w = w.clamp_max(50.0)  # cap at 50
                # optional: normalise by the batch mean so that the mean weight is ~1
                gate = w / (w.mean().detach() + 1e-6)



                # 2) pred_x0
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_x0 = model_output
                elif self.model_mean_type == ModelMeanType.EPSILON:
                    pred_x0 = self._predict_xstart_from_eps(x_t, t, model_output)
                else:  # PREVIOUS_X
                    pred_x0 = self._predict_xstart_from_xprev(x_t, t, model_output)



                # 3) top-k over under-predicted pixels only
                with th.no_grad():
                    mask_x0 = self._topk_mask_per_sample(x_start, topk_frac=topk_frac) # auroral arc region
                    err_pos = (x_start - pred_x0).clamp_min(0.0)
                    mask_weight = mask_x0 * th.sigmoid(err_pos / 0.08)  # [B,1,H,W]


                # 4) normalised error and top-k MSE
                if_normalize_topk_error = bool(loss_cfg.get("if_normalize_topk_error", False))
                if if_normalize_topk_error:
                    with th.no_grad():
                        q = max(0.85, 1 - 0.5*topk_frac)  # frac = 0.15 -> q ~ 0.925
                        vals = x_start.clamp_min(0.0).view(B, -1)  # positive side only
                        scale_raw = th.quantile(vals, q, dim=1)       # [B]
                    scale = scale_raw.clamp(float(loss_cfg.get("topk_norm_min", 0.20)), float(loss_cfg.get("topk_norm_max", 0.55))).view(B,1,1,1)

                    resid = (pred_x0 - x_start) / scale
                    topk_mse = self._weighted_mean_flat(resid.pow(2), mask_weight)
                else:
                    topk_mse = self._weighted_mean_flat((pred_x0 - x_start) ** 2, mask_weight)


                # 5) total MSE
                base_w = float(loss_cfg.get("base_weight", 1.0))   # keep some of the base loss
                topk_w = float(loss_cfg.get("topk_weight", 1.0))   # how hard to push the peaks
                terms["mse"] = base_w * terms["mse"] + (topk_w*gate) * topk_mse
            # --------------minjing end 2025/09/03--------------

            # -------------- start supervised loss (optional) --------------
            sup_w = float(loss_cfg.get("sup_weight", 0.0))
            if sup_w > 0.0:
                # Case A: pixel-aligned supervision (same shape as x_start)
                
                # pred_x0
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_x0 = model_output
                elif self.model_mean_type == ModelMeanType.EPSILON:
                    pred_x0 = self._predict_xstart_from_eps(x_t, t, model_output)
                else:  # PREVIOUS_X
                    pred_x0 = self._predict_xstart_from_xprev(x_t, t, model_output)
                    
                sup_type = loss_cfg.get("sup_type", "l1")  # "l1" or "mse"
                if sup_type == "l1":
                    sup_loss = (pred_x0 - x_start).abs()
                    sup_loss = mean_flat(sup_loss)
                elif sup_type == "smooth_l1":
                    # smooth L1 / Huber, beta adjustable
                    beta = float(loss_cfg.get("sup_beta", 0.1))
                    diff = pred_x0 - x_start
                    abs_diff = diff.abs()
                    # element-wise Huber, then mean_flat to [B]
                    huber = th.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
                    sup_loss = mean_flat(huber)    
                else:
                    sup_loss = mean_flat((pred_x0 - x_start) ** 2)
                terms["mse"] = terms["mse"] + sup_w * sup_loss
            # -------------- end supervised loss (optional) --------------

            # -------------- start Laplacian smoothing loss (optional) --------------
            smooth_w = float(loss_cfg.get("smooth_weight", 0.0))
            if smooth_w > 0.0:
                # Compute pred_x0 if not already computed in supervised loss
                if sup_w <= 0.0:
                    if self.model_mean_type == ModelMeanType.START_X:
                        pred_x0 = model_output
                    elif self.model_mean_type == ModelMeanType.EPSILON:
                        pred_x0 = self._predict_xstart_from_eps(x_t, t, model_output)
                    else:  # PREVIOUS_X
                        pred_x0 = self._predict_xstart_from_xprev(x_t, t, model_output)
                
                smooth_type = loss_cfg.get("smooth_type", "laplacian")  # "laplacian" or "gradient"
                smooth_loss_type = loss_cfg.get("smooth_loss_type", "l2")  # "l1" or "l2"
                
                # Determine data dimensionality: [B, C, ...] 
                # For 1D data: [B, C, L], for 2D data: [B, C, H, W]
                ndim = pred_x0.dim()
                
                if smooth_type == "laplacian":
                    # Second-order Laplacian: penalize curvature
                    if ndim == 3:  # 1D case: [B, C, L]
                        # Laplacian 1D: x[i-1] - 2*x[i] + x[i+1]
                        lap = pred_x0[:, :, :-2] - 2 * pred_x0[:, :, 1:-1] + pred_x0[:, :, 2:]
                    elif ndim == 4:  # 2D case: [B, C, H, W]
                        # Laplacian 2D using convolution kernel
                        lap_kernel = th.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                               dtype=pred_x0.dtype, device=pred_x0.device)
                        lap_kernel = lap_kernel.view(1, 1, 3, 3).expand(pred_x0.shape[1], -1, -1, -1)
                        lap = th.nn.functional.conv2d(pred_x0, lap_kernel, padding=1, groups=pred_x0.shape[1])
                    else:
                        raise ValueError(f"Unsupported tensor dimension {ndim} for Laplacian smoothing")
                        
                elif smooth_type == "gradient":
                    # First-order gradient: penalize rapid changes
                    if ndim == 3:  # 1D case: [B, C, L]
                        lap = pred_x0[:, :, 1:] - pred_x0[:, :, :-1]
                    elif ndim == 4:  # 2D case: [B, C, H, W]
                        grad_h = pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]
                        grad_w = pred_x0[:, :, :, 1:] - pred_x0[:, :, :, :-1]
                        # Combine gradients (pad to same size and sum)
                        lap = th.cat([mean_flat(grad_h.abs() if smooth_loss_type == "l1" else grad_h ** 2).unsqueeze(-1),
                                      mean_flat(grad_w.abs() if smooth_loss_type == "l1" else grad_w ** 2).unsqueeze(-1)], dim=-1).mean(dim=-1)
                    else:
                        raise ValueError(f"Unsupported tensor dimension {ndim} for gradient smoothing")
                else:
                    raise ValueError(f"Unknown smooth_type: {smooth_type}")
                
                # Compute loss from Laplacian/gradient
                if smooth_type != "gradient" or ndim != 4:  # gradient + 2D already computed above
                    if smooth_loss_type == "l1":
                        smooth_loss = mean_flat(lap.abs())
                    else:  # l2
                        smooth_loss = mean_flat(lap ** 2)
                else:
                    smooth_loss = lap  # already computed for gradient + 2D case
                
                # Time-dependent weighting: reduce weight at high noise levels
                smooth_time_weight_type = loss_cfg.get("smooth_time_weight", "sqrt_snr")  # "none", "sqrt_snr", "snr", "linear"
                if smooth_time_weight_type == "sqrt_snr":
                    # weight = sqrt(alpha_bar_t), higher at small t (less noise)
                    time_weight = _extract_into_tensor(self.sqrt_alphas_cumprod, t, (t.shape[0],))
                elif smooth_time_weight_type == "snr":
                    # weight = alpha_bar_t
                    time_weight = _extract_into_tensor(self.alphas_cumprod, t, (t.shape[0],))
                elif smooth_time_weight_type == "linear":
                    # weight = (T - t) / T, linearly decreases with t
                    time_weight = (self.num_timesteps - t.float()) / self.num_timesteps
                else:  # "none"
                    time_weight = 1.0
                    
                terms["mse"] = terms["mse"] + smooth_w * time_weight * smooth_loss
                # terms["smooth_loss"] = smooth_loss  # for logging
            # -------------- end Laplacian smoothing loss (optional) --------------

            # -------------- start gradient matching loss (optional) --------------
            # Compares gradients between pred_x0 and x_start (ground truth)
            grad_match_w = float(loss_cfg.get("grad_match_weight", 0.0))
            if grad_match_w > 0.0:
                # Compute pred_x0 if not already computed
                if sup_w <= 0.0 and smooth_w <= 0.0:
                    if self.model_mean_type == ModelMeanType.START_X:
                        pred_x0 = model_output
                    elif self.model_mean_type == ModelMeanType.EPSILON:
                        pred_x0 = self._predict_xstart_from_eps(x_t, t, model_output)
                    else:  # PREVIOUS_X
                        pred_x0 = self._predict_xstart_from_xprev(x_t, t, model_output)
                
                grad_match_type = loss_cfg.get("grad_match_type", "l1")  # "l1" or "l2"
                ndim = pred_x0.dim()
                
                if ndim == 3:  # 1D case: [B, C, L]
                    grad_pred = pred_x0[:, :, 1:] - pred_x0[:, :, :-1]
                    grad_gt = x_start[:, :, 1:] - x_start[:, :, :-1]
                    grad_diff = grad_pred - grad_gt
                elif ndim == 4:  # 2D case: [B, C, H, W]
                    # Horizontal gradient
                    grad_h_pred = pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]
                    grad_h_gt = x_start[:, :, 1:, :] - x_start[:, :, :-1, :]
                    # Vertical gradient
                    grad_w_pred = pred_x0[:, :, :, 1:] - pred_x0[:, :, :, :-1]
                    grad_w_gt = x_start[:, :, :, 1:] - x_start[:, :, :, :-1]
                    grad_diff_h = grad_h_pred - grad_h_gt
                    grad_diff_w = grad_w_pred - grad_w_gt
                else:
                    raise ValueError(f"Unsupported tensor dimension {ndim} for gradient matching")
                
                # Compute loss
                if ndim == 3:
                    if grad_match_type == "l1":
                        grad_match_loss = mean_flat(grad_diff.abs())
                    else:  # l2
                        grad_match_loss = mean_flat(grad_diff ** 2)
                else:  # ndim == 4
                    if grad_match_type == "l1":
                        grad_match_loss = 0.5 * (mean_flat(grad_diff_h.abs()) + mean_flat(grad_diff_w.abs()))
                    else:  # l2
                        grad_match_loss = 0.5 * (mean_flat(grad_diff_h ** 2) + mean_flat(grad_diff_w ** 2))
                
                # Time-dependent weighting: reduce weight at high noise levels
                grad_time_weight_type = loss_cfg.get("grad_time_weight", "sqrt_snr")  # "none", "sqrt_snr", "snr", "linear"
                if grad_time_weight_type == "sqrt_snr":
                    # weight = sqrt(alpha_bar_t), higher at small t (less noise)
                    time_weight = _extract_into_tensor(self.sqrt_alphas_cumprod, t, (t.shape[0],))
                elif grad_time_weight_type == "snr":
                    # weight = alpha_bar_t
                    time_weight = _extract_into_tensor(self.alphas_cumprod, t, (t.shape[0],))
                elif grad_time_weight_type == "linear":
                    # weight = (T - t) / T, linearly decreases with t
                    time_weight = (self.num_timesteps - t.float()) / self.num_timesteps
                else:  # "none"
                    time_weight = 1.0
                
                terms["mse"] = terms["mse"] + grad_match_w * time_weight * grad_match_loss
                # terms["grad_match_loss"] = grad_match_loss  # for logging
            # -------------- end gradient matching loss (optional) --------------

            # -------------- start multiscale loss (optional) --------------
            # L = λ₀‖x̂₀ - x₀‖₁ + λ₁‖LP(x̂₀) - LP(x₀)‖₁ + λ₂‖HP(x̂₀) - HP(x₀)‖₁
            # LP = low-pass (Gaussian blur), HP = high-pass (original - LP)
            multiscale_w = float(loss_cfg.get("multiscale_weight", 0.0))
            if multiscale_w > 0.0:
                # Compute pred_x0 if not already computed
                if sup_w <= 0.0 and smooth_w <= 0.0 and grad_match_w <= 0.0:
                    if self.model_mean_type == ModelMeanType.START_X:
                        pred_x0 = model_output
                    elif self.model_mean_type == ModelMeanType.EPSILON:
                        pred_x0 = self._predict_xstart_from_eps(x_t, t, model_output)
                    else:  # PREVIOUS_X
                        pred_x0 = self._predict_xstart_from_xprev(x_t, t, model_output)
                
                # Get multiscale weights (relative within multiscale loss)
                lambda_0 = float(loss_cfg.get("ms_lambda_0", 1.0))  # original scale
                lambda_lp = float(loss_cfg.get("ms_lambda_lp", 1.0))  # low-pass
                lambda_hp = float(loss_cfg.get("ms_lambda_hp", 1.0))  # high-pass
                ms_kernel_size = int(loss_cfg.get("ms_kernel_size", 5))  # Gaussian kernel size
                ms_sigma = float(loss_cfg.get("ms_sigma", 1.5))  # Gaussian sigma
                
                ndim = pred_x0.dim()
                B, C = pred_x0.shape[:2]
                
                # Create Gaussian kernel for low-pass filter
                if ndim == 3:  # 1D case: [B, C, L]
                    # 1D Gaussian kernel
                    x_coord = th.arange(ms_kernel_size, dtype=pred_x0.dtype, device=pred_x0.device)
                    x_coord = x_coord - ms_kernel_size // 2
                    gauss_1d = th.exp(-x_coord ** 2 / (2 * ms_sigma ** 2))
                    gauss_1d = gauss_1d / gauss_1d.sum()
                    gauss_kernel = gauss_1d.view(1, 1, -1).expand(C, -1, -1)
                    
                    # Low-pass filter
                    pad_size = ms_kernel_size // 2
                    pred_lp = F.conv1d(F.pad(pred_x0, (pad_size, pad_size), mode='reflect'), 
                                       gauss_kernel, groups=C)
                    gt_lp = F.conv1d(F.pad(x_start, (pad_size, pad_size), mode='reflect'), 
                                     gauss_kernel, groups=C)
                    
                elif ndim == 4:  # 2D case: [B, C, H, W]
                    # 2D Gaussian kernel (separable)
                    x_coord = th.arange(ms_kernel_size, dtype=pred_x0.dtype, device=pred_x0.device)
                    x_coord = x_coord - ms_kernel_size // 2
                    gauss_1d = th.exp(-x_coord ** 2 / (2 * ms_sigma ** 2))
                    gauss_1d = gauss_1d / gauss_1d.sum()
                    gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
                    gauss_kernel = gauss_2d.view(1, 1, ms_kernel_size, ms_kernel_size).expand(C, -1, -1, -1)
                    
                    # Low-pass filter
                    pad_size = ms_kernel_size // 2
                    pred_lp = F.conv2d(F.pad(pred_x0, (pad_size, pad_size, pad_size, pad_size), mode='reflect'), 
                                       gauss_kernel, groups=C)
                    gt_lp = F.conv2d(F.pad(x_start, (pad_size, pad_size, pad_size, pad_size), mode='reflect'), 
                                     gauss_kernel, groups=C)
                else:
                    raise ValueError(f"Unsupported tensor dimension {ndim} for multiscale loss")
                
                # High-pass = original - low-pass
                pred_hp = pred_x0 - pred_lp
                gt_hp = x_start - gt_lp
                
                # Compute losses at each scale (L1)
                loss_original = mean_flat((pred_x0 - x_start).abs())
                loss_lp = mean_flat((pred_lp - gt_lp).abs())
                loss_hp = mean_flat((pred_hp - gt_hp).abs())
                
                multiscale_loss = lambda_0 * loss_original + lambda_lp * loss_lp + lambda_hp * loss_hp
                
                # Time-dependent weighting: reduce weight at high noise levels
                ms_time_weight_type = loss_cfg.get("ms_time_weight", "sqrt_snr")  # "none", "sqrt_snr", "snr", "linear"
                if ms_time_weight_type == "sqrt_snr":
                    time_weight = _extract_into_tensor(self.sqrt_alphas_cumprod, t, (t.shape[0],))
                elif ms_time_weight_type == "snr":
                    time_weight = _extract_into_tensor(self.alphas_cumprod, t, (t.shape[0],))
                elif ms_time_weight_type == "linear":
                    time_weight = (self.num_timesteps - t.float()) / self.num_timesteps
                else:  # "none"
                    time_weight = 1.0
                
                terms["mse"] = terms["mse"] + multiscale_w * time_weight * multiscale_loss
                # terms["multiscale_loss"] = multiscale_loss  # for logging
            # -------------- end multiscale loss (optional) --------------

            # -------------- start HP-based sample weighting (optional) (mj_20260205) --------------
            # Applies per-sample weights based on Hemispheric Power (HP) to rebalance rare high-HP events
            # Weights are precomputed using compute_hp_weights.py and loaded via hp_weights.npz
            hp_sample_weight_enabled = bool(loss_cfg.get("hp_sample_weight", False))
            if hp_sample_weight_enabled:
                if model_kwargs is None or "hp_weight" not in model_kwargs:
                    # HP weights not available; skip weighting
                    pass
                else:
                    # Get HP weights from model_kwargs
                    hp_weights = model_kwargs["hp_weight"]  # shape: [B]
                    
                    # Apply optional scaling and clipping
                    hp_weight_scale = float(loss_cfg.get("hp_weight_scale", 1.0))
                    hp_weight_clip_min = float(loss_cfg.get("hp_weight_clip_min", 0.1))
                    hp_weight_clip_max = float(loss_cfg.get("hp_weight_clip_max", 10.0))
                    
                    hp_weights = hp_weights * hp_weight_scale
                    hp_weights = hp_weights.clamp(hp_weight_clip_min, hp_weight_clip_max)
                    
                    # Normalize to have mean 1 within batch to preserve overall loss magnitude
                    hp_weights = hp_weights / (hp_weights.mean() + 1e-8)
                    
                    # Apply sample-wise weighting to MSE loss
                    # terms["mse"] has shape [B], hp_weights has shape [B]
                    terms["mse"] = terms["mse"] * hp_weights
            # -------------- end HP-based sample weighting (optional) (mj_20260205) --------------

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
