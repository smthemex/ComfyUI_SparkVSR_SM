"""CogVideoX scheduler math, reimplemented in pure torch.

Why this exists:
  The official SparkVSR script does ONE denoising step. It calls
      pipe.scheduler.get_velocity(predicted_noise, lq_sample, timesteps=[399])
  DDPM's get_velocity(sample, noise, t) computes
      sqrt(alpha_t) * noise - sqrt(1 - alpha_t) * sample
  and SparkVSR deliberately passes (v, x_t) into (sample, noise), so the
  result is the predicted x_0:
      x_0 = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v

  ComfyUI's V_PREDICTION_DDPM.calculate_denoised is numerically identical
  (verified: sigma(399) = 1.2413748503 vs sqrt((1-a)/a) = 1.241374867),
  but we keep our own alpha table so the plugin is immune to ComfyUI
  scheduler changes and can reproduce the official `noise_step` add_noise
  and the CFG-on-velocity behaviour exactly.
"""

import torch

# scheduler/scheduler_config.json:
#   beta_schedule = scaled_linear, beta_start = .00085, beta_end = .012,
#   num_train_timesteps = 1000, rescale_betas_zero_snr = true


def _make_alphas_cumprod(n=1000, beta_start=0.00085, beta_end=0.012):
    # diffusers "scaled_linear" == what ComfyUI names "linear"
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=torch.float64) ** 2
    ac = torch.cumprod(1.0 - betas, dim=0)
    # rescale_zero_terminal_snr (diffusers.rescale_betas_zero_snr)
    s = ac.sqrt()
    s0, sT = s[0], s[-1]
    s = (s - sT) * (s0 / (s0 - sT))
    return (s ** 2).float()


ALPHAS_CUMPROD = _make_alphas_cumprod()


def _reshape(a, ref):
    return a.view(-1, *([1] * (ref.ndim - 1))).to(device=ref.device, dtype=ref.dtype)


def alpha_at(timesteps, ref):
    """alpha_cumprod[t] shaped for broadcasting against `ref`."""
    t = torch.as_tensor(timesteps, device="cpu").long().clamp(0, len(ALPHAS_CUMPROD) - 1)
    a = ALPHAS_CUMPROD[t].to(device=ref.device, dtype=ref.dtype)
    return _reshape(a, ref)


def get_velocity(model_output, sample, timesteps):
    """Official `scheduler.get_velocity(v, x_t, t)` -> predicted x_0."""
    a = alpha_at(timesteps, sample)
    return a.sqrt() * sample - (1.0 - a).sqrt() * model_output


def add_noise(x, noise, timesteps):
    """Official `scheduler.add_noise(x_0, eps, t)` -> sqrt(a)*x + sqrt(1-a)*eps."""
    a = alpha_at(timesteps, x)
    return a.sqrt() * x + (1.0 - a).sqrt() * noise


def sigma_from_alpha(timesteps, ref):
    a = alpha_at(timesteps, ref)
    return ((1.0 - a) / a).sqrt()


def sigma_for_timestep(model_sampling, timestep, device):
    """Sigma whose ModelSamplingDiscrete.timestep() maps back to `timestep`.

    Verified on this ComfyUI build: sigma(399) -> timestep == 399.
    Falls back to the diffusers alpha table if a future ComfyUI changes the
    schedule (we only need the DiT to receive the right integer timestep).
    """
    t = torch.tensor([float(timestep)], device=device)
    try:
        sigma = model_sampling.sigma(t)
        back = model_sampling.timestep(sigma)
        if int(back.reshape(-1)[0].item()) == int(timestep):
            return sigma
    except Exception:
        pass
    ref = torch.zeros(1, device=device)
    return sigma_from_alpha([int(timestep)], ref).reshape(1)


class RawVelocitySampling(torch.nn.Module):
    """Drop-in replacement for `BaseModel.model_sampling`.

    Makes BaseModel.apply_model return the RAW model output (velocity / eps)
    instead of a denoised sample, while delegating everything else (timestep
    mapping, sigmas table, noise_scaling ...) to the original object.

    It MUST subclass nn.Module: BaseModel is an nn.Module and PyTorch refuses
    to assign a plain object to a child-module attribute. Assigning another
    nn.Module also keeps `_modules` bookkeeping valid.
    """

    def __init__(self, inner):
        super().__init__()
        object.__setattr__(self, "_inner", inner)

    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output

    def timestep(self, sigma):
        return self._inner.timestep(sigma)

    def sigma(self, timestep):
        return self._inner.sigma(timestep)

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_inner"), name)
