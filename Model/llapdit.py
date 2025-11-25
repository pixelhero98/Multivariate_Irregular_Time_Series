"""Latent Laplace-DiT diffusion model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.lapformer import LapFormer
from Model.llapdit_utils import NoiseScheduler
from Model.pos_time_emb import timestep_embedding


class LLapDiT(nn.Module):
    """Latent Laplace-DiT for multivariate time series with global conditioning.

    The model consumes noisy inputs together with optional external summaries
    (e.g. from a frozen encoder) and predicts the clean ``x0`` target by
    default. Alternative parameterisations (``eps`` or ``v``) remain supported
    via ``predict_type``.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        *,
        predict_type: str = "x0",
        laplace_k: Union[int, List[int]] = 32,
        timesteps: int = 1000,
        schedule: str = "cosine",
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        self_conditioning: bool = False,
        lap_mode_main: str = "effective",
    ) -> None:
        super().__init__()
        if predict_type not in {"eps", "v", "x0"}:
            raise ValueError("predict_type must be either 'eps', 'v', or 'x0'")
            
        self.predict_type = predict_type
        self.self_conditioning = bool(self_conditioning)

        # Diffusion scheduler utilities (not exercised in forward-only tests).
        self.scheduler = NoiseScheduler(timesteps=timesteps, schedule=schedule)

        # Main LapFormer backbone (mode tied to external summariser if any).
        self.model = LapFormer(
            input_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            laplace_k=laplace_k,
            lap_mode=lap_mode_main,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_conditioning=self_conditioning,
        )

        self.time_dim = hidden_dim  # time embedding dimension

    # -------------------------------
    # Embeddings & conditioning
    # -------------------------------
    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Return a SiLU-activated sinusoidal embedding for diffusion steps."""

        te = timestep_embedding(t, self.time_dim)  # [B, H]
        return F.silu(te)

    # -------------------------------
    # Forward call
    # -------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        series: Optional[torch.Tensor] = None,
        series_mask: Optional[torch.Tensor] = None,
        cond_summary: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        sc_feat: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        series_dt: Optional[torch.Tensor] = None,
        series_diff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict ``eps`` or ``v`` for a batch of noisy inputs ``x_t``.

        The method delegates temporal reasoning to :class:`LapFormer`.  Only the
        externally supplied ``cond_summary`` is consumed for conditioning; the
        legacy in-module summarisation hooks now raise informative errors.
        """

        if cond_summary is None:
            if any(
                arg is not None
                for arg in (series, series_mask, entity_ids, series_dt, series_diff)
            ):
                raise ValueError(
                    "LLapDiT no longer builds conditioning internally. "
                    "Provide `cond_summary` from a pre-trained summarizer."
                )
        t_emb = self._time_embed(t).to(x_t.dtype)
        # Pass dt to LapFormer (only used when lap_mode='recurrent')
        return self.model(
            x_t,
            t_emb,
            cond_summary=cond_summary,
            sc_feat=sc_feat,
            dt=dt,
        )

    # -------------------------------
    # Sampling
    # -------------------------------
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, int, int],
        steps: int = 36,
        guidance_strength=2.0,
        guidance_power: float = 2.0,
        eta: float = 0.0,
        *,
        series: Optional[torch.Tensor] = None,
        series_mask: Optional[torch.Tensor] = None,
        cond_summary: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        y_obs: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        series_dt: Optional[torch.Tensor] = None,
        series_diff: Optional[torch.Tensor] = None,
        cfg_rescale: bool = True,
        self_cond: bool = False,
        rho: float = 7.5,
        dynamic_thresh_p: float = 0.0,
        dynamic_thresh_max: float = 1.0,
    ) -> torch.Tensor:
        """Sample trajectories with DDIM+Karras schedule and optional CFG.

        Args:
            shape: Tuple ``(B, L, D)`` describing the desired sample batch.
            steps: Number of denoising steps to take (``<= timesteps``).
            guidance_strength: CFG strength (scalar or ``(min, max)`` schedule).
            guidance_power: Exponent for scheduled CFG strength.
            eta: DDIM stochasticity parameter.
            series / series_mask / cond_summary / entity_ids: Optional conditioning
                data. ``cond_summary`` must be provided when any of the others are.
            y_obs / obs_mask: Optional observations for inpainting.
            dt / series_dt / series_diff: Optional step-size metadata forwarded to
                the Laplace backbone.
            cfg_rescale: Whether to apply the CFG rescaling heuristic.
            self_cond: Enable self-conditioning during sampling.
            rho: Karras sigma schedule exponent.
            dynamic_thresh_p / dynamic_thresh_max: Parameters for dynamic
                thresholding of ``x0``.

        Returns:
            The final ``x0`` prediction corresponding to the denoised samples.
        """

        device = next(self.parameters()).device
        B, L, D = shape
        T = int(self.scheduler.timesteps)

        # ---- build context once (dt/mask/diff aware) ----
        if cond_summary is None and any(
            arg is not None for arg in (series, series_mask, series_dt, series_diff, entity_ids)
        ):
            raise ValueError(
                "LLapDiT.generate requires `cond_summary` from the frozen summarizer; "
                "internal summarizer construction has been removed."
            )

        # ---- vectorized Karras step indices (descending) ----
        ab = self.scheduler.alpha_bars.to(device)
        sigmas = torch.sqrt((1.0 - ab) / (ab + 1e-12))  # monotonically increasing w.r.t. t
        n = int(max(1, min(steps, T)))
        if n >= T:
            step_indices = torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)
        else:
            smin, smax = sigmas[0].item(), sigmas[-1].item()
            i = torch.linspace(0, 1, n, device=device)
            target = (smax ** (1 / rho) + i * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
            idx = torch.searchsorted(sigmas, target).clamp(max=T - 1)
            idxm = (idx - 1).clamp(min=0)
            pick_lower = (torch.abs(sigmas[idxm] - target) <= torch.abs(sigmas[idx] - target))
            idx = torch.where(pick_lower, idxm, idx)
            step_indices = torch.flip(torch.unique(idx, sorted=True), dims=[0])

        ts_prev = torch.cat([step_indices[1:], step_indices.new_tensor([-1])])

        # ---- helpers ----
        def _alpha_bar_batched(t_b: torch.Tensor) -> torch.Tensor:
            return self.scheduler._gather(self.scheduler.alpha_bars, t_b).view(B, 1, 1)

        def _cfg(
            pred_u: torch.Tensor,
            pred_c: torch.Tensor,
            g_scalar: torch.Tensor,
            mask_scale: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if mask_scale is not None:
                g_eff = (1.0 + (g_scalar - 1.0) * mask_scale).to(pred_u.dtype)
            else:
                g_eff = g_scalar.to(pred_u.dtype)
            guided = pred_u + g_eff * (pred_c - pred_u)

            if not cfg_rescale:
                return guided
            reduce_dims = (1, 2)

            mu_c = pred_c.mean(dim=reduce_dims, keepdim=True)
            std_c = pred_c.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            mu_g = guided.mean(dim=reduce_dims, keepdim=True)
            std_g = guided.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)

            return (guided - mu_g) / std_g * std_c + mu_c

        def _dynamic_threshold(x0: torch.Tensor, p: float, max_val: float) -> torch.Tensor:
            if p <= 0.0:
                return x0
            s = torch.quantile(x0.reshape(B, -1).abs(), q=p, dim=1).clamp_min(1.0).view(B, 1, 1)
            return (x0 / s).clamp_(-max_val, max_val)

        # ---- init state (+ optional inpainting obs at t=T) ----
        x_t = torch.randn(B, L, D, device=device)
        sc_feat_next = torch.zeros_like(x_t) if self_cond else None

        obs_u = None
        tar_scale = None
        if obs_mask is not None:
            obs = obs_mask.to(device=device, dtype=x_t.dtype)
            obs_u = obs.unsqueeze(-1)
            tar_scale = (1.0 - obs).unsqueeze(-1)

        if (y_obs is not None) and (obs_u is not None):
            t0_b = step_indices[0].expand(B)
            x_T_obs, _ = self.scheduler.q_sample(y_obs.to(device=device, dtype=x_t.dtype), t0_b)
            x_t = obs_u * x_T_obs + (1.0 - obs_u) * x_t

        # ---- main loop ----
        last_x0 = None
        for t_i, t_prev_i in zip(step_indices, ts_prev):
            t_b = torch.full((B,), int(t_i.item()), device=device, dtype=torch.long)

            # unconditional / conditional passes
            pred_u = self.forward(
                x_t,
                t_b,
                series=None,
                series_mask=None,
                cond_summary=None,
                sc_feat=sc_feat_next if self_cond else None,
                entity_ids=entity_ids,
                dt=dt,
                series_dt=series_dt,
                series_diff=series_diff,
            )
            pred_c = self.forward(
                x_t,
                t_b,
                series=series,
                series_mask=series_mask,
                cond_summary=cond_summary,
                sc_feat=sc_feat_next if self_cond else None,
                entity_ids=entity_ids,
                dt=dt,
                series_dt=series_dt,
                series_diff=series_diff,
            )

            # classifier-free guidance (optionally scheduled)
            if isinstance(guidance_strength, (tuple, list)):
                g_min, g_max = guidance_strength
                ab_b = _alpha_bar_batched(t_b)  # [B,1,1]
                g_min_t = torch.as_tensor(g_min, device=device, dtype=ab_b.dtype)
                g_max_t = torch.as_tensor(g_max, device=device, dtype=ab_b.dtype)
                g_scalar = g_min_t + (g_max_t - g_min_t) * (1.0 - ab_b) ** guidance_power
            else:
                g_scalar = (
                    torch.as_tensor(float(guidance_strength), device=device)
                    .view(1, 1, 1)
                    .expand(B, 1, 1)
                )

            pred = _cfg(pred_u, pred_c, g_scalar, mask_scale=tar_scale)

            # predict x0
            x0_hat = self.scheduler.to_x0(x_t, t_b, pred, param_type=self.predict_type)

            # optional dynamic thresholding (disabled by default)
            x0_hat = _dynamic_threshold(x0_hat, dynamic_thresh_p, dynamic_thresh_max)

            last_x0 = x0_hat
            if self_cond:
                sc_feat_next = x0_hat.detach()

            # time update
            if int(t_prev_i) >= 0:
                tprev_b = torch.full((B,), int(t_prev_i.item()), device=device, dtype=torch.long)
                x_t = self.scheduler.ddim_step_from(
                    x_t,
                    t_b,
                    tprev_b,
                    pred,
                    param_type=self.predict_type,
                    eta=eta,
                )
            else:
                x_t = x0_hat

            # keep observed values consistent across steps
            if (y_obs is not None) and (obs_u is not None):
                if int(t_prev_i) >= 0:
                    x_obs_t, _ = self.scheduler.q_sample(
                        y_obs.to(device=device, dtype=x_t.dtype), tprev_b
                    )
                    x_t = obs_u * x_obs_t + (1.0 - obs_u) * x_t
                else:
                    x_t = obs_u * y_obs.to(device=device, dtype=x_t.dtype) + (1.0 - obs_u) * x_t

        return last_x0 if last_x0 is not None else x_t

