import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from global_summary import ODELaplaceGuidedSummarizer
from cond_diffusion_utils import NoiseScheduler
from pos_time_emb import timestep_embedding
from lapformer import LapFormer


class LLapDiT(nn.Module):
    """
    Latent conditional diffusion model for multivariate time series.
    - Global multi-entity conditioning (learned queries)
    - Positional encodings in context & target
    - Native parameterization throughout ('eps' or 'v')
    """
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        *,
        predict_type: str = 'v',                   # 'eps' or 'v'
        laplace_k: Union[int, List[int]] = 32,
        global_k: int = 64,
        timesteps: int = 1000,
        schedule: str = 'cosine',
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        self_conditioning: bool = False,
        context_dim: Optional[int] = None,
        num_entities: int = None,                   # REQUIRED
        context_len: int = 16,                      # number of learned query tokens in summary
    ):
        super().__init__()
        assert predict_type in ('eps', 'v'), "predict_type must be 'eps' or 'v'"
        if num_entities is None:
            raise ValueError("num_entities must be provided (int) for the global summarizer.")
        self.predict_type = predict_type
        self.self_conditioning = bool(self_conditioning)

        # diffusion utils
        self.scheduler = NoiseScheduler(timesteps=timesteps, schedule=schedule)

        # global context summarizer
        ctx_dim = context_dim if context_dim is not None else data_dim
        self.context = ODELaplaceGuidedSummarizer(
            num_entities=num_entities,
            feat_dim=ctx_dim,
            hidden_dim=hidden_dim,
            out_len=context_len,
            num_heads=num_heads,
            lap_k=global_k,
        )

        # main model
        self.model = LapFormer(
            input_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            laplace_k=laplace_k,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_conditioning=self_conditioning,
        )

        # time embedding dimension equals hidden_dim (processed inside LapFormer)
        self.time_dim = hidden_dim

    # -------------------------------
    # Embeddings & conditioning
    # -------------------------------
    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        te = timestep_embedding(t, self.time_dim)    # [B, H]
        te = F.silu(te)
        return te

    def _maybe_build_cond(
        self,
        series: Optional[torch.Tensor],
        cond_summary: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        ctx_dt: Optional[torch.Tensor] = None,
        ctx_diff: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Build or reuse the global context summary.
        Do NOT use observation/imputation masks here. Padding masks only (if ever needed).
        """
        if cond_summary is not None:
            return cond_summary
        if series is None:
            return None
        # Build summary from historical context only (no imputation mask)
        summary, _ = self.context(series, dt=ctx_dt, ctx_diff=ctx_diff)     # [B, Lq, H]
        return summary

    # -------------------------------
    # U-Net call
    # -------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        series: Optional[torch.Tensor] = None,
        series_mask: Optional[torch.Tensor] = None,     # kept for API compat; ignored here
        cond_summary: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        sc_feat: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        series_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict native param ('eps' or 'v') at timestep t for x_t.
        """
        cond_summary = self._maybe_build_cond(series, cond_summary, entity_ids, series_dt=series_dt)
        t_emb = self._time_embed(t).to(x_t.dtype)
        raw = self.model(x_t, t_emb, cond_summary=cond_summary, sc_feat=sc_feat, dt=dt)
        return raw

    # -------------------------------
    # DDIM sampling with (optional) inpainting and self-conditioning
    # -------------------------------
    @torch.no_grad()
    def generate(
            self,
            shape,
            steps: int = 36,
            guidance_strength=2.0,  # float or (g_min, g_max)
            guidance_power: float = 0.3,  # schedule: (1 - alpha_bar)^power
            eta: float = 0.0,  # DDIM noise (keep 0 for deterministic)
            *,
            series=None,
            series_mask=None,
            cond_summary=None,
            entity_ids=None,
            y_obs: torch.Tensor | None = None,  # [B,L,D] observed values
            obs_mask: torch.Tensor | None = None,  # [B,L]  1=observed, 0=free
            dt: Optional[torch.Tensor] = None,
            series_dt: Optional[torch.Tensor] = None,
            x_T: torch.Tensor | None = None,
            self_cond: bool | None = None,
            cfg_rescale: bool = True,  # guidance stabilization
    ):
        """
        Clean DDIM sampler (v-pred) with:
          - Karras step selection
          - CFG + rescale
          - Mask-aware inpainting
          - Self-conditioning
        Returns x0.
        """
        device = next(self.parameters()).device
        B, L, D = shape
        if self_cond is None:
            self_cond = self.self_conditioning

        # ----- helpers -----
        def _alpha_bar(t_b):
            return self.scheduler._gather(self.scheduler.alpha_bars, t_b).view(B, 1, 1)

        def _karras_step_indices(T: int, n: int, rho: float = 7.0):
            # map continuous Karras sigmas to nearest discrete t, unique & descending
            ab = self.scheduler.alpha_bars.to(device=device)
            sigma_all = torch.sqrt((1.0 - ab) / (ab + 1e-12))  # monotone in t
            n = int(max(1, min(n, T)))
            if n >= T:
                return torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)
            sigma_min = float(sigma_all[0].item())  # t=0
            sigma_max = float(sigma_all[-1].item())  # t=T-1
            i = torch.linspace(0, 1, n, device=device)
            sigmas = (sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            idxs = [int(torch.argmin(torch.abs(sigma_all - s))) for s in sigmas]
            return torch.tensor(sorted(set(idxs), reverse=True), device=device, dtype=torch.long)

        def _cfg(pred_u, pred_c, g_scalar, mask_scale=None):
            # pred_u/pred_c: [B,L,D]; g_scalar: [B,1,1]; mask_scale: [B,L,1] or None
            g = g_scalar
            if mask_scale is not None:
                g = 1.0 + (g - 1.0) * mask_scale  # observed -> 1.0 (no extra push)
            guided = pred_u + g * (pred_c - pred_u)
            if not cfg_rescale:
                return guided
            # Rescale: match per-sample mean/std to conditional branch
            reduce_dims = (1, 2)
            mu_c = pred_c.mean(dim=reduce_dims, keepdim=True)
            std_c = pred_c.std(dim=reduce_dims, keepdim=True) + 1e-6
            mu_g = guided.mean(dim=reduce_dims, keepdim=True)
            std_g = guided.std(dim=reduce_dims, keepdim=True) + 1e-6
            return (guided - mu_g) * (std_c / std_g) + mu_c

        # ----- prepare -----
        x_t = torch.randn(B, L, D, device=device) if x_T is None else x_T.to(device)
        summary = self._maybe_build_cond(series, cond_summary, entity_ids, series_dt=series_dt)

        T = int(self.scheduler.timesteps)
        step_indices = _karras_step_indices(T, int(steps))
        ts_prev = torch.cat([step_indices[1:], step_indices.new_tensor([-1])])  # land at x0

        obs = obs_mask.to(device) if obs_mask is not None else None
        tar_scale = (1.0 - obs).unsqueeze(-1) if obs is not None else None  # [B,L,1]

        # project initial x_T to respect observations at the first time index
        if (y_obs is not None) and (obs is not None):
            t0_b = step_indices[0].repeat(B)
            x_T_obs, _ = self.scheduler.q_sample(y_obs.to(device=device, dtype=x_t.dtype), t0_b)
            x_t = obs * x_T_obs + (1.0 - obs) * x_t

        sc_feat_next = None
        last_x0 = None

        # ----- main loop -----
        for t_i, t_prev_i in zip(step_indices, ts_prev):
            t_b = t_i.repeat(B)

            # unconditional / conditional predictions (reuse self-conditioning)
            pred_u = self.forward(
                x_t, t_b,
                series=None, series_mask=None, cond_summary=None,
                sc_feat=sc_feat_next if self_cond else None,
                dt=dt, series_dt=series_dt,
            )
            pred_c = self.forward(
                x_t, t_b,
                series=series, series_mask=series_mask,
                cond_summary=summary, entity_ids=entity_ids,
                sc_feat=sc_feat_next if self_cond else None,
                dt=dt, series_dt=series_dt,
            )

            # scheduled guidance g_t = g_min + (g_max-g_min) * (1 - ab)^p
            ab_t = _alpha_bar(t_b)  # [B,1,1]
            if isinstance(guidance_strength, (tuple, list)):
                g_min, g_max = float(guidance_strength[0]), float(guidance_strength[1])
            else:
                g_min, g_max = 1.0, float(guidance_strength)
            w = torch.pow(1.0 - ab_t, guidance_power)
            g_t = g_min + (g_max - g_min) * w

            pred = _cfg(pred_u, pred_c, g_t, mask_scale=tar_scale)

            # x0 estimate (v/eps handled in scheduler)
            last_x0 = self.scheduler.to_x0(x_t, t_b, pred, param_type=self.predict_type)

            # teacher for next step
            sc_feat_next = last_x0.detach() if self_cond else None

            # take DDIM step (or finish at x0)
            if int(t_prev_i) < 0:
                x_t = last_x0
            else:
                tprev_b = t_prev_i.repeat(B)
                x_t = self.scheduler.ddim_step_from(
                    x_t, t_b, tprev_b, pred,
                    param_type=self.predict_type,
                    eta=eta,
                )

            # enforce observations at new time level
            if (y_obs is not None) and (obs is not None):
                if int(t_prev_i) >= 0:
                    t_inpaint = t_prev_i.repeat(B)
                    x_obs_t, _ = self.scheduler.q_sample(y_obs.to(device=device, dtype=x_t.dtype), t_inpaint)
                    x_t = obs * x_obs_t + (1.0 - obs) * x_t
                else:
                    x_t = obs * y_obs.to(device=device, dtype=x_t.dtype) + (1.0 - obs) * x_t

        return last_x0 if last_x0 is not None else x_t
