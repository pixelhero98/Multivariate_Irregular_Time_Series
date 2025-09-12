
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from Model.global_summary import UnifiedGlobalSummarizer
from Model.cond_diffusion_utils import NoiseScheduler
from Model.pos_time_emb import timestep_embedding
from Model.lapformer import LapFormer


class LLapDiT(nn.Module):
    """Latent Laplace-DiT for multivariate time series with global conditioning.

    - Uses UnifiedGlobalSummarizer for multi-entity context (mode tied to LapFormer).
    - LapFormer does per-target temporal modeling in parallel or recurrent Laplace mode.
    """
    def __init__(self,
                 data_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_heads: int,
                 *,
                 predict_type: str = 'v',
                 laplace_k: Union[int, List[int]] = 32,
                 global_k: int = 64,
                 timesteps: int = 1000,
                 schedule: str = 'cosine',
                 dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 self_conditioning: bool = False,
                 context_dim: Optional[int] = None,
                 num_entities: int = None,
                 context_len: int = 16,
                 lap_mode: str = 'parallel'):
        super().__init__()
        assert predict_type in ('eps', 'v')
        if num_entities is None:
            raise ValueError("num_entities must be provided (int) for the global summarizer.")
        self.predict_type = predict_type
        self.self_conditioning = bool(self_conditioning)

        # diffusion utils (not used in forward path tests)
        self.scheduler = NoiseScheduler(timesteps=timesteps, schedule=schedule)

        # global context summarizer; choose simple vs full via lap_mode
        ctx_dim = context_dim if context_dim is not None else data_dim
        self.context = UnifiedGlobalSummarizer(
            lap_mode=lap_mode,
            num_entities=num_entities,
            feat_dim=ctx_dim,
            hidden_dim=hidden_dim,
            out_len=context_len,
            num_heads=num_heads,
            lap_k=global_k,
            dropout=dropout,
            add_guidance_tokens=True,
        )

        # main LapFormer (mode tied to summarizer)
                     
        self.model = LapFormer(
            input_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            laplace_k=laplace_k,
            lap_mode=lap_mode,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_conditioning=self_conditioning,
        )

        self.time_dim = hidden_dim  # time embedding dimension

    # -------------------------------
    # Embeddings & conditioning
    # -------------------------------
    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        te = timestep_embedding(t, self.time_dim)    # [B, H]
        return F.silu(te)

    def _maybe_build_cond(self, series, cond_summary=None, entity_ids=None,
                          ctx_dt=None, ctx_diff=None, entity_mask=None):
        if cond_summary is not None or series is None:
            return cond_summary
        # Pass dt, ctx_diff, and entity_mask so summarizer can be dt-/mask-aware
        summary, _ = self.context(series, pad_mask=None, dt=ctx_dt,
                                  ctx_diff=ctx_diff, entity_mask=entity_mask)
        return summary

    # -------------------------------
    # Forward call
    # -------------------------------
    def forward(self,
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
                series_diff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict native param ('eps' or 'v') at timestep t for x_t."""
        cond_summary = self._maybe_build_cond(
            series,
            cond_summary=cond_summary,
            entity_ids=entity_ids,
            ctx_dt=series_dt,
            ctx_diff=series_diff,
            entity_mask=series_mask
        )
        t_emb = self._time_embed(t).to(x_t.dtype)
        # Pass dt to LapFormer (only used when lap_mode='recurrent')
        raw = self.model(x_t, t_emb, cond_summary=cond_summary, sc_feat=sc_feat, dt=dt)
        return raw
                    
    # -------------------------------
    # Sampling
    # -------------------------------
    
    @torch.no_grad()
    def generate(
        self,
        shape,
        steps: int = 36,
        guidance_strength=2.0,
        guidance_power: float = 0.3,
        eta: float = 0.0,
        *,
        series=None,
        series_mask=None,
        cond_summary=None,
        entity_ids=None,
        y_obs: torch.Tensor = None,
        obs_mask: torch.Tensor = None,
        dt: Optional[torch.Tensor] = None,
        series_dt: Optional[torch.Tensor] = None,
        series_diff: Optional[torch.Tensor] = None,
        cfg_rescale: bool = True,
        self_cond: bool = False,
        rho: float = 7.0,
        dynamic_thresh_p: float = 0.0,
        dynamic_thresh_max: float = 1.0,
    ):
        """
        Efficient DDIM sampler with Karras time selection and optional CFG rescale.
        Returns x0 prediction. Keeps inpainting & self-conditioning support.
        """
        device = next(self.parameters()).device
        B, L, D = shape
        T = int(self.scheduler.timesteps)
    
        # ---- build context once (dt/mask/diff aware) ----
        if (series is not None) and (cond_summary is None):
            cond_summary, _ = self.context(
                series, pad_mask=None, dt=series_dt, ctx_diff=series_diff, entity_mask=series_mask
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
    
        def _cfg(pred_u: torch.Tensor, pred_c: torch.Tensor, g_scalar: torch.Tensor, mask_scale=None) -> torch.Tensor:
            guided = pred_u + g_scalar * (pred_c - pred_u)
            if not cfg_rescale:
                return guided
            reduce_dims = (1, 2)
            mu_c = pred_c.mean(dim=reduce_dims, keepdim=True)
            std_c = pred_c.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            mu_g = guided.mean(dim=reduce_dims, keepdim=True)
            std_g = guided.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            guided = (guided - mu_g) / std_g * std_c + mu_c
            if mask_scale is not None:
                # Smoothly reduce over-guidance on observed tokens
                guided = pred_u + (1.0 + (g_scalar - 1.0) * mask_scale) * (pred_c - pred_u)
            return guided
    
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
                x_t, t_b,
                series=None, series_mask=None, cond_summary=None,
                sc_feat=sc_feat_next if self_cond else None,
                entity_ids=entity_ids, dt=dt, series_dt=series_dt, series_diff=series_diff
            )
            pred_c = self.forward(
                x_t, t_b,
                series=series, series_mask=series_mask, cond_summary=cond_summary,
                sc_feat=sc_feat_next if self_cond else None,
                entity_ids=entity_ids, dt=dt, series_dt=series_dt, series_diff=series_diff
            )
    
            # classifier-free guidance (optionally scheduled)
            if isinstance(guidance_strength, (tuple, list)):
                g_min, g_max = guidance_strength
                ab = _alpha_bar_batched(t_b)  # [B,1,1]
                g_scalar = g_min + (g_max - g_min) * (1.0 - ab) ** guidance_power
            else:
                g_scalar = torch.as_tensor(float(guidance_strength), device=device).view(1, 1, 1).expand(B, 1, 1)
    
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
                x_t = self.scheduler.ddim_step_from(x_t, t_b, tprev_b, pred, param_type=self.predict_type, eta=eta)
            else:
                x_t = x0_hat
    
            # keep observed values consistent across steps
            if (y_obs is not None) and (obs_u is not None):
                if int(t_prev_i) >= 0:
                    x_obs_t, _ = self.scheduler.q_sample(y_obs.to(device=device, dtype=x_t.dtype), tprev_b)
                    x_t = obs_u * x_obs_t + (1.0 - obs_u) * x_t
                else:
                    x_t = obs_u * y_obs.to(device=device, dtype=x_t.dtype) + (1.0 - obs_u) * x_t
    
        return last_x0 if last_x0 is not None else x_t

