
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
    def generate(self, shape,
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
                 self_cond: bool = True,
                 # --- new knobs ---
                 rho: float = 7.0,                 # Karras schedule sharpness
                 tail_refine_steps: int = 0,       # extra indices inserted near t≈0
                 dynamic_thresh_p: float = 0.0,    # 0 disables; e.g., 0.995 to enable
                 dynamic_thresh_max: float = 1.0   # clamp after rescale
                 ):
        device = next(self.parameters()).device
        B, L, D = shape
    
        def _alpha_bar(t_b):
            return self.scheduler._gather(self.scheduler.alpha_bars, t_b).view(B, 1, 1)
    
        # --- step index picker: Karras + optional tail densification ---
        def _karras_step_indices(T: int, n: int, rho: float, tail_refine_steps: int):
            ab = self.scheduler.alpha_bars.to(device=device)
            sigma_all = torch.sqrt((1.0 - ab) / (ab + 1e-12))     # [T], increasing with t
            n = int(max(1, min(n, T)))
            if n >= T:
                base = torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)
            else:
                sigma_min = float(sigma_all[0].item())
                sigma_max = float(sigma_all[-1].item())
                i = torch.linspace(0, 1, n, device=device)
                sigmas = (sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
                idxs = [int(torch.argmin(torch.abs(sigma_all - s))) for s in sigmas]
                base = torch.tensor(sorted(set(idxs), reverse=True), device=device, dtype=torch.long)
    
            # densify last ~10% of steps (closest to x0) by inserting midpoints
            if tail_refine_steps > 0 and base.numel() >= 2:
                k = max(1, int(round(0.10 * base.numel())))
                tail = base[-k:].tolist()
                mids = []
                for a, b in zip(tail[:-1], tail[1:]):
                    if a != b:
                        mids.append((a + b) // 2)
                # pad/crop mids to requested amount
                if len(mids) < tail_refine_steps:
                    # repeat mids to reach the count (if very few tail gaps)
                    reps = (tail_refine_steps + max(1, len(mids)) - 1) // max(1, len(mids))
                    mids = (mids * reps)[:tail_refine_steps]
                else:
                    mids = mids[:tail_refine_steps]
                merged = sorted(set(base.tolist() + mids), reverse=True)
                base = torch.tensor(merged, device=device, dtype=torch.long)
            return base
    
        def _cfg(pred_u, pred_c, g_scalar, mask_scale=None):
            g = g_scalar
            if mask_scale is not None:
                g = 1.0 + (g - 1.0) * mask_scale
            guided = pred_u + g * (pred_c - pred_u)
            if not cfg_rescale:
                return guided
            reduce_dims = (1, 2)
            mu_c = pred_c.mean(dim=reduce_dims, keepdim=True)
            std_c = pred_c.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            mu_g = guided.mean(dim=reduce_dims, keepdim=True)
            std_g = guided.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            return (guided - mu_g) / std_g * std_c + mu_c
    
        # --- imagen-style dynamic thresholding ---
        def _dynamic_threshold(x0: torch.Tensor, p: float, max_val: float) -> torch.Tensor:
            if p <= 0.0:
                return x0
            x_flat = x0.view(x0.size(0), -1).abs()
            s = torch.quantile(x_flat, q=p, dim=1, interpolation="linear").clamp_min(1.0)  # [B]
            s = s.view(-1, 1, 1)
            return (x0 / s).clamp_(-max_val, max_val)
    
        x_t = torch.randn(B, L, D, device=device)
        sc_feat_next = torch.zeros_like(x_t) if self_cond else None
    
        # build (or reuse) conditioning
        if (series is not None) and (cond_summary is None):
            cond_summary = self.context(series, pad_mask=None, dt=series_dt, entity_mask=series_mask)[0]
    
        T = int(self.scheduler.timesteps)
        step_indices = _karras_step_indices(T, int(steps), rho=rho, tail_refine_steps=int(tail_refine_steps))
        ts_prev = torch.cat([step_indices[1:], step_indices.new_tensor([-1])])
    
        obs = obs_mask if obs_mask is not None else None
        if obs is not None:
            obs = obs.to(device=device, dtype=x_t.dtype)
            obs_u = obs.unsqueeze(-1)
            tar_scale = (1.0 - obs).unsqueeze(-1)
        else:
            obs_u = None
            tar_scale = None
    
        if (y_obs is not None) and (obs_u is not None):
            t0_b = step_indices[0].repeat(B)
            x_T_obs, _ = self.scheduler.q_sample(y_obs.to(device=device, dtype=x_t.dtype), t0_b)
            x_t = obs_u * x_T_obs + (1.0 - obs_u) * x_t
    
        last_x0 = None
        for t_i, t_prev_i in zip(step_indices, ts_prev):
            t_b = t_i.repeat(B)
    
            # unconditional / conditional passes
            pred_u = self.forward(x_t, t_b, series=None, series_mask=None, cond_summary=None,
                                  sc_feat=sc_feat_next if self_cond else None,
                                  entity_ids=entity_ids, dt=dt, series_dt=series_dt, series_diff=series_diff)
            pred_c = self.forward(x_t, t_b, series=series, series_mask=series_mask, cond_summary=cond_summary,
                                  sc_feat=sc_feat_next if self_cond else None,
                                  entity_ids=entity_ids, dt=dt, series_dt=series_dt, series_diff=series_diff)
    
            # classifier-free guidance (optionally rescaled)
            ab = _alpha_bar(t_b)
            if isinstance(guidance_strength, (tuple, list)):
                g_min, g_max = guidance_strength
                g_scalar = g_min + (g_max - g_min) * (1.0 - ab) ** guidance_power
            else:
                g_scalar = torch.as_tensor(float(guidance_strength), device=device).view(1, 1, 1)
            g_scalar = g_scalar.expand(B, 1, 1)
            pred = _cfg(pred_u, pred_c, g_scalar, mask_scale=tar_scale)
    
            # predict x0
            x0_hat = self.scheduler.to_x0(x_t, t_b, pred, param_type=self.predict_type)
    
            # dynamic threshold (if enabled)
            x0_hat = _dynamic_threshold(x0_hat, dynamic_thresh_p, dynamic_thresh_max)
    
            last_x0 = x0_hat
            if self_cond:
                sc_feat_next = x0_hat.detach()
    
            # time update
            if int(t_prev_i) >= 0:
                tprev_b = t_prev_i.repeat(B)
                x_t = self.scheduler.ddim_step_from(x_t, t_b, tprev_b, pred, param_type=self.predict_type, eta=eta)
            else:
                x_t = x0_hat
    
            # honor observations (inpainting)
            if (y_obs is not None) and (obs_u is not None):
                if int(t_prev_i) >= 0:
                    t_inpaint = t_prev_i.repeat(B)
                    x_obs_t, _ = self.scheduler.q_sample(y_obs.to(device=device, dtype=x_t.dtype), t_inpaint)
                    x_t = obs_u * x_obs_t + (1.0 - obs_u) * x_t
                else:
                    x_t = obs_u * y_obs.to(device=device, dtype=x_t.dtype) + (1.0 - obs_u) * x_t
    
        return last_x0 if last_x0 is not None else x_t
