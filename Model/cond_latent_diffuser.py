import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from global_summary import PDELaplaceGuidedSummarizer
from cond_diffusion_utils import NoiseScheduler
from pos_time_emb import timestep_embedding
from lapformer import LapFormer


class LLapDiT(nn.Module):
    """
    Latent conditional diffusion model for multivariate time series.
    - Global multi-entity conditioning
    - Positional encodings in context & target
    - Native parameterization throughout ('eps' or 'v')
    - Self-conditioning (optional)
    - Multi-resolution Laplace (optional via list)
    """
    def __init__(self,
                 data_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 laplace_k: Union[int, List[int]] = 16,   # int or list: [k_stem, k1, k2, ...]
                 global_k: int = 32,
                 num_entities: Optional[int] = None,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 predict_type: str = "eps",               # "eps" or "v"
                 timesteps: int = 1000,
                 schedule: str = "cosine",
                 self_conditioning: bool = False,
                 context_dim: int = None):
        super().__init__()
        assert predict_type in {"eps", "v"}
        self.predict_type = predict_type
        self.self_conditioning = self_conditioning

        self.scheduler = NoiseScheduler(timesteps=timesteps, schedule=schedule)
        ctx_dim = context_dim if context_dim is not None else data_dim
        self.context = PDELaplaceGuidedSummarizer(
            lap_k=global_k,
            feat_dim=ctx_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_entities=num_entities,
        )

        self.model = LapFormer(
            input_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            laplace_k=laplace_k,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_conditioning=self_conditioning
        )

        self.time_dim = hidden_dim

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        te = timestep_embedding(t, self.time_dim)
        te = F.silu(te)
        return te

    def _maybe_build_cond(self,
                          series: Optional[torch.Tensor],
                          series_mask: Optional[torch.Tensor],
                          cond_summary: Optional[torch.Tensor],
                          entity_ids: Optional[torch.Tensor]):
        if cond_summary is not None:
            return cond_summary, None
        if series is None:
            return None, None
        ctx = self.context(series, context_mask=series_mask)
        return ctx, None

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                series: Optional[torch.Tensor] = None,
                series_mask: Optional[torch.Tensor] = None,
                cond_summary: Optional[torch.Tensor] = None,
                entity_ids: Optional[torch.Tensor] = None,
                sc_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns native-param prediction: [B, L, D] in 'eps' or 'v'
        """
        cond_summary, cond_mask = self._maybe_build_cond(series, series_mask, cond_summary, entity_ids)
        t_emb = self._time_embed(t)
        raw = self.model(x_t, t_emb, cond_summary, cond_mask, sc_feat=sc_feat)
        return raw

    @torch.no_grad()
    def generate(self,
                 shape: Tuple[int, int, int],
                 steps: int = 50,
                 guidance_strength: Union[float, Tuple[float, float]] = 1.5,
                 guidance_power: float = 0.3,
                 eta: float = 0.0,
                 series: Optional[torch.Tensor] = None,
                 series_mask: Optional[torch.Tensor] = None,
                 cond_summary: Optional[torch.Tensor] = None,
                 entity_ids: Optional[torch.Tensor] = None,
                 y_obs: Optional[torch.Tensor] = None,
                 obs_mask: Optional[torch.Tensor] = None,
                 x_T: Optional[torch.Tensor] = None,
                 self_cond: Optional[bool] = None) -> torch.Tensor:
        """
        DDIM sampling with classifier-free guidance (scheduled) and optional inpainting.
        Guidance scheduling:
          - if guidance_strength is float G: g_t = 1 + (G-1) * (1 - alpha_bar_t)^{guidance_power}
          - if (g_min,g_max): g_t = g_min + (g_max-g_min) * (1 - alpha_bar_t)^{guidance_power}
        """
        device = next(self.parameters()).device
        B, L, D = shape
        if self_cond is None:
            self_cond = self.self_conditioning # Default to whatever the model was trained with
        x_t = torch.randn(B, L, D, device=device) if x_T is None else x_T.to(device)

        built_cond, built_mask = self._maybe_build_cond(series, series_mask, cond_summary, entity_ids)
        cond_summary = built_cond
        cond_mask = built_mask

        if (y_obs is not None) and (obs_mask is not None):
            y_obs = y_obs.to(device)
            obs_mask = obs_mask.to(device)

        total_T = self.scheduler.timesteps
        steps = max(1, min(steps, total_T))
        step_indices = torch.linspace(0, total_T - 1, steps, device=device).round().long().flip(0)
        ts_prev = torch.cat([step_indices[1:], step_indices[-1:]])

        for t_i, t_prev_i in zip(step_indices, ts_prev):
            t_b = t_i.repeat(B)
            tprev_b = t_prev_i.repeat(B)

            # optional self-conditioning (teacher is previous prediction of x0)
            sc_feat_uncond = sc_feat_cond = None
            if self.self_conditioning and self_cond:
                with torch.no_grad():
                    # self-cond for UNCONDITIONAL
                    pred_sc_u = self.forward(x_t, t_b, series=None, series_mask=None, cond_summary=None)
                    x0_sc_u = self.scheduler.to_x0(x_t, t_b, pred_sc_u, param_type=self.predict_type)
                    sc_feat_uncond = x0_sc_u
            
                    # self-cond for CONDITIONAL
                    pred_sc_c = self.forward(x_t, t_b, series=series, series_mask=series_mask,
                                             cond_summary=cond_summary, entity_ids=entity_ids)
                    x0_sc_c = self.scheduler.to_x0(x_t, t_b, pred_sc_c, param_type=self.predict_type)
                    sc_feat_cond = x0_sc_c
            
            pred_uncond = self.forward(x_t, t_b, series=None, series_mask=None, cond_summary=None,
                                       sc_feat=sc_feat_uncond)
            pred_cond   = self.forward(x_t, t_b, series=series, series_mask=series_mask,
                                       cond_summary=cond_summary, entity_ids=entity_ids,
                                       sc_feat=sc_feat_cond)

            # scheduled classifier-free guidance
            ab_t = self.scheduler._gather(self.scheduler.alpha_bars, t_b)  # [B]
            if isinstance(guidance_strength, (tuple, list)):
                g_min, g_max = guidance_strength
            else:
                g_min, g_max = 1.0, float(guidance_strength)
            w = torch.pow(1.0 - ab_t, guidance_power).view(-1, 1, 1)
            g_t = g_min + (g_max - g_min) * w  # [B,1,1]

            pred = pred_uncond + g_t * (pred_cond - pred_uncond)

            # DDIM step with native param
            x_t = self.scheduler.ddim_step_from(x_t, t_b, tprev_b, pred, param_type=self.predict_type, eta=eta)

            if (y_obs is not None) and (obs_mask is not None):
                x_t_obs, _ = self.scheduler.q_sample(y_obs, t_b)
                x_t = obs_mask * x_t_obs + (1.0 - obs_mask) * x_t

        return x_t  # x_0