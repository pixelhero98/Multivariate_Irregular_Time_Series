import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, ff_mult: int,
                 input_dim: int, enc_dims: list[int] | None = None):
        super().__init__()
        self.d_model = d_model
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, d_model))  # example size; keep yours
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, ff_mult,
                                    enc_dim=(enc_dims[i] if enc_dims is not None and i < len(enc_dims) else None))
            for i in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, input_dim)

    def forward(self, z: torch.Tensor, encoder_skips: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        """
        z:             [B, T, Z] — decoder d_model == Z (latent channels)
        encoder_skips: list of [B, T, D] from encoder (deepest→shallowest), or None to disable skip attention.
                       Each provided skip will be projected D→Z inside the block.
        """
        h = z + self.pos_emb[:, :z.size(1), :]
        # If skips are provided, we consume them deepest→shallowest; else pass None to blocks.
        skips = list(reversed(encoder_skips)) if encoder_skips is not None else [None] * len(self.layers)
        for i, dec_layer in enumerate(self.layers):
            skip = skips[i] if i < len(skips) else None
            h = dec_layer(h, skip=skip)
        return self.out_proj(h)  # [B, T, input_dim]


class LatentVAE(nn.Module):
    def __init__(self,
                 input_dim: int = 1,
                 d_model: int,
                 n_layers_enc: int,
                 n_layers_dec: int,
                 n_heads: int,
                 ff_mult: int,
                 latent_channel: int,
                 enc_dims: list[int],
                 *,
                 use_encoder_skips: bool = False
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_channel = latent_channel
        self.use_encoder_skips = use_encoder_skips

        # ----- Encoder -----
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers_enc,
            n_heads=n_heads,
            ff_mult=ff_mult,
            input_dim=input_dim
        )

        # bottleneck heads (D -> Z)
        self.mu_head = nn.Linear(d_model, latent_channel)
        self.logvar_head = nn.Linear(d_model, latent_channel)

        # ----- Decoder -----
        # decoder d_model is the latent_channel (Z)
        self.decoder = TransformerDecoder(
            d_model=latent_channel,
            n_layers=n_layers_dec,
            n_heads=n_heads,
            ff_mult=ff_mult,
            input_dim=input_dim,
            enc_dims=enc_dims  # used to size skip projections inside blocks
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, *, use_encoder_skips: Optional[bool] = None):
        """
        x: [B, T, input_dim]

        use_encoder_skips:
            - None (default): use the module's configuration flag set at construction time.
            - True:  force-enable encoder skip cross-attention for this call.
            - False: force-disable encoder skip cross-attention for this call.

        Returns:
            x_hat:  [B, T, input_dim]
            mu:     [B, T, Z]
            logvar: [B, T, Z]
        """
        # Encode -> list of hidden states (deepest last), each [B, T, D]
        encoder_hidden_states = self.encoder(x)
        z_last = encoder_hidden_states[-1]                      # [B, T, D]

        # Bottleneck → [B, T, Z]
        mu = self.mu_head(z_last)                               # [B, T, Z]
        logvar = self.logvar_head(z_last).clamp(-10.0, 10.0)    # [B, T, Z]
        z = self.reparameterize(mu, logvar)                     # [B, T, Z]

        # Decide whether to pass skips
        use_skips = self.use_encoder_skips if use_encoder_skips is None else bool(use_encoder_skips)
        skips = encoder_hidden_states if use_skips else None

        # Decode (with or without cross-attention to encoder skips)
        x_hat = self.decoder(z, encoder_skips=skips)            # [B, T, input_dim]
        return x_hat, mu, logvar

