import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from vae.models.components.modules.transformer import RotaryPositionalEmbedding


def _modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Apply an adaptive LayerNorm (AdaLN) affine modulation to the input.

    :param x: Input with shape [B, T, D].
    :type x: torch.Tensor
    :param shift: Additive shift. Must be broadcastable to `x` (typically [B, 1, D]).
    :type shift: torch.Tensor
    :param scale: Multiplicative scale offset. Must be broadcastable to `x` (typically [B, 1, D]).
    :type scale: torch.Tensor

    :returns: Modulated output with the same shape as `x`.
    :rtype: torch.Tensor
    """

    return x * (1.0 + scale) + shift


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding of the flow time t in [0, 1], followed by an MLP.
    """

    def __init__(
        self,
        model_dimension: int,
        frequency_embedding_dimension: int = 256,
        max_period: int = 10_000,
    ):
        """
        Initialise the timestep embedding.

        :param model_dimension: Dimensionality of the output embedding.
        :type model_dimension: int
        :param frequency_embedding_dimension: Dimensionality of the sinusoidal embedding fed to the MLP,
            split evenly between cosine and sine components. Must be even.
        :type frequency_embedding_dimension: int
        :param max_period: Controls the lowest sinusoidal frequency;
            frequencies are geometrically spaced from 1 down towards `1 / max_period`.
        :type max_period: int
        """

        super().__init__()

        self.frequency_embedding_dimension = frequency_embedding_dimension
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dimension, model_dimension),
            nn.SiLU(),
            nn.Linear(model_dimension, model_dimension),
        )

    def forward(
        self,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed flow times.

        :param timestep: Flow times in [0, 1] with shape [B].
        :type timestep: torch.Tensor

        :returns: Timestep embeddings with shape [B, model_dimension].
        :rtype: torch.Tensor
        """

        half = self.frequency_embedding_dimension // 2
        frequencies = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=timestep.device, dtype=torch.float32) / half
        )
        # Scale t from [0, 1] to [0, 1000] to match the integer-timestep range this embedding was designed for
        angles = timestep.float()[:, None] * 1000.0 * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(timestep.dtype)
        return self.mlp(embedding)


class DiTBlock(nn.Module):
    """
    Bidirectional transformer block with AdaLN-zero timestep conditioning.
    """

    def __init__(
        self,
        model_dimension: int,
        num_heads: int,
        feedforward_dimension: int,
        rotary_positional_embedding: RotaryPositionalEmbedding,
    ):
        """
        Initialise the DiT block.

        :param model_dimension: Transformer width. Must be divisible by `num_heads`.
        :type model_dimension: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param feedforward_dimension: Hidden dimension of the feedforward network.
        :type feedforward_dimension: int
        :param rotary_positional_embedding: Rotary positional embedding applied to the queries and keys.
        :type rotary_positional_embedding: RotaryPositionalEmbedding
        """

        super().__init__()

        if model_dimension % num_heads != 0:
            raise ValueError(f"{model_dimension=} must be divisible by {num_heads=}")

        self.num_heads = num_heads
        self.rotary_positional_embedding = rotary_positional_embedding

        self.attention_norm = nn.LayerNorm(model_dimension, elementwise_affine=False, eps=1e-6)
        self.qkv_projection = nn.Linear(model_dimension, 3 * model_dimension)
        self.attention_output_projection = nn.Linear(model_dimension, model_dimension)

        self.feedforward_norm = nn.LayerNorm(model_dimension, elementwise_affine=False, eps=1e-6)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dimension, feedforward_dimension),
            nn.GELU(),
            nn.Linear(feedforward_dimension, model_dimension),
        )

        modulation_projection = nn.Linear(model_dimension, 6 * model_dimension)
        nn.init.zeros_(modulation_projection.weight)
        nn.init.zeros_(modulation_projection.bias)
        self.modulation = nn.Sequential(nn.SiLU(), modulation_projection)

    def _attention(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, model_dimension = x.shape
        queries, keys, values = self.qkv_projection(x).chunk(3, dim=-1)

        shape = (batch_size, sequence_length, self.num_heads, model_dimension // self.num_heads)
        queries, keys, values = (t.view(shape).transpose(1, 2) for t in (queries, keys, values))
        queries, keys = self.rotary_positional_embedding(queries, keys)

        attention = F.scaled_dot_product_attention(queries, keys, values)
        attention = attention.transpose(1, 2).reshape(batch_size, sequence_length, model_dimension)
        return self.attention_output_projection(attention)

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform the input.

        :param x: Input sequence with shape [B, T, model_dimension].
        :type x: torch.Tensor
        :param timestep_embedding: Timestep embeddings with shape [B, model_dimension].
        :type timestep_embedding: torch.Tensor

        :returns: Output sequence with shape [B, T, model_dimension].
        :rtype: torch.Tensor
        """

        modulation = self.modulation(timestep_embedding)[:, None]  # [B, 1, 6 * D]
        attention_shift, attention_scale, attention_gate, feedforward_shift, feedforward_scale, feedforward_gate = (
            modulation.chunk(6, dim=-1)
        )

        x = x + attention_gate * self._attention(_modulate(self.attention_norm(x), attention_shift, attention_scale))

        x = x + feedforward_gate * self.feedforward(
            _modulate(self.feedforward_norm(x), feedforward_shift, feedforward_scale)
        )
        return x


class DiT(nn.Module):
    """
    Diffusion transformer predicting flow-matching velocities for voice restoration.
    """

    def __init__(
        self,
        latent_dimension: int,
        conditioning_dimension: int,
        model_dimension: int,
        num_heads: int,
        num_layers: int,
        feedforward_dimension: int,
    ):
        """
        Initialise the DiT.

        :param latent_dimension: Dimensionality of the (standardized) bottleneck latents being generated.
        :type latent_dimension: int
        :param conditioning_dimension: Dimensionality of the per-frame conditioning features.
        :type conditioning_dimension: int
        :param model_dimension: Transformer width.
        :type model_dimension: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param num_layers: Number of transformer blocks.
        :type num_layers: int
        :param feedforward_dimension: Hidden dimension of each block's feedforward network.
        :type feedforward_dimension: int
        """

        super().__init__()

        self.latent_dimension = latent_dimension
        self.conditioning_dimension = conditioning_dimension

        input_dimension = latent_dimension + conditioning_dimension
        self.input_projection = nn.Linear(input_dimension, model_dimension)
        self.timestep_embedding = TimestepEmbedding(model_dimension)

        rotary_positional_embedding = RotaryPositionalEmbedding()
        self.blocks = nn.ModuleList(
            DiTBlock(
                model_dimension,
                num_heads,
                feedforward_dimension,
                rotary_positional_embedding,
            )
            for _ in range(num_layers)
        )

        self.final_norm = nn.LayerNorm(model_dimension, elementwise_affine=False, eps=1e-6)
        final_modulation_projection = nn.Linear(model_dimension, 2 * model_dimension)
        nn.init.zeros_(final_modulation_projection.weight)
        nn.init.zeros_(final_modulation_projection.bias)
        self.final_modulation = nn.Sequential(nn.SiLU(), final_modulation_projection)

        self.output_projection = nn.Linear(model_dimension, latent_dimension)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the flow-matching velocity.

        :param noisy_latents: Flow state x_t with shape [B, T, latent_dimension].
        :type noisy_latents: torch.Tensor
        :param timestep: Flow times in [0, 1] with shape [B].
        :type timestep: torch.Tensor
        :param conditioning: Conditioning features with shape [B, T, conditioning_dimension].
            Pass zeros to drop conditioning (for classifier-free guidance).
        :type conditioning: torch.Tensor

        :returns: Predicted velocity with shape [B, T, latent_dimension].
        :rtype: torch.Tensor
        """

        x = self.input_projection(torch.cat([noisy_latents, conditioning], dim=-1))
        timestep_embedding = self.timestep_embedding(timestep)

        for block in self.blocks:
            x = block(x, timestep_embedding)

        shift, scale = self.final_modulation(timestep_embedding)[:, None].chunk(2, dim=-1)
        return self.output_projection(_modulate(self.final_norm(x), shift, scale))
