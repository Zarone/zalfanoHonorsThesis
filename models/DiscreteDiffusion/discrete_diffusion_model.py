"""
Discrete Diffusion Language Model Implementation (Token-Space Diffusion)

This module implements a diffusion-based language model that generates tokens
through an iterative unmasking process in discrete token space (LLaDA-style).

Instead of adding Gaussian noise to embeddings (continuous diffusion), this model
corrupts tokens by randomly masking them with a special [MASK] token, then learns
to predict the original tokens from the masked sequence — analogous to BERT but
generalised to a full generative diffusion process.

Reference: "LLaDA: Large Language Diffusion with mAsking" (arxiv 2502.09992)
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class DiscreteDiffusionConfig(PretrainedConfig):
    """Configuration for Discrete Diffusion Language Model."""

    model_type = "discrete_diffusion_lm"

    # Architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5

    # Model-specific
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    n_positions: int = 1024  # Alias for compatibility

    # Diffusion process
    num_diffusion_steps: int = 50
    diffusion_schedule: str = "linear"  # "linear", "sqrt", "cosine"
    mask_token_id: int = -1  # Set to tokenizer.mask_token_id at init time

    # Training
    learning_rate: float = 1e-4
    use_cache: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "n_positions" in kwargs:
            self.max_position_embeddings = kwargs["n_positions"]


# ---------------------------------------------------------------------------
# Masking schedule
# ---------------------------------------------------------------------------

class MaskingSchedule(nn.Module):
    """
    Noise schedule for discrete (masking) diffusion.

    At each timestep t ∈ {1, …, T}, each token is independently masked with
    probability mask_ratio(t).  The schedule maps t → mask_ratio, mirroring
    the role of the noise schedule in continuous diffusion.

    Convention (LLaDA): t=0 → no masking (clean data);  t=T → fully masked.
    The forward process q(x_t | x_0) is therefore:
        P(x_t[i] = MASK | x_0) = mask_ratio(t)
        P(x_t[i] = x_0[i]  | x_0) = 1 - mask_ratio(t)
    """

    def __init__(self, num_steps: int, schedule_type: str = "linear"):
        super().__init__()
        self.num_steps = num_steps
        self.schedule_type = schedule_type

        # mask_ratios[t] = fraction of tokens masked at step t
        # t=0 → 0.0,  t=T-1 → ~1.0
        t = torch.arange(num_steps, dtype=torch.float32)
        if schedule_type == "linear":
            ratios = t / (num_steps - 1)
        elif schedule_type == "sqrt":
            ratios = torch.sqrt(t / (num_steps - 1))
        elif schedule_type == "cosine":
            # cosine schedule: starts slow, ends slow
            ratios = 1.0 - torch.cos((t / (num_steps - 1)) * math.pi * 0.5)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.register_buffer("mask_ratios", ratios)

    def q_sample(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        mask_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: randomly mask tokens according to schedule at step t.

        Args:
            input_ids: Clean token IDs  [batch, seq_len]
            t:         Timestep per sample  [batch]
            mask_token_id: ID of the [MASK] token

        Returns:
            x_t:       Partially-masked token IDs  [batch, seq_len]
            mask:      Boolean mask — True where tokens were masked  [batch, seq_len]
        """
        mask_ratio = self.mask_ratios[t]  # [batch]

        # Broadcast to [batch, seq_len]
        mask_ratio = mask_ratio.unsqueeze(1).expand_as(input_ids)

        # Sample a Bernoulli mask
        mask = torch.bernoulli(mask_ratio).bool()

        # Avoid in-place operations for better MPS memory management:
        # Use torch.where instead of clone + in-place indexing
        x_t = torch.where(mask, torch.full_like(input_ids, mask_token_id), input_ids)

        return x_t, mask


# ---------------------------------------------------------------------------
# Denoising network (same bidirectional transformer architecture)
# ---------------------------------------------------------------------------

class DenoisingNetwork(nn.Module):
    """
    Transformer-based denoising network for discrete diffusion.

    Takes the partially-masked token embeddings + timestep embedding and
    predicts a distribution over the vocabulary for every position — the
    discrete analogue of predicting the noise in continuous diffusion.
    """

    def __init__(self, config: DiscreteDiffusionConfig):
        super().__init__()

        # Pre-layer norm
        self.norm1 = nn.LayerNorm(config.hidden_size)

        # Timestep embedding: scalar t/T → hidden vector added to all positions
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Bidirectional transformer encoder (identical topology to continuous version)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers,
            norm=nn.LayerNorm(config.hidden_size),
        )

        # Output projection back to hidden size (lm_head projects further to vocab)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict clean-token representations from masked-token embeddings.

        Args:
            x_t:          Embeddings of (masked) input tokens  [batch, seq_len, hidden]
            timesteps:    Diffusion step per sample  [batch]
            attention_mask: HuggingFace-style mask (1=attend, 0=ignore)  [batch, seq_len]

        Returns:
            Contextualised hidden states  [batch, seq_len, hidden]
        """
        # Timestep embedding
        t_emb = (timesteps.float() / 1000.0).unsqueeze(1)   # [batch, 1]
        t_emb = self.time_embedding(t_emb)                   # [batch, hidden]
        t_emb = t_emb.unsqueeze(1)                           # [batch, 1, hidden]

        # Add time information to every position
        x = x_t + t_emb

        # Convert HF attention mask → PyTorch src_key_padding_mask
        # (True = *ignore* that position)
        src_key_padding_mask = None
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            src_key_padding_mask = (attention_mask == 0).bool()
        elif attention_mask is not None:
            src_key_padding_mask = attention_mask

        # Transformer with pre-norm
        x = self.norm1(x)
        
        # Apply gradient checkpointing if enabled (and in training mode)
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.transformer),
                x,
                src_key_padding_mask,
                use_reentrant=False,
            )
        else:
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Project and add residual
        x = self.output_projection(x)
        return x_t + x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DiscreteDiffusionLanguageModel(PreTrainedModel):
    """
    Discrete Diffusion Language Model (LLaDA-style).

    The forward/reverse process operates entirely in token space:
    - Forward  (training):   randomly mask tokens at sampled timestep t,
                              predict original tokens via cross-entropy.
    - Reverse  (inference):  start from a fully-masked sequence, iteratively
                              unmask tokens with the highest confidence.

    Architecture mirrors ContinuousDiffusionLanguageModel as closely as
    possible, replacing the Gaussian noise schedule with a masking schedule
    and MSE loss with cross-entropy loss on masked positions.
    """

    config_class = DiscreteDiffusionConfig

    def __init__(self, config: DiscreteDiffusionConfig):
        super().__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size

        # Token embedding layer (shared with lm_head — weight tying)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Masking schedule for diffusion
        self.masking_schedule = MaskingSchedule(
            num_steps=config.num_diffusion_steps,
            schedule_type=config.diffusion_schedule,
        )

        # Denoising network (bidirectional transformer)
        self.denoising_network = DenoisingNetwork(config)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights: embedding ↔ lm_head
        self.lm_head.weight = self.token_embeddings.weight

        self.post_init()

        self._tied_weights_keys = ["lm_head.weight", "token_embeddings.weight"]

    def _tie_weights(self) -> None:
        """Tie weights between embedding and output layers."""
        if hasattr(self, "lm_head") and hasattr(self, "token_embeddings"):
            self.lm_head.weight = self.token_embeddings.weight

    def _create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create position IDs for the input."""
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        return positions.unsqueeze(0).expand(bsz, -1)

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable/disable gradient checkpointing for the denoising network."""
        # Set on the denoising network
        if hasattr(self, "denoising_network"):
            self.denoising_network.gradient_checkpointing = value
        # Also set on any module passed in (for compatibility with Trainer)
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, CausalLMOutputWithPast]:
        """
        Forward pass for discrete diffusion language model.

        During training:
            - Sample timestep t and randomly mask tokens with probability mask_ratio(t)
            - Embed the (partially masked) sequence
            - Predict original tokens at masked positions via cross-entropy

        During inference:
            - Start from a fully-masked sequence
            - Iteratively unmask tokens (greedy or sampling) over T steps

        Args:
            input_ids:      Token IDs  [batch, seq_len]
            attention_mask: Optional HF-style mask  [batch, seq_len]
            labels:         Target token IDs for training  [batch, seq_len]
            timestep:       Override diffusion step (optional, mainly for debugging)
            token_type_ids: Accepted but unused (compatibility)
            past_key_values, use_cache: Accepted but unused (compatibility)
            **kwargs:       Additional arguments for compatibility

        Returns:
            CausalLMOutputWithPast with .loss (training) and/or .logits
        """
        if labels is not None:
            return self._training_forward(input_ids, attention_mask, labels, input_ids.device)
        else:
            return self._inference_forward(input_ids, attention_mask, input_ids.device)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def _training_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        device: torch.device,
    ) -> CausalLMOutputWithPast:
        """
        Training forward pass.

        Loss = cross-entropy over masked positions only (like a weighted BERT MLM
        objective averaged over uniformly-sampled timesteps).
        """
        batch_size, seq_len = input_ids.shape

        # 1. Sample random timesteps
        t = torch.randint(
            0, self.config.num_diffusion_steps, (batch_size,), device=device
        )

        # 2. Forward diffusion: mask tokens according to schedule
        mask_token_id = self._get_mask_token_id()
        x_t, token_mask = self.masking_schedule.q_sample(input_ids, t, mask_token_id)
        # x_t:        [batch, seq_len]  — partially masked input
        # token_mask: [batch, seq_len]  — True where token was masked

        # 3. Embed the (partially masked) sequence
        position_ids = self._create_position_ids(x_t)
        x_emb = self.token_embeddings(x_t) + self.position_embeddings(position_ids)
        # [batch, seq_len, hidden]

        # 4. Denoise: predict clean hidden states
        hidden = self.denoising_network(x_emb, t, attention_mask)
        # [batch, seq_len, hidden]

        # 5. Project to logits
        logits = self.lm_head(hidden)
        # [batch, seq_len, vocab_size]

        # 6. Cross-entropy loss on masked positions only
        #    (equivalent to the diffusion ELBO under the masking schedule)
        if token_mask.any():
            masked_logits = logits[token_mask]          # [num_masked, vocab_size]
            masked_labels = labels[token_mask]          # [num_masked]
            loss = nn.functional.cross_entropy(
                masked_logits, masked_labels, reduction="mean"
            )
        else:
            # Fallback: full-sequence CE (rare edge case when t≈0)
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                reduction="mean",
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    # Inference forward
    # ------------------------------------------------------------------

    def _inference_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> CausalLMOutputWithPast:
        """
        Inference forward pass: iterative unmasking (LLaDA-style).

        Starts from a fully-masked sequence and repeatedly unmasks the
        tokens with the highest predicted probability, analogous to DDIM
        reverse diffusion in the continuous case.
        """
        batch_size, seq_len = input_ids.shape
        mask_token_id = self._get_mask_token_id()
        num_steps = self.config.num_diffusion_steps

        # Start from fully-masked sequence (t = T)
        x_t = torch.full_like(input_ids, mask_token_id)

        with torch.no_grad():
            for step in reversed(range(num_steps)):
                t = torch.full((batch_size,), step, dtype=torch.long, device=device)

                # Embed current (partially masked) sequence
                position_ids = self._create_position_ids(x_t)
                x_emb = (
                    self.token_embeddings(x_t)
                    + self.position_embeddings(position_ids)
                )

                # Predict clean token logits
                hidden = self.denoising_network(x_emb, t, attention_mask)
                logits = self.lm_head(hidden)  # [batch, seq_len, vocab_size]

                # Greedy prediction of clean tokens
                predicted_ids = logits.argmax(dim=-1)  # [batch, seq_len]

                # Determine how many tokens should *remain* masked after this step
                ratio_current = self.masking_schedule.mask_ratios[step]
                ratio_prev = (
                    self.masking_schedule.mask_ratios[step - 1]
                    if step > 0
                    else torch.tensor(0.0, device=device)
                )

                # Number of tokens to unmask at this step
                currently_masked = (x_t == mask_token_id)  # [batch, seq_len]
                n_masked = currently_masked.sum(dim=1).float()  # [batch]
                target_remaining = (ratio_prev / (ratio_current + 1e-8)) * n_masked
                n_to_unmask = (n_masked - target_remaining).long().clamp(min=0)

                # For each sample, unmask the n_to_unmask positions with highest confidence
                probs = torch.softmax(logits, dim=-1)                      # [batch, seq_len, V]
                max_probs = probs.max(dim=-1).values                       # [batch, seq_len]

                # Only unmask positions that are currently masked
                max_probs = max_probs.masked_fill(~currently_masked, -1.0)

                for b in range(batch_size):
                    k = n_to_unmask[b].item()
                    if k <= 0:
                        continue
                    topk_indices = torch.topk(max_probs[b], k=int(k)).indices
                    x_t[b, topk_indices] = predicted_ids[b, topk_indices]

        # Final logit computation on the fully-unmasked sequence
        position_ids = self._create_position_ids(x_t)
        x_emb = self.token_embeddings(x_t) + self.position_embeddings(position_ids)
        hidden = self.denoising_network(
            x_emb,
            torch.zeros(batch_size, dtype=torch.long, device=device),
            attention_mask,
        )
        logits = self.lm_head(hidden)

        return CausalLMOutputWithPast(logits=logits)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mask_token_id(self) -> int:
        """Return the mask token ID, falling back to a safe default."""
        mid = self.config.mask_token_id
        if mid < 0:
            raise ValueError(
                "mask_token_id has not been set. "
                "Pass mask_token_id=tokenizer.mask_token_id in the config."
            )
        return mid

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embeddings = value

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens by iterative unmasking of a masked suffix.

        The prompt (input_ids) is kept fixed; only the appended masked
        positions are iteratively unmasked.

        Args:
            input_ids:      Prompt token IDs  [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature:    Sampling temperature
            top_p:          Nucleus sampling threshold

        Returns:
            Generated token IDs  [batch, seq_len + max_new_tokens]
        """
        device = input_ids.device
        batch_size, prompt_len = input_ids.shape
        mask_token_id = self._get_mask_token_id()
        num_steps = self.config.num_diffusion_steps

        # Build the full sequence: prompt + fully-masked generation region
        masked_suffix = torch.full(
            (batch_size, max_new_tokens), mask_token_id,
            dtype=input_ids.dtype, device=device,
        )
        x_t = torch.cat([input_ids, masked_suffix], dim=1)  # [batch, prompt+gen]

        # Attention mask: attend to everything
        attn_mask = torch.ones(
            batch_size, prompt_len + max_new_tokens, dtype=torch.long, device=device
        )

        with torch.no_grad():
            for step in reversed(range(num_steps)):
                t = torch.full((batch_size,), step, dtype=torch.long, device=device)

                position_ids = self._create_position_ids(x_t)
                x_emb = (
                    self.token_embeddings(x_t)
                    + self.position_embeddings(position_ids)
                )
                hidden = self.denoising_network(x_emb, t, attn_mask)
                logits = self.lm_head(hidden)  # [batch, full_len, vocab]

                # Apply temperature
                logits = logits / temperature

                # Top-p filtering
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumprobs > top_p
                remove_mask[..., 0] = False
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(2, sorted_indices, sorted_logits)

                predicted_ids = torch.multinomial(
                    torch.softmax(logits.view(-1, self.config.vocab_size), dim=-1),
                    num_samples=1,
                ).view(batch_size, -1)  # [batch, full_len]

                # Only unmask positions in the *generation region* that are still masked
                gen_mask = (x_t == mask_token_id)
                gen_mask[:, :prompt_len] = False  # never overwrite the prompt

                ratio_current = self.masking_schedule.mask_ratios[step]
                ratio_prev = (
                    self.masking_schedule.mask_ratios[step - 1]
                    if step > 0
                    else torch.tensor(0.0, device=device)
                )
                probs_max = torch.softmax(logits, dim=-1).max(dim=-1).values
                probs_max = probs_max.masked_fill(~gen_mask, -1.0)

                n_masked = gen_mask.sum(dim=1).float()
                n_to_unmask = (
                    n_masked * (1.0 - ratio_prev / (ratio_current + 1e-8))
                ).long().clamp(min=0)

                for b in range(batch_size):
                    k = n_to_unmask[b].item()
                    if k <= 0:
                        continue
                    topk_idx = torch.topk(probs_max[b], k=int(k)).indices
                    x_t[b, topk_idx] = predicted_ids[b, topk_idx]

        return x_t
