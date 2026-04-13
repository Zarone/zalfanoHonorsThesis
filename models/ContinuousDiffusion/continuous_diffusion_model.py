"""
Continuous Diffusion Language Model Implementation (Embedding-Space Diffusion)

This module implements a diffusion-based language model that generates token embeddings
through an iterative denoising process in continuous embedding space (DDIM-style).
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class ContinuousDiffusionConfig(PretrainedConfig):
    """Configuration for Continuous Diffusion Language Model."""
    
    model_type = "continuous_diffusion_lm"
    
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
    noise_scale: float = 1.0
    
    # Training
    learning_rate: float = 1e-4
    use_cache: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure n_positions syncs with max_position_embeddings
        if 'n_positions' in kwargs:
            self.max_position_embeddings = kwargs['n_positions']


class NoiseSchedule(nn.Module):
    """Learnable or fixed noise schedule for diffusion."""
    
    def __init__(self, num_steps: int, schedule_type: str = "linear", device: str = "cuda"):
        super().__init__()
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.device = device
        
        # Create noise schedule
        if schedule_type == "linear":
            betas = torch.linspace(0.0001, 0.02, num_steps)
        elif schedule_type == "sqrt":
            betas = torch.linspace(0.0001, 0.02, num_steps) ** 0.5
        elif schedule_type == "cosine":
            # Cosine schedule from improved DDPM
            s = 0.008
            steps = torch.arange(num_steps + 1)
            alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Register buffers for alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Precomputed values for efficiency
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise to embeddings."""
        sqrt_alphas_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for batch operations
        while len(sqrt_alphas_t.shape) < len(x_0.shape):
            sqrt_alphas_t = sqrt_alphas_t.unsqueeze(-1)
            sqrt_one_minus_alphas_t = sqrt_one_minus_alphas_t.unsqueeze(-1)
        
        return sqrt_alphas_t * x_0 + sqrt_one_minus_alphas_t * noise


class EmbeddingProjection(nn.Module):
    """Project vocabulary tokens to continuous embedding space."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Project token IDs to embeddings."""
        return self.embedding(input_ids)


class DenoisingNetwork(nn.Module):
    """Transformer-based denoising network (U-Net style architecture)."""
    
    def __init__(self, config: ContinuousDiffusionConfig):
        super().__init__()
        
        # Pre-layer norm architecture
        self.norm1 = nn.LayerNorm(config.hidden_size)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        
        # Transformer encoder layers
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
        
        # Output projection to embedding dimension
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoise embeddings conditioned on timestep.
        
        Args:
            x_t: Noisy embeddings [batch, seq_len, hidden_dim]
            timesteps: Timestep for each sample [batch]
            attention_mask: Optional attention mask (1 = attend, 0 = ignore)
            
        Returns:
            Denoised embeddings [batch, seq_len, hidden_dim]
        """
        batch_size = x_t.shape[0]
        
        # Add time embedding
        t_emb = (timesteps.float() / 1000.0).unsqueeze(1)  # [batch, 1]
        t_emb = self.time_embedding(t_emb)  # [batch, hidden_size]
        t_emb = t_emb.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Add time information to all positions
        x = x_t + t_emb
        
        # Convert attention mask from HuggingFace format to PyTorch format
        # HF: 1 = attend, 0 = ignore
        # PT: True = ignore (mask), False = attend
        src_key_padding_mask = None
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Invert: where attention_mask is 0, set to True (mask those positions)
            src_key_padding_mask = (attention_mask == 0).bool()
        elif attention_mask is not None:
            src_key_padding_mask = attention_mask
        
        # Apply transformer with pre-norm
        x = self.norm1(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Project to output
        x = self.output_projection(x)
        
        # Return predicted noise (no residual—we predict the noise, not the denoised embedding)
        return x


class ContinuousDiffusionLanguageModel(PreTrainedModel):
    """
    Continuous Diffusion Language Model.
    
    Uses an iterative denoising process to generate token embeddings in continuous space,
    which are then projected to vocabulary logits.
    """
    
    config_class = ContinuousDiffusionConfig
    
    def __init__(self, config: ContinuousDiffusionConfig):
        super().__init__(config)
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Token embedding layer
        self.token_embeddings = EmbeddingProjection(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Noise schedule for diffusion
        self.noise_schedule = NoiseSchedule(
            num_steps=config.num_diffusion_steps,
            schedule_type=config.diffusion_schedule,
        )
        
        # Denoising network
        self.denoising_network = DenoisingNetwork(config)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings with output projection
        self.lm_head.weight = self.token_embeddings.embedding.weight
        
        self.post_init()
        
        # Register tied weights keys for proper serialization
        self._tied_weights_keys = ["lm_head.weight", "token_embeddings.embedding.weight"]
    
    def _tie_weights(self) -> None:
        """Tie weights between embedding and output layers."""
        if hasattr(self, "lm_head") and hasattr(self, "token_embeddings"):
            self.lm_head.weight = self.token_embeddings.embedding.weight
    
    def _create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create position IDs for the input."""
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
        )
        return positions.unsqueeze(0).expand(bsz, -1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, CausalLMOutputWithPast]:
        """
        Forward pass for continuous diffusion language model.
        
        During training:
            - Encode input tokens to embeddings
            - Sample timesteps and noise
            - Apply forward diffusion (add noise)
            - Denoise with the network
            - Compute loss between denoised embs and target
        
        During inference:
            - Start from noise
            - Iteratively denoise the full sequence
            - Decode to token logits
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len]
            labels: Target token IDs for training
            timestep: Which diffusion step (for training)
            token_type_ids: Optional token type IDs (accepted but not used)
            past_key_values: For compatibility with transformers
            use_cache: For compatibility with transformers
            **kwargs: Additional arguments for compatibility
            
        Returns:
            Loss (training) or logits (inference)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get position IDs
        position_ids = self._create_position_ids(input_ids)
        
        # Get target embeddings from input tokens
        x_0 = self.token_embeddings(input_ids)  # [batch, seq_len, hidden_size]
        pos_emb = self.position_embeddings(position_ids)
        x_0 = x_0 + pos_emb
        
        # Always compute loss if labels are provided (for training)
        if labels is not None:
            # Training mode: use teacher forcing and diffusion loss
            return self._training_forward(
                x_0, attention_mask, labels, device
            )
        else:
            # Inference mode: iterative denoising
            return self._inference_forward(x_0, attention_mask, device)
    
    def _iterative_denoise(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Iterative denoising for evaluation: full T-step DDIM process.
        Measures actual perplexity by evaluating the model's full output.
        
        Returns:
            Denoised embeddings [batch, seq_len, hidden_size]
        """
        # Start from pure noise
        x_t = torch.randn(batch_size, seq_len, self.hidden_size, device=device) * self.config.noise_scale
        
        with torch.no_grad():
            for step in reversed(range(self.config.num_diffusion_steps)):
                t = torch.full((batch_size,), step, dtype=torch.long, device=device)
                
                # Predict noise at this timestep
                predicted_noise = self.denoising_network(x_t, t, attention_mask)
                
                # DDIM reverse step
                alpha_t = self.noise_schedule.alphas_cumprod[step]
                alpha_prev = self.noise_schedule.alphas_cumprod_prev[step]
                
                sqrt_alpha_t = math.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = math.sqrt(1.0 - alpha_t)
                sqrt_alpha_prev = math.sqrt(alpha_prev)
                sqrt_one_minus_alpha_prev = math.sqrt(1.0 - alpha_prev)
                
                # Predict x_0
                x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
                
                # Compute x_{t-1}
                x_t = (
                    sqrt_alpha_prev * x_0_pred +
                    sqrt_one_minus_alpha_prev * predicted_noise
                )
        
        return x_t
    
    def _training_forward(
        self,
        x_0: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        device: torch.device,
    ):
        """Training forward pass with diffusion loss and evaluation perplexity."""
        batch_size, seq_len = x_0.shape[:2]
        
        # Sample random timesteps
        t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)
        
        # Sample random noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        x_t = self.noise_schedule.q_sample(x_0, t, noise)
        
        # Denoise: predict the noise
        predicted_noise = self.denoising_network(x_t, t, attention_mask)
        
        # Compute diffusion loss (predict noise) — training objective
        diffusion_loss = torch.nn.functional.mse_loss(predicted_noise, noise, reduction='mean')
        
        # For evaluation perplexity: do full iterative denoising (expensive but accurate)
        x_0_denoised = self._iterative_denoise(batch_size, seq_len, device, attention_mask)
        
        # Compute token prediction loss on fully denoised embeddings (true perplexity)
        logits = self.lm_head(x_0_denoised)  # [batch, seq_len, vocab_size]
        
        # Flatten for loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute cross-entropy loss — this is the perplexity metric
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        # Combine losses for optimization: primary is diffusion (MSE noise prediction), secondary is CE
        # But ce_loss now reflects true generalization, not noisy embeddings
        total_loss = diffusion_loss + 0.1 * ce_loss
        
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
        )
    
    def _inference_forward(
        self,
        x_0: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ):
        """Inference forward pass with iterative denoising."""
        batch_size, seq_len = x_0.shape[:2]
        
        # Start with pure noise
        x_t = torch.randn_like(x_0) * self.config.noise_scale
        
        # Iteratively denoise
        with torch.no_grad():
            for step in reversed(range(self.config.num_diffusion_steps)):
                t = torch.full((batch_size,), step, dtype=torch.long, device=device)
                
                # Predict noise
                predicted_noise = self.denoising_network(x_t, t, attention_mask)
                
                # One step of DDIM reverse diffusion
                # x_{t-1} = sqrt(alpha_{t-1}) * (x_t - sqrt(1-alpha_t)*eps_t) / sqrt(alpha_t) + sqrt(1-alpha_{t-1}) * eps_t
                alpha_t = self.noise_schedule.alphas_cumprod[step]
                alpha_prev = self.noise_schedule.alphas_cumprod_prev[step]
                
                sqrt_alpha_t = math.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = math.sqrt(1.0 - alpha_t)
                sqrt_alpha_prev = math.sqrt(alpha_prev)
                sqrt_one_minus_alpha_prev = math.sqrt(1.0 - alpha_prev)
                
                # Predict x_0
                x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
                
                # Compute x_{t-1}
                x_t = (
                    sqrt_alpha_prev * x_0_pred +
                    sqrt_one_minus_alpha_prev * predicted_noise
                )
        
        # Project to logits
        logits = self.lm_head(x_t)
        
        return CausalLMOutputWithPast(logits=logits)
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input embedding layer."""
        return self.token_embeddings.embedding
    
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input embedding layer."""
        self.token_embeddings.embedding = value
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        for _ in range(max_new_tokens):
            # Forward pass with iterative denoising
            with torch.no_grad():
                outputs = self.forward(input_ids)
                logits = outputs.logits  # [batch, seq_len, vocab_size]
            
            # Get logits for the last position
            next_logits = logits[:, -1, :] / temperature
            
            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False  # Keep at least one
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[:, indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids
