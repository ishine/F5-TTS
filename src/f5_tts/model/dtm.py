"""
DTM (Distillation Transition Matching) Model.

This module implements the DTM framework for accelerating TTS inference.
It wraps a frozen DiT backbone with a trainable MLP head.

Training follows Algorithm 3 and inference follows Algorithm 4 from the guideline.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.dtm_head import DTMHead
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
)


class DTM(nn.Module):
    """
    DTM (Distillation Transition Matching) model for accelerated TTS inference.
    
    Architecture:
    - Frozen DiT backbone (pretrained)
    - Trainable MLP head
    
    Training (Algorithm 3):
    - Sample discrete timestep t and compute X_t from X_0 and X_T
    - Extract frozen backbone features h_t
    - Sample microscopic time s and compute Y_s
    - Train head to predict velocity field
    
    Inference (Algorithm 4):
    - Loop over T global timesteps
    - For each timestep, solve ODE in microscopic space using head
    - Update global state
    
    Args:
        backbone: Frozen DiT backbone model
        head: Trainable DTM head
        global_timesteps: Number of global timesteps T (for inference)
        ode_solver_steps: Number of substeps for ODE solver
        ode_solver_method: ODE solver method ("euler" or "midpoint")
        mel_spec_module: Mel spectrogram module
        mel_spec_kwargs: Mel spectrogram configuration
        vocab_char_map: Vocabulary character map for tokenization
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        head: DTMHead | None = None,
        global_timesteps: int = 8,
        ode_solver_steps: int = 1,
        ode_solver_method: str = "euler",
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()
        
        # Frozen backbone
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Trainable head
        if head is None:
            head = DTMHead(
                backbone_dim=backbone.dim,
                mel_dim=mel_spec_kwargs.get('n_mel_channels', 100),
            )
        self.head = head
        
        # DTM parameters
        self.global_timesteps = global_timesteps
        self.T = global_timesteps
        self.ode_solver_steps = ode_solver_steps
        self.ode_solver_method = ode_solver_method
        
        # Mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        self.num_channels = self.mel_spec.n_mel_channels
        
        # Vocab map for tokenization
        self.vocab_char_map = vocab_char_map
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def extract_backbone_features(
        self,
        x: torch.Tensor,  # [batch, seq_len, mel_dim]
        cond: torch.Tensor,  # [batch, seq_len, mel_dim]
        text: torch.Tensor,  # [batch, text_len]
        time: torch.Tensor,  # [batch] or scalar
        mask: torch.Tensor | None = None,  # [batch, seq_len]
    ) -> torch.Tensor:
        """
        Extract features from frozen backbone (before final projection).
        
        Returns:
            h_t: Backbone features [batch, seq_len, backbone_dim]
        """
        # Get backbone hidden dimension
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Ensure time is batched
        if time.ndim == 0:
            time = time.repeat(batch_size)
        
        # Get input embeddings (x, cond, text)
        x_embedded = self.backbone.get_input_embed(
            x, cond, text,
            drop_audio_cond=False,
            drop_text=False,
            cache=False,
            audio_mask=mask,
        )
        
        # Get time embedding
        t_emb = self.backbone.time_embed(time)
        
        # Forward through transformer blocks
        rope = self.backbone.rotary_embed.forward_from_seq_len(seq_len)
        
        h = x_embedded
        for block in self.backbone.transformer_blocks:
            h = block(h, t_emb, mask=mask, rope=rope)
        
        # Apply final norm (but not projection)
        h_t = self.backbone.norm_out(h, t_emb)
        
        return h_t  # [batch, seq_len, backbone_dim]
    
    def forward(
        self,
        inp: torch.Tensor,  # mel or raw wave [batch, seq_len, mel_dim] or [batch, wave_len]
        text: torch.Tensor | list[str],  # [batch, text_len]
        *,
        lens: torch.Tensor | None = None,  # [batch]
    ):
        """
        Training forward pass (Algorithm 3).
        
        Args:
            inp: Input mel spectrogram or raw waveform
            text: Text input (tokens or strings)
            lens: Sequence lengths
        
        Returns:
            loss: MSE loss
            cond: Conditioning (for logging)
            pred: Predicted velocity field (for logging)
        """
        # Handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        
        batch_size, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
        
        # Handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch_size
        
        # Lens and mask
        if not exists(lens):
            lens = torch.full((batch_size,), seq_len, device=device)
        
        mask = lens_to_mask(lens, length=seq_len)
        
        # X_T is the real mel spectrogram
        X_T = inp
        
        # X_0 is Gaussian noise
        X_0 = torch.randn_like(X_T)
        
        # Sample discrete timestep t uniformly from {1, ..., T-1}
        # We use continuous time in [0, 1] for compatibility with backbone
        t = torch.randint(1, self.T, (batch_size,), device=device, dtype=torch.long)
        t_continuous = t.float() / self.T  # Normalize to [0, 1]
        
        # Compute X_t = (1 - t/T) * X_0 + (t/T) * X_T
        t_ratio = t.float().unsqueeze(-1).unsqueeze(-1) / self.T  # [batch, 1, 1]
        X_t = (1 - t_ratio) * X_0 + t_ratio * X_T
        
        # Extract frozen backbone features h_t
        # For conditioning, we use zero (unconditional)
        cond = torch.zeros_like(X_T)
        
        with torch.no_grad():
            h_t = self.extract_backbone_features(
                x=X_t,
                cond=cond,
                text=text,
                time=t_continuous,
                mask=mask,
            )
        
        # Compute target displacement Y = X_T - X_0
        Y = X_T - X_0
        
        # Sample microscopic time s uniformly from [0, 1]
        s = torch.rand((batch_size,), device=device, dtype=dtype)
        
        # Sample microscopic noise
        Y_noise = torch.randn_like(Y)
        
        # Compute Y_s = (1 - s) * Y_noise + s * Y
        s_expand = s.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        Y_s = (1 - s_expand) * Y_noise + s_expand * Y
        
        # Forward through trainable head
        v_pred = self.head(h_t, Y_s, s)
        
        # Compute target velocity: Y - Y_noise
        v_target = Y - Y_noise
        
        # Compute MSE loss with mask
        loss = F.mse_loss(v_pred, v_target, reduction='none')  # [batch, seq_len, mel_dim]
        
        # Apply mask to ignore padded regions
        if exists(mask):
            mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            loss = loss * mask_expanded
            loss = loss.sum() / (mask_expanded.sum() * self.num_channels)
        else:
            loss = loss.mean()
        
        return loss, cond, v_pred
    
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,  # [batch, cond_len, mel_dim] or [batch, wave_len]
        text: torch.Tensor | list[str],  # [batch, text_len]
        duration: int | torch.Tensor,  # target duration
        *,
        lens: torch.Tensor | None = None,  # [batch]
        steps: int | None = None,  # override global_timesteps
        seed: int | None = None,
        max_duration: int = 4096,
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        Inference sampling (Algorithm 4).
        
        Args:
            cond: Conditioning mel spectrogram or waveform
            text: Text input
            duration: Target duration
            lens: Conditioning lengths
            steps: Number of global timesteps (override self.T)
            seed: Random seed
            max_duration: Maximum duration
            vocoder: Vocoder for converting mel to waveform
        
        Returns:
            out: Generated mel spectrogram [batch, duration, mel_dim]
            trajectory: Trajectory of generation (for visualization)
        """
        self.eval()
        
        # Handle raw wave conditioning
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        
        cond = cond.to(next(self.parameters()).dtype)
        
        batch_size, cond_seq_len, device = *cond.shape[:2], cond.device
        
        if not exists(lens):
            lens = torch.full((batch_size,), cond_seq_len, device=device, dtype=torch.long)
        
        # Handle text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch_size
        
        # Handle duration
        cond_mask = lens_to_mask(lens)
        
        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        
        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()
        
        # Pad conditioning
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        
        if batch_size > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None
        
        # Use specified steps or default global_timesteps
        T = steps if steps is not None else self.T
        
        # Initialize X_0 with Gaussian noise
        X = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            X.append(torch.randn(dur, self.num_channels, device=device, dtype=cond.dtype))
        X = pad_sequence(X, padding_value=0, batch_first=True)  # [batch, max_dur, mel_dim]
        
        # Algorithm 4: Loop over global timesteps
        trajectory = [X.clone()]
        
        for t in range(T):
            # Current global time (normalized to [0, 1])
            t_continuous = torch.full((batch_size,), t / T, device=device, dtype=cond.dtype)
            
            # Extract backbone features h_t
            h_t = self.extract_backbone_features(
                x=X,
                cond=step_cond,
                text=text,
                time=t_continuous,
                mask=mask,
            )
            
            # Solve ODE for Y using the head
            # dy/ds = head(h_t, y_s, s)
            # Initial condition: Y_0 ~ N(0, I)
            Y_0 = torch.randn_like(X)
            
            # Define ODE dynamics
            def ode_fn(s, y):
                # s is scalar time, y is [batch, seq_len, mel_dim]
                s_batch = torch.full((batch_size,), s.item(), device=device, dtype=cond.dtype)
                return self.head(h_t, y, s_batch)
            
            # Solve ODE from s=0 to s=1
            s_span = torch.linspace(0, 1, self.ode_solver_steps + 1, device=device, dtype=cond.dtype)
            Y_trajectory = odeint(
                ode_fn,
                Y_0,
                s_span,
                method=self.ode_solver_method,
            )
            Y_final = Y_trajectory[-1]  # [batch, seq_len, mel_dim]
            
            # Update global state: X_{t+1} = X_t + (1/T) * Y_final
            X = X + (1.0 / T) * Y_final
            
            trajectory.append(X.clone())
        
        # Final output
        out = X
        out = torch.where(cond_mask, cond, out)  # Keep conditioning part
        
        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        
        trajectory = torch.stack(trajectory, dim=0)  # [T+1, batch, seq_len, mel_dim]
        
        return out, trajectory

