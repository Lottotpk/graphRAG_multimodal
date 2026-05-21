"""
Image Embedding module for InternVL3_5-8B

Implements six strategies based on transformer hidden states:
- last_layer_token_level
- last_layer_cls_token
- last_layer_mean_pooling
- -2_layer_token_level
- -2_layer_cls_token
- -2_layer_mean_pooling

Usage: import in CLI to embed description texts from JSON.
"""

from typing import List, Dict, Any, Tuple
import torch


class ImgEmbedder:
    def __init__(self, model, device: str = 'cuda', max_length: int = 384):
        self.model = model
        self.device = device
        self.max_length = max_length
        self.model.eval()

    @torch.no_grad()
    def _forward_hidden_states(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run the vision model to obtain hidden states.

        Returns:
            hidden_states: list of tensors with shape (B, L, D)
        """
        inputs_embeds = self.model.extract_feature(pixel_values=pixel_values)
        outputs = self.model.language_model(inputs_embeds=inputs_embeds, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden_states: List[torch.Tensor] = outputs.hidden_states  # list length = num_layers+1 (includes embeddings)
        return hidden_states

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pool over sequence dimension with attention mask.
        x: (B, L, D), mask: (B, L)
        Returns (B, D)
        """
        mask = mask.unsqueeze(-1)  # (B, L, 1)
        x = x * mask
        lengths = mask.sum(dim=1).clamp_min(1.0)  # (B, 1)
        return x.sum(dim=1) / lengths

    def _select_layer(self, hidden_states: List[torch.Tensor], which: str) -> torch.Tensor:
        """
        Select hidden state layer tensor (B, L, D).
        which: 'last' or 'minus2'
        """
        if which == 'last':
            return hidden_states[-1]
        elif which in ['-2', 'minus2', 'penultimate']:
            return hidden_states[-2]
        else:
            raise ValueError(f"Unknown layer selector: {which}")

    def get_embedding_dimension(self, pixel_values: torch.Tensor) -> int:
        # Run one mini batch to detect dimension
        hidden_states = self._forward_hidden_states(pixel_values[:1])
        dim = hidden_states[-1].shape[-1]
        return int(dim)

    def embed(self, pixel_values: torch.Tensor, strategy: str) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Compute embeddings for texts using the given strategy.

        Returns:
            embeddings: list of tensors (either (D,) or (T, D) per sample)
            per_sample_metadata: list of dicts; for token-level, will contain token indices
        """
        hidden_states = self._forward_hidden_states(pixel_values)

        layer_sel = 'last'
        if strategy.startswith('-2_layer') or strategy.startswith('penultimate'):
            layer_sel = 'minus2'
        layer_states = self._select_layer(hidden_states, layer_sel)  # (B, L, D)

        pooled = []
        metadata = None

        if strategy.endswith('token_level'):
            # Return per-token embeddings for each sample; exclude padding positions
            for i in range(layer_states.shape[0]):
                token_embeddings = layer_states[i]  # (L, D)
                pooled.append(token_embeddings.detach().cpu())
            metadata = {'token_level': True}
        elif strategy.endswith('cls_token'):
            # First non-pad token as CLS
            for i in range(layer_states.shape[0]):
                # Find first valid token index
                cls_vec = layer_states[i, -1]
                pooled.append(cls_vec.detach().cpu())
            metadata = {'token_level': False, 'pooled': 'cls'}
        elif strategy.endswith('mean_pooling'):
            mean_vec = layer_states.mean(dim=1)  # (B, D)
            for i in range(mean_vec.shape[0]):
                pooled.append(mean_vec[i].detach().cpu())
            metadata = {'token_level': False, 'pooled': 'mean'}
        else:
            raise ValueError(f"Unknown text embedding strategy: {strategy}")

        return pooled, metadata