"""
Text Embedding module for InternVL3_5-8B

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


class TextEmbedder:
    def __init__(self, model, tokenizer, device: str = 'cuda', max_length: int = 384):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    @torch.no_grad()
    def _forward_hidden_states(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Tokenize and run the language model to obtain hidden states.

        Returns:
            input_ids (B, L), attention_mask (B, L), hidden_states: list of tensors with shape (B, L, D)
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Try common attribute names for LM; fallback to calling model directly
        kwargs = dict(output_hidden_states=True, use_cache=False)
        outputs = None
        for attr in ['language_model', 'llm', 'model', 'backbone', None]:
            try:
                if attr is None:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                else:
                    lm = getattr(self.model, attr)
                    outputs = lm(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                break
            except Exception as e:
                print(e)
                continue
        if outputs is None or getattr(outputs, 'hidden_states', None) is None:
            # Last resort: try calling model without named args (rare)
            outputs = self.model(input_ids, attention_mask, **kwargs)
        hidden_states: List[torch.Tensor] = outputs.hidden_states  # list length = num_layers+1 (includes embeddings)
        # Return device-placed tensors to avoid CPU/CUDA mismatch downstream
        return input_ids, attention_mask, hidden_states

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

    def get_embedding_dimension(self, texts: List[str]) -> int:
        # Run one mini batch to detect dimension
        _, _, hidden_states = self._forward_hidden_states(texts[:1])
        dim = hidden_states[-1].shape[-1]
        return int(dim)

    def embed(self, texts: List[str], strategy: str) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Compute embeddings for texts using the given strategy.

        Returns:
            embeddings: list of tensors (either (D,) or (T, D) per sample)
            per_sample_metadata: list of dicts; for token-level, will contain token indices
        """
        input_ids, attention_mask, hidden_states = self._forward_hidden_states(texts)

        layer_sel = 'last'
        if strategy.startswith('-2_layer') or strategy.startswith('penultimate'):
            layer_sel = 'minus2'
        layer_states = self._select_layer(hidden_states, layer_sel)  # (B, L, D)

        pooled = []
        metadata = []

        if strategy.endswith('token_level'):
            # Return per-token embeddings for each sample; exclude padding positions
            for i in range(layer_states.shape[0]):
                valid_mask = attention_mask[i].bool()
                token_embeddings = layer_states[i][valid_mask]  # (T_i, D)
                pooled.append(token_embeddings.detach().cpu())
                # Record token positions of valid tokens
                token_indices = torch.arange(layer_states.shape[1], device='cuda')[valid_mask].tolist()
                metadata.append({'token_level': True, 'token_indices': token_indices})
        elif strategy.endswith('cls_token'):
            # First non-pad token as CLS
            for i in range(layer_states.shape[0]):
                # Find first valid token index
                valid_mask = attention_mask[i].bool()
                if valid_mask.any():
                    first_idx = int(valid_mask.nonzero(as_tuple=False)[0])
                else:
                    first_idx = 0
                cls_vec = layer_states[i, first_idx]
                pooled.append(cls_vec.detach().cpu())
                metadata.append({'token_level': False, 'pooled': 'cls'})
        elif strategy.endswith('mean_pooling'):
            mean_vec = self._masked_mean(layer_states, attention_mask)  # (B, D)
            for i in range(mean_vec.shape[0]):
                pooled.append(mean_vec[i].detach().cpu())
                metadata.append({'token_level': False, 'pooled': 'mean'})
        else:
            raise ValueError(f"Unknown text embedding strategy: {strategy}")

        return pooled, metadata