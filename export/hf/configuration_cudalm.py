"""cudalm model config for Hugging Face transformers (written by `cudalm hf`)."""
from transformers import PretrainedConfig


class CudalmConfig(PretrainedConfig):
    model_type = "cudalm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size=16384, hidden_size=160, num_hidden_layers=11, num_attention_heads=5,
                 num_key_value_heads=1, head_dim=None, intermediate_size=432, max_position_embeddings=2048,
                 rope_theta=10000.0, rms_norm_eps=1e-5, layer_ffn=None, layer_num_experts=None,
                 layer_intermediate_size=None, layer_top_k=None, layer_window=None, attn_gate=False,
                 value_residual=False, router_bias=False, ngram=None, tie_word_embeddings=True, use_cache=True, **kwargs):
        L = num_hidden_layers
        self.use_cache = use_cache                                  # forward(use_cache=None) falls back to this
        self.vocab_size, self.hidden_size, self.num_hidden_layers = vocab_size, hidden_size, L
        self.num_attention_heads, self.num_key_value_heads = num_attention_heads, num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.intermediate_size, self.max_position_embeddings = intermediate_size, max_position_embeddings
        self.rope_theta, self.rms_norm_eps = rope_theta, rms_norm_eps
        self.layer_ffn = layer_ffn or ["dense"] * L                 # "dense" | "moe"
        self.layer_num_experts = layer_num_experts or [0] * L
        self.layer_intermediate_size = layer_intermediate_size or [intermediate_size] * L
        self.layer_top_k = layer_top_k or [0] * L
        self.layer_window = layer_window or [0] * L                 # 0 = full causal attention
        self.attn_gate, self.value_residual = attn_gate, value_residual
        self.router_bias = int(router_bias)                         # bias rows: 0 none, 1 shared, K one per routing slot
        self.ngram = ngram                                          # {rows, dim, order, layer, gate} or None
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
