import torch
from torch import nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)) # row-major memory ordering (必须这么搞)

        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size = num_embeddings    # size of the vocab
        self.d_model = embedding_dim      # dimension of the embedding vectors
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # lookup the embedding vectors for the given token ids
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        # gain is a trainable param, initialized as all "1"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        # rms shape: (B, T, 1) 
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms # normalize
        x = x.to(in_dtype)
        return self.gain * x  # 把 self.gain shape (C,) 广播到 (1, 1, C) 再和 x 逐元素相乘

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    # Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype) # d_model -> d_ff  gate preact
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) # d_model -> d_ff  value
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype) # d_ff -> d_model projection back
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        x1 = self.w1(x)
        x3 = self.w3(x)
        x = silu(x1) * x3 # gated silu. Hardamard product.
        x = self.w2(x)
        return x

class RoPE_llama(nn.Module):
    """Rotary Position Embeddings (RoPE) for queries/keys. This is the Llama-style RoPE (precomputed cache version), which is more traditional.

    Inputs:
      x: (..., seq_len, d_k)
      token_positions: (..., seq_len) or (seq_len,) or (batch, seq_len)
    Outputs:
      Same shape as x: (..., seq_len, d_k)
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        - theta: float Θ value for the RoPE
        - d_k: int dimension of query and key vectors
        - max_seq_len: int Maximum sequence length that will be inputted
        - device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires d_k to be even."
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        self.device = device

        p = torch.arange(0, d_k // 2, dtype=torch.float64, device=device)
        inv_freq = 1.0 / (self.theta ** (2.0 * p / d_k))

        # Key fix: register_buffer before converting to float32, otherwise it cannot run on mps
        inv_freq = inv_freq.to(torch.float32)
        
        # positions: 0..max_seq_len-1
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (L,)
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (L, d_k/2)

        # Precompute and cache; persistent=False means not saved in state_dict (not saved to checkpoint files - because it can be recomputed anytime)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)  # (L, d_k/2)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)  # (L, d_k/2)
    
    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        x:   (..., seq_len, d_k)
        cos: (..., seq_len, d_k/2)  (already aligned to batch/seq dims)
        sin: (..., seq_len, d_k/2)

        Treat (x_even, x_odd) as 2D vectors and rotate:
          [x_even'] = x_even*cos - x_odd*sin
          [x_odd' ] = x_even*sin + x_odd*cos
        """
        # Python slicing syntax: start:stop:step
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2). Ellipsis means all preceding dimensions are retained. Take all even positions. 
        x_odd  = x[..., 1::2]  # (..., seq_len, d_k/2) Take all odd positions. 

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # Interleave back to (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """This function mainly performs the computation in _apply_rope, and itself is only responsible for handling the shape and type conversion of inputs and outputs.
        x: (..., seq_len, d_k)
        token_positions: shape can be
          - (seq_len,)
          - (..., seq_len)  (e.g., (batch, seq_len) or more batch dims)
        """
        assert x.size(-1) == self.d_k, f"Expected last dim d_k={self.d_k}, got {x.size(-1)}"
        seq_len = x.size(-2)

        # token_positions converted to long for indexing
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)
        
        # Check that the maximum position does not exceed the precomputed length (torch.numel() returns the total number of elements in an input tensor.)
        max_pos = int(token_positions.max().item()) if token_positions.numel() > 0 else 0 
        if max_pos >= self.max_seq_len:
            raise ValueError(
                f"token_positions has max={max_pos}, but max_seq_len={self.max_seq_len}. "
                "Please increase max_seq_len in RoPE init."
            )
        
        # Fetch cos/sin from cache
        cos = self.cos_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        sin = self.sin_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        
        # Align batch dims of cos/sin with x: target shape should be x.shape[:-1] with last dim d_k/2
        if cos.shape[-2] != seq_len:
            raise ValueError(f"token_positions seq dim {cos.shape[-2]} != x seq_len {seq_len}")
        
        # dtype alignment
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return self._apply_rope(x, cos, sin)

class RoPE(nn.Module):
    """This is Lazy / Auto-expand RoPE. A more mainstream engineering solution. 
    This way, Attn / Block do not need to input the max_seq_len parameter.

    - init does not require max_seq_len
    - forward automatically expands cos/sin cache based on the maximum value of token_positions
    """
    def __init__(self, theta: float, d_k: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires d_k to be even."
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.device = device

        p = torch.arange(0, d_k // 2, dtype=torch.float64, device=device)
        inv_freq = 1.0 / (self.theta ** (2.0 * p / d_k))

        # Key fix: register_buffer before converting to float32, otherwise it cannot run on mps
        self.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)  # (d_k/2,)

        # Lazy caches
        self.register_buffer("cos_cached", torch.empty(0, d_k // 2, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, d_k // 2, dtype=torch.float32, device=device), persistent=False)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        x:   (..., seq_len, d_k)
        cos: (..., seq_len, d_k/2)
        sin: (..., seq_len, d_k/2)
        """
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd  = x[..., 1::2]  # (..., seq_len, d_k/2)

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # Interleave back (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    @torch.no_grad()
    def _maybe_extend_cache(self, needed_len: int, device: torch.device):
        """确保缓存至少覆盖 [0, needed_len) positions。
        """
        cur_len = int(self.cos_cached.size(0))
        if needed_len <= cur_len:
            return

        # Range to extend cache [cur_len, needed_len)
        new_positions = torch.arange(cur_len, needed_len, dtype=torch.float32, device=device)  # (ΔL,)
        # angles: (ΔL, d_k/2)
        # inv_freq might be on a different device (if module init with device=None), align here
        inv_freq = self.inv_freq.to(device=device)
        angles = torch.einsum("i,j->ij", new_positions, inv_freq) # float64 * float32 -> float64
        
        new_cos = angles.cos().to(dtype=torch.float32)  # (ΔL, d_k/2)
        new_sin = angles.sin().to(dtype=torch.float32)

        # If the current cache device is incorrect (e.g., module created on CPU but x is on CUDA), migrate it
        if self.cos_cached.device != device:
            self.cos_cached = self.cos_cached.to(device=device)
            self.sin_cached = self.sin_cached.to(device=device)

        if cur_len == 0:
            self.cos_cached = new_cos
            self.sin_cached = new_sin
        else:
            self.cos_cached = torch.cat([self.cos_cached, new_cos], dim=0)
            self.sin_cached = torch.cat([self.sin_cached, new_sin], dim=0)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (seq_len,) or (..., seq_len)
        """
        assert x.size(-1) == self.d_k, f"Expected last dim d_k={self.d_k}, got {x.size(-1)}"
        seq_len = x.size(-2)

        # Convert token_positions to long for indexing
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)
        
        max_pos = int(token_positions.max().item()) if token_positions.numel() > 0 else 0 
        needed_len = max_pos + 1
        self._maybe_extend_cache(needed_len=needed_len, device=x.device)

        # Fetch corresponding cos/sin from cache and reshape back to token_positions' batch shape
        cos = self.cos_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        sin = self.sin_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        
        # sanity check: seq dimension alignment
        if cos.shape[-2] != seq_len:
            raise ValueError(f"token_positions seq dim {cos.shape[-2]} != x seq_len {seq_len}")
        
        # dtype alignment
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return self._apply_rope(x, cos, sin)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute softmax over specified dimension.
    
    Params:
     - x: input tensor of any shape
     - dim: dimension to compute softmax over. Last dimension by default.
    
    Returns:
     - tensor of the same shape as input, containing softmax values for each element
    """
    x_normalized = x - x.max(dim=dim, keepdim=True).values 
    
    x_exp = x_normalized.exp()
    
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Params:
     - q: (..., seq_len_q, d_q)
     - k: (..., seq_len_k, d_k)
     - v: (..., seq_len_v, d_v)
     - mask: boolean mask, (seq_len, seq_len)
    
    Returns:
     - tensor with shape (..., seq_len_q, d_v)
    """
    score = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))

    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))

    attention_weights = softmax(score, dim=-1)

    return attention_weights @ v

class CausalSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        # self.max_seq_len = max_seq_len
        # self.register_buffer("bias", torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).view(1, 1, self.max_seq_len, self.max_seq_len) # (1, 1, seq_len, seq_len), 前两维是为了 attn 的时候形状对齐)

        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = Linear(self.d_model, 3 * self.d_model)
        # output projection
        self.out_proj = Linear(self.d_model, self.d_model)
    
    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, d_model)
        """
        B, T, C = x.size()

        # 计算 key, query, value
        qkv = self.qkv_proj(x) # (batch, seq_len, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=-1) # each is (batch, seq_len, d_model)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)

        attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        attn_output = scaled_dot_product_attention(q, k, v, mask=attn_mask) # (batch, n_head, seq_len, head_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # (batch, seq_len, d_model) <- `concatenation` operation
        y = self.out_proj(attn_output) # (batch, seq_len, d_model)
        
        return y

class CausalSelfAttention_RoPE(nn.Module):
    """
    Causal multi-head self-attention.
    """
    def __init__(self, d_model: int, n_head: int, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = Linear(self.d_model, 3 * self.d_model)
        # output projection
        self.out_proj = Linear(self.d_model, self.d_model)

        # RoPE：注意 d_k = head_dim
        self.rope = RoPE(theta=theta, d_k=self.head_dim)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        B, T, C = x.size()

        qkv = self.qkv_proj(x) # (batch, seq_len, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=-1) # each is (batch, seq_len, d_model)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (batch, n_head, seq_len, head_size)

        if token_positions is None:
            token_positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1,T)
        elif token_positions.ndim == 1:
            token_positions = token_positions.unsqueeze(0)  # (1,T)

        q = self.rope(q, token_positions)  # (B,H,T,hd)
        k = self.rope(k, token_positions)

        attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        attn_output = scaled_dot_product_attention(q, k, v, mask=attn_mask) # (batch, n_head, seq_len, head_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # (batch, seq_len, d_model) <- `concatenation` operation
        y = self.out_proj(attn_output) # (batch, seq_len, d_model)
        
        return y

class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, theta: float = 10000.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention_RoPE(d_model, n_head, theta)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), token_positions=token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, theta: float, vocab_size: int, context_length: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(d_model, n_head, d_ff, theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.context_length = context_length
        self.embedding = Embedding(vocab_size, d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.context_length, f"Cannot forward sequence of length {T}, context length is only {self.context_length}"

        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits





