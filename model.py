"""
RWKV-like SSM + Sparse External Memory LM (strict causal, truncated SSM).

- Transformer Self-Attention 없음
- RWKV 스타일 time-mix / channel-mix 블록
- SSM: y_t = decay * y_{t-1} + rv_t 를
  최근 L step 지수 커널로 근사 (causal depthwise conv1d)
- Causal depthwise Conv1d + pointwise Conv1d (local pattern)
- Top-k cross-attention 기반 Sparse External Memory (slot 기반 장기 메모리)

Tokenizer: Rust BPE (/home/test/Project_LLM/rust_bpe_tokenizer.json)
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# ============================================================
# BPETokenizerWrapper
# ============================================================
class BPETokenizerWrapper:
    def __init__(self, model_path: str):
        self.tk = Tokenizer.from_file(model_path)

        # tokenizers.Tokenizer에는 get_special_tokens가 없음. pad/eos를 heuristics로 추정.
        pad_candidates = ["<pad>", "[PAD]", "[pad]", "<PAD>"]
        pad_id = None
        for tok in pad_candidates:
            tid = self.tk.token_to_id(tok)
            if tid is not None:
                pad_id = tid
                break
        # pad 토큰이 없으면 None 그대로 둔다
        self.pad_token_id = pad_id

        self.vocab_size = self.tk.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        out = self.tk.encode(text)
        return out.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tk.decode(ids, skip_special_tokens=skip_special_tokens)

    def __len__(self):
        return self.vocab_size

    @property
    def eos_token_id(self):
        for tok in ["</s>", "<eos>", "[EOS]", "[eos]"]:
            tid = self.tk.token_to_id(tok)
            if tid is not None:
                return tid
        return None


# (현재 사용은 안 하지만 남겨둔 sinusoidal time embedding)
def sinusoidal_t(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device) * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.size(1) < dim:
        emb = F.pad(emb, (0, 1), value=0.0)
    return emb


# ============================================================
# Sparse External Memory
# ============================================================
class SparseMemory(nn.Module):
    """
    slot 기반 외부 메모리.

    mem_state: (B, S, D)
    read:  query (B, T, D) → context (B, T, D)
    write: summary (B, D) + mem_state (B, S, D) → mem_state_new (B, S, D)
    """

    def __init__(
        self,
        dim: int,
        slots: int,
        heads: int = 4,
        top_k: int = 4,
        block_size: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.heads = heads
        self.top_k = min(top_k, slots)
        self.block_size = block_size

        # 초기 메모리 슬롯 (학습 가능한 글로벌 메모리)
        self.mem_slots = nn.Parameter(torch.zeros(slots, dim))
        nn.init.normal_(self.mem_slots, std=0.02)

        # 메모리 읽기용 프로젝션
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # 메모리 쓰기용 GRU-style 업데이트
        self.writer = nn.GRUCell(dim, dim)
        self.gate = nn.Linear(dim * 2, 1)

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """
        초기 mem_state 생성.
        return: (B, S, D)
        """
        return self.mem_slots.unsqueeze(0).expand(batch, -1, -1).to(device)

    def read(self, query: torch.Tensor, mem_state: torch.Tensor) -> torch.Tensor:
        """
        query: (B, T, D)
        mem_state: (B, S, D)
        return: (B, T, D)
        """
        B, T, D = query.shape
        S = mem_state.size(1)

        q = self.q_proj(query)       # (B, T, D)
        k = self.k_proj(mem_state)   # (B, S, D)
        v = self.v_proj(mem_state)   # (B, S, D)

        block = self.block_size if self.block_size > 0 else T
        outputs = []
        scale = 1.0 / math.sqrt(D)

        for start in range(0, T, block):
            end = min(T, start + block)
            q_blk = q[:, start:end, :]                     # (B, Tb, D)
            scores = torch.matmul(q_blk, k.transpose(1, 2)) * scale  # (B, Tb, S)

            if self.top_k < S:
                topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
                mask = torch.full_like(scores, float("-inf"))
                mask.scatter_(-1, topk_idx, topk_vals)
                scores = mask

            attn = torch.softmax(scores, dim=-1)           # (B, Tb, S)
            ctx_blk = torch.matmul(attn, v)                # (B, Tb, D)
            outputs.append(ctx_blk)

        ctx = torch.cat(outputs, dim=1)                    # (B, T, D)
        return self.out_proj(ctx)

    def write(self, summary: torch.Tensor, mem_state: torch.Tensor) -> torch.Tensor:
        """
        summary: (B, D) - 시퀀스 전체 요약 벡터
        mem_state: (B, S, D)
        return: (B, S, D)
        """
        B, S, D = mem_state.shape

        mem_mean = mem_state.mean(dim=1)                   # (B, D)
        gate = torch.sigmoid(self.gate(torch.cat([summary, mem_mean], dim=-1)))  # (B, 1)

        # 각 슬롯에 동일 summary를 확장
        summary_rep = summary.unsqueeze(1).expand(-1, S, -1)  # (B, S, D)

        mem_flat = mem_state.reshape(-1, D)               # (B*S, D)
        sum_flat = summary_rep.reshape(-1, D)             # (B*S, D)
        upd_flat = self.writer(sum_flat, mem_flat)        # (B*S, D)
        upd = upd_flat.reshape(B, S, D)                   # (B, S, D)

        return gate.unsqueeze(-1) * upd + (1 - gate.unsqueeze(-1)) * mem_state


# ============================================================
# RWKV-like Block (SSM + Causal Conv)
# ============================================================
class RWKVBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        time_mix_init: float = 0.5,
        channel_mix_init: float = 0.5,
        dropout: float = 0.2,
        res_scale: float = 0.5,
        dilation: int = 2,
        ssm_kernel: int = 64,   # SSM 근사 horizon (최근 ssm_kernel step만 본다)
    ):
        super().__init__()
        self.dim = dim
        self.ssm_kernel = ssm_kernel

        # time-mix / channel-mix 파라미터
        self.time_mix = nn.Parameter(torch.full((1, 1, dim), time_mix_init))
        self.channel_mix = nn.Parameter(torch.full((1, 1, dim), channel_mix_init))

        # RWKV-like time-mix 부분
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.receptance = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

        # SSM decay 파라미터
        self.decay_logit = nn.Parameter(torch.zeros(dim))

        # LayerNorm
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln_conv = nn.LayerNorm(dim)   # conv 브랜치 전용 LN

        # FFN
        self.ffn_up = nn.Linear(dim, dim * 4)
        self.ffn_down = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)
        self.res_scale = res_scale

        # Local causal conv (depthwise + pointwise)
        self.loc_depth = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=0,
            dilation=dilation,
            groups=dim,
        )
        self.loc_point = nn.Conv1d(dim, dim, kernel_size=1, padding=0)
        self.loc_kernel_size = 3
        self.loc_dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        # --- RWKV time-mix ---
        x_ln = self.ln1(x)

        # 시프트된 버전 (과거 토큰)
        x_shift = torch.zeros_like(x_ln)
        x_shift[:, 1:, :] = x_ln[:, :-1, :]

        # time-mix
        xk = x_ln * (1 - self.time_mix) + x_shift * self.time_mix
        xr = x_ln * (1 - self.time_mix) + x_shift * self.time_mix

        # k는 현재 구조에서는 직접 쓰지 않지만, 나중 확장용으로 남김
        k = self.key(xk)                       # (B, T, D)
        v = self.value(xk)                     # (B, T, D)
        r = torch.sigmoid(self.receptance(xr)) # (B, T, D)

        rv = r * v                             # (B, T, D)

        # ----------------------------------------------------
        # SSM 근사: y_t ≈ sum_{k=0}^{L-1} decay^k * rv_{t-k}
        # → depthwise causal conv1d로 구현 (L = self.ssm_kernel)
        # ----------------------------------------------------
        L = min(self.ssm_kernel, T)            # 시퀀스보다 길게 잡지 않도록

        # decay: (D,) in x.dtype
        decay = torch.sigmoid(self.decay_logit).to(x.dtype)  # (D,)

        # 지수 커널: [decay^0, decay^1, ..., decay^{L-1}]  (D, L)
        l_idx = torch.arange(L, device=x.device, dtype=x.dtype)  # (L,)
        kernel = decay.unsqueeze(1) ** l_idx.unsqueeze(0)        # (D, L)

        # conv1d는 커널을 뒤집어서 적용되므로, 인덱스 맞추기 위해 reverse
        # conv1d with padding = L-1:
        #   y_t = sum_{i=0}^{L-1} w[i] * x_{t + i - (L-1)}
        # 여기서 w[i] = decay^{L-1-i} 로 두면 → y_t = sum_{k=0}^{L-1} decay^k * rv_{t-k}
        kernel_flipped = torch.flip(kernel, dims=[1])           # (D, L)

        # depthwise conv weight shape: (out_channels, 1, kernel_size)
        w = kernel_flipped.unsqueeze(1)                         # (D, 1, L)

        # 입력: (B, D, T)
        rv_in = rv.transpose(1, 2)                              # (B, D, T)

        # 왼쪽 padding = L-1 → causal
        y_full = F.conv1d(rv_in, w, bias=None, padding=L - 1, groups=D)  # (B, D, T+L-1)

        # 앞쪽 T 타임스텝만 사용 (미래 안 봄)
        y_conv = y_full[:, :, :T]                               # (B, D, T)

        rwkv_out = y_conv.transpose(1, 2)                       # (B, T, D)
        rwkv_out = self.output(rwkv_out)                        # (B, T, D)

        x = x + self.res_scale * self.dropout(rwkv_out)

        # --- FFN (channel-mix) ---
        x_ln2 = self.ln2(x)
        x_prev = torch.zeros_like(x_ln2)
        x_prev[:, 1:, :] = x_ln2[:, :-1, :]

        xc = x_ln2 * (1 - self.channel_mix) + x_prev * self.channel_mix
        ffn = self.ffn_down(F.silu(self.ffn_up(xc)))
        x = x + self.res_scale * self.dropout(ffn)

        # --- Local causal conv ---
        x_conv_in = self.ln_conv(x)          # (B, T, D)
        xc_conv = x_conv_in.transpose(1, 2)  # (B, D, T)

        pad = (self.loc_kernel_size - 1) * self.loc_dilation
        xc_conv = F.pad(xc_conv, (pad, 0))   # (B, D, T + pad)

        xc_conv = self.loc_depth(xc_conv)    # (B, D, T)
        xc_conv = self.loc_point(xc_conv)    # (B, D, T)

        xc_conv = xc_conv.transpose(1, 2)    # (B, T, D)
        x = x + self.res_scale * self.dropout(xc_conv)

        return x


# ============================================================
# RWKVMemoryLM (전체 LM)
# ============================================================
class RWKVMemoryLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 16,
        mem_slots: int = 32,
        mem_heads: int = 4,
        mem_top_k: int = 4,
        max_len: int = 4096,
        pad_id: Optional[int] = None,
        dropout: float = 0.2,
        ssm_kernel: int = 64,  # RWKVBlock에 전달
    ):
        super().__init__()
        self.dim = dim
        self.pad_id = pad_id
        self.num_layers = num_layers
        self.max_len = max_len

        # 임베딩
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.rel_bias = nn.Parameter(torch.zeros(max_len))

        # 입력 정규화
        self.input_ln = nn.LayerNorm(dim)

        # RWKV Blocks
        self.layers = nn.ModuleList(
            [
                RWKVBlock(dim=dim, dropout=dropout, ssm_kernel=ssm_kernel)
                for _ in range(num_layers)
            ]
        )

        # Sparse External Memory
        self.memory = SparseMemory(
            dim=dim,
            slots=mem_slots,
            heads=mem_heads,
            top_k=mem_top_k,
        )
        self.mem_gate = nn.Linear(dim * 2, 1)

        # 출력쪽 LayerNorm + LM Head
        self.final_ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        tokens: torch.Tensor,
        mem_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: (B, T)
        mem_state: (B, S, D) 또는 None

        return:
          - logits: (B, T, vocab_size)
          - mem_state_new: (B, S, D)
        """
        B, T = tokens.shape
        device = tokens.device

        # 메모리 초기화 / detach
        if mem_state is None:
            mem_state = self.memory.init_state(B, device)
        else:
            # 학습 시 시퀀스 간 그래프 끊기
            mem_state = mem_state.detach()

        # 길이/마스크 계산
        if self.pad_id is not None:
            lengths = (tokens != self.pad_id).sum(dim=1)  # (B,)
        else:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)

        # 포지션
        if T > self.max_len:
            raise ValueError(f"tokens length {T} > max_len {self.max_len}")
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)

        # 임베딩 + 입력 LN
        x = self.tok_emb(tokens) + self.pos_emb(pos_ids)  # (B, T, D)
        x = self.input_ln(x)

        # 외부 메모리 read + gate
        mem_ctx = self.memory.read(x, mem_state)          # (B, T, D)
        rel = self.rel_bias[:T].unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        gate_input = torch.cat([x, mem_ctx], dim=-1)      # (B, T, 2D)
        gate = torch.sigmoid(self.mem_gate(gate_input) + rel)  # (B, T, 1)
        x = x + gate * mem_ctx

        # RWKV Blocks
        for layer in self.layers:
            x = layer(x)

        # 출력 LayerNorm + LM head
        x = self.final_ln(x)
        logits = self.lm_head(x)                          # (B, T, vocab_size)

        # 시퀀스 요약 → 메모리 쓰기
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
        mask = mask.unsqueeze(-1)                        # (B, T, 1)
        masked_x = x * mask                              # (B, T, D)
        denom = mask.sum(dim=1).clamp(min=1)             # (B, 1)
        summary = masked_x.sum(dim=1) / denom            # (B, D)

        mem_state_new = self.memory.write(summary, mem_state)  # (B, S, D)

        return logits, mem_state_new


# ============================================================
# Tokenizer & Demo
# ============================================================
TOKENIZER_MODEL_PATH = "/home/test/Project_LLM/rust_bpe_tokenizer.json"
tokenizer = BPETokenizerWrapper(TOKENIZER_MODEL_PATH)
TOKENIZER_VOCAB_SIZE = len(tokenizer)


def demo_forward(text: str = "안녕하세요, RWKV + Sparse Memory 테스트입니다."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = torch.tensor([tokenizer.encode(text)], device=device)
    model = RWKVMemoryLM(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        dim=512,
        num_layers=4,
        mem_slots=16,
        mem_top_k=4,
        pad_id=tokenizer.pad_token_id,
        ssm_kernel=64,  # 근사 horizon
    ).to(device)
    logits, mem_state = model(tokens)
    print("logits:", logits.shape, "mem_state:", mem_state.shape)


if __name__ == "__main__":
    demo_forward()
