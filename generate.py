"""
체크포인트를 로드해 간단히 텍스트 생성하는 스크립트.

사용 예:
    python generate_from_ckpt.py --prompt "안녕하세요" --max_new_tokens 50
    python generate_from_ckpt.py --ckpt checkpoints/step_10000.pt --prompt "..." --top_k 20 --temperature 0.8

기본적으로 checkpoints 디렉토리에서 최신 step_* 체크포인트를 찾아 로드합니다.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional

import torch
from llm_model import RWKVMemoryLM, tokenizer
from train_stateful_lm import TrainConfig  # for loading pickled config in ckpt
from torch.serialization import add_safe_globals


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    def step_num(p: Path) -> int:
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1
    ckpts = sorted([p for p in ckpt_dir.glob("step_*.pt") if p.is_file()], key=step_num)
    return ckpts[-1] if ckpts else None


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = 100,
    top_p: Optional[float] = 0.95,
    repetition_penalty: float = 1.1,
    generated: torch.Tensor | None = None,
) -> int:
    # 반복 패널티 (HuggingFace 스타일: 부호 고려)
    if generated is not None and repetition_penalty != 1.0:
        logits = logits.clone()
        for tok in set(generated.view(-1).tolist()):
            if logits[tok] > 0:
                logits[tok] /= repetition_penalty
            else:
                logits[tok] *= repetition_penalty
    else:
        logits = logits.clone()

    # temperature
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    logits = logits / temperature

    # top_k
    if top_k is not None and top_k > 0 and top_k < logits.size(0):
        values, indices = torch.topk(logits, top_k)
        new_logits = torch.full_like(logits, float("-inf"))
        new_logits.scatter_(0, indices, values)
        logits = new_logits

    # top_p (nucleus)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        if mask.any():
            first_mask = mask.int().argmax(dim=-1)
            sorted_logits[first_mask + 1 :] = float("-inf")
        new_logits = torch.full_like(logits, float("-inf"))
        new_logits.scatter_(0, sorted_indices, sorted_logits)
        logits = new_logits

    # 최종 softmax + 샘플링
    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def load_model_and_tokenizer(device: torch.device, ckpt_path: Path):
    model = RWKVMemoryLM(vocab_size=len(tokenizer), pad_id=getattr(tokenizer, "pad_token_id", None)).to(device)

    # 체크포인트에는 TrainConfig 객체가 포함되어 있어 weights_only=True로는 로드가 실패할 수 있음
    # TrainConfig 클래스를 안전 글로벌에 추가해 언피클 허용
    add_safe_globals([TrainConfig])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # llm_model.py의 tokenizer는 Rust BPE를 이미 로드하고 있으므로 그대로 반환
    return model, tokenizer


# === 추가: 프롬프트 기준 next-token 분포 debug용 함수 ===
def debug_next_token_topk(
    model: RWKVMemoryLM,
    tok,
    prompt: str,
    device: torch.device,
    k: int = 20,
) -> None:
    """프롬프트 직후 next-token raw logits 기준 top-k 토큰과 확률을 출력."""
    ids = tok.encode(prompt)
    tokens = torch.tensor([ids], device=device)

    model.eval()
    with torch.inference_mode():
        logits, _ = model(tokens, None)
        last_logits = logits[:, -1, :].squeeze(0)  # (vocab_size,)
        probs = torch.softmax(last_logits, dim=-1)

        values, indices = torch.topk(last_logits, k)
        print("=== Top-{} next-token (raw logits 기준) ===".format(k))
        print(f"Prompt: {prompt!r}")
        print(f"{'rank':>4} {'id':>6} {'token':>15} {'logit':>10} {'prob':>10}")
        for rank, (logit, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            token_str = tok.decode([idx])
            prob = float(probs[idx].item())
            print(f"{rank:4d} {idx:6d} {repr(token_str):>15} {logit:10.4f} {prob:10.6f}")
    print()


def generate(
    model: RWKVMemoryLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
):
    # 인코딩 (Rust BPE; special 토큰 추가 없음)
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], device=device)
    mem_state = None

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # 학습 시와 동일하게 전체 시퀀스를 한 번에 forward에 넣고 마지막 위치만 사용
            logits, mem_state = model(tokens, mem_state)
            next_logits = logits[:, -1, :].squeeze(0)
            next_id = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated=tokens,
            )
            next_tensor = torch.tensor([[next_id]], device=device)
            tokens = torch.cat([tokens, next_tensor], dim=1)
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
    print("DEBUG token ids:", tokens[0].tolist())
    print("DEBUG decoded:", text[:500])
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="불러올 체크포인트 경로 (미지정 시 최신 step_*.pt 사용)")
    parser.add_argument("--prompt", type=str, default="France is a country in Western Europe. Its capital city is", help="프롬프트 텍스트")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    # === 추가: top-k 분포 디버그 옵션 ===
    parser.add_argument("--debug_topk", action="store_true", help="프롬프트 직후 next-token top-k 분포를 출력")
    parser.add_argument("--debug_k", type=int, default=20, help="top-k 디버그 시 사용할 k 값")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    ckpt_dir = base_dir / "checkpoints"

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다. (디렉토리: {ckpt_dir})")
    print(f"[load] checkpoint: {ckpt_path}")

    model, tok = load_model_and_tokenizer(device, ckpt_path)

    # 먼저 top-k 분포를 보고 싶으면 여기서 출력
    if args.debug_topk:
        debug_next_token_topk(
            model=model,
            tok=tok,
            prompt=args.prompt,
            device=device,
            k=args.debug_k,
        )

    text = generate(
        model,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print("=== Generation ===")
    print(text)


if __name__ == "__main__":
    main()
