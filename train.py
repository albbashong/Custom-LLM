"""
간단한 상태+장기 메모리 LM 학습 스크립트.
- 데이터: /home/test/final_155gb_stable_data/parquet_tok/wiki (tokenized parquet, input_ids 컬럼)
- 매 4000 배치마다 체크포인트 저장(.pt)
- 매 2000 배치마다 간단한 프리뷰 생성
- TensorBoard 로깅 + 강한 디버그 출력 포함
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List
from pathlib import Path

import torch
import torch.nn.functional as F
import pyarrow.dataset as ds
from torch.serialization import add_safe_globals
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from llm_model import RWKVMemoryLM, tokenizer, TOKENIZER_VOCAB_SIZE


# --- 텍스트 정제 / 토크나이즈 공통 (지금은 tokenized_input=True라 사용 X 이지만 남겨둠) ---
def clean_text(txt: str, max_char_len: int, use_unicode_escape: bool = False) -> str:
    if not isinstance(txt, str):
        try:
            txt = str(txt)
        except Exception:
            return ""
    if len(txt) > max_char_len:
        txt = txt[:max_char_len]
    if use_unicode_escape:
        try:
            txt = txt.encode("utf-8").decode("unicode_escape", errors="ignore")
        except Exception:
            pass
    txt = txt.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    txt = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", txt)
    return txt


def tokenize_batch_texts(
    texts: List[str],
    tokenizer,
    max_length: int,
    max_char_len: int,
    use_unicode_escape: bool,
    min_tokens: int,
) -> List[List[int]]:
    def chunk_ids(ids: List[int]) -> List[List[int]]:
        chunks = []
        for start in range(0, len(ids), max_length):
            part = ids[start : start + max_length]
            if len(part) >= min_tokens:
                chunks.append(part)
        return chunks

    cleaned = []
    for t in texts:
        if not t:
            continue
        t = clean_text(t, max_char_len, use_unicode_escape)
        if not t:
            continue
        cleaned.append(t)
    if not cleaned:
        return []
    if hasattr(tokenizer, "encode") and not hasattr(tokenizer, "batch_encode_plus"):
        ids_batch = tokenizer.encode(cleaned, add_special_tokens=True)
    else:
        ids_batch = tokenizer.batch_encode_plus(
            cleaned,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
    if ids_batch and isinstance(ids_batch[0], int):
        ids_batch = [ids_batch]
    result = []
    for ids in ids_batch:
        if len(ids) < min_tokens:
            continue
        # 길이가 max_length보다 길면 여러 윈도우로 쪼개어 모두 포함
        result.extend(chunk_ids(ids))
    return result


# ----- Parquet iterator helpers -----
def iter_parquet_round_robin(
    path: str,
    batch_rows: int,
    text_col: str,
    tokenizer,
    max_length: int,
    shuffle_buffer: int,
    seed: int | None,
    log_fragments: bool = False,
    log_once: bool = True,
    max_char_len: int = 20000,
    use_unicode_escape: bool = False,
    tokenized_input: bool = False,
    max_batches_per_shard: int | None = None,
    max_random_skip_batches: int = 0,
    max_random_row_offset: int = 0,
    rows_per_shard_limit: int | None = None,
    min_tokens: int = 8,
    fragment_filter=None,
):
    """
    Infinite generator:
      - Shard list is shuffled once per epoch.
      - Opens one shard at a time, reads batches, tokenizes rows.
      - shuffle_buffer stores token IDs only.
    """
    rng = torch.Generator()
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)
    dataset = ds.dataset(path, format="parquet")
    fragments = list(dataset.get_fragments())
    if fragment_filter is not None:
        fragments = [f for f in fragments if fragment_filter(getattr(f, "path", None))]
    if len(fragments) == 0:
        return
    seen = set()
    buffer: list[list[int]] = []

    while True:
        order = torch.randperm(len(fragments), generator=rng).tolist()
        for idx in order:
            frag = fragments[idx]
            frag_path = getattr(frag, "path", None)
            if log_fragments:
                key = str(frag_path) if frag_path is not None else str(frag)
                if not log_once or key not in seen:
                    print(f"[data] reading fragment: {key}")
                    seen.add(key)
            columns = None if not tokenized_input else ["input_ids"]
            scanner = frag.scanner(batch_size=batch_rows, columns=columns)

            skip_batches = 0
            if max_random_skip_batches > 0:
                skip_batches = int(torch.randint(max_random_skip_batches + 1, (1,), generator=rng).item())

            # 행 단위 무작위 오프셋/제한
            skip_rows = 0
            if max_random_row_offset < 0:
                try:
                    frag_rows = frag.count_rows()
                except Exception:
                    frag_rows = None
                if frag_rows is not None:
                    if rows_per_shard_limit is not None and rows_per_shard_limit > 0:
                        offset_max = max(0, frag_rows - rows_per_shard_limit)
                    else:
                        offset_max = max(0, frag_rows - 1)
                    if offset_max > 0:
                        skip_rows = int(torch.randint(offset_max + 1, (1,), generator=rng).item())
            elif max_random_row_offset > 0:
                skip_rows = int(torch.randint(max_random_row_offset + 1, (1,), generator=rng).item())

            rows_budget = rows_per_shard_limit
            batch_cnt = 0

            for batch in scanner.to_batches():
                if rows_budget is not None and rows_budget <= 0:
                    break
                if skip_batches > 0:
                    skip_batches -= 1
                    batch_cnt += 1
                    continue

                start = 0
                end = batch.num_rows
                if skip_rows > 0:
                    if skip_rows >= end:
                        skip_rows -= end
                        batch_cnt += 1
                        continue
                    start = skip_rows
                    skip_rows = 0

                if rows_budget is not None:
                    take = min(rows_budget, end - start)
                    end = start + take
                else:
                    take = end - start
                if take <= 0:
                    batch_cnt += 1
                    continue

                if tokenized_input:
                    if "input_ids" not in batch.schema.names:
                        batch_cnt += 1
                        continue
                    ids_list = batch.column("input_ids").to_pylist()[start:end]
                    for ids in ids_list:
                        if not ids or len(ids) < min_tokens:
                            continue
                        if len(ids) > max_length:
                            ids = ids[:max_length]
                        yield ids
                else:
                    if text_col not in batch.schema.names:
                        batch_cnt += 1
                        continue
                    texts = batch.column(text_col).to_pylist()[start:end]
                    tokenized = tokenize_batch_texts(
                        texts,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        max_char_len=max_char_len,
                        use_unicode_escape=use_unicode_escape,
                        min_tokens=min_tokens,
                    )
                    if shuffle_buffer > 0:
                        for ids in tokenized:
                            buffer.append(ids)
                            if len(buffer) >= shuffle_buffer:
                                b_idx = torch.randint(len(buffer), (1,), generator=rng).item()
                                yield buffer.pop(b_idx)
                    else:
                        for ids in tokenized:
                            yield ids

                if rows_budget is not None:
                    rows_budget -= take
                batch_cnt += 1
                if max_batches_per_shard is not None and batch_cnt >= max_batches_per_shard:
                    break

            del scanner
            del frag

        if shuffle_buffer > 0:
            while buffer:
                b_idx = torch.randint(len(buffer), (1,), generator=rng).item()
                yield buffer.pop(b_idx)


# ----- 설정 -----
@dataclass
class TrainConfig:
    # tokenized parquet 경로 (input_ids, labels 컬럼만 포함)
    data_path: str = "/home/test/final_155gb_stable_data/parquet_tok/wiki"
    data_path2: str = None
    text_col: str = "text"
    arrow_batch_rows: int = 2000
    max_length: int = 256
    batch_size: int = 8
    num_workers_jsonl: int = 2
    num_workers_parquet: int = 2
    lr: float = 1e-4
    log_dir: str = "./runs/stateful_lm"
    ckpt_dir: str = "./checkpoints"
    save_every: int = 10000
    enable_preview: bool = True
    preview_every: int = 2000
    preview_max_new_tokens: int = 64
    max_char_len: int = 20000
    use_unicode_escape: bool = False
    tokenized_input: bool = True  # parquet에 input_ids/labels가 있는 경우
    prompt_text: str = "In this article, we discuss the history of"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_seed_code: int | None = None
    data_seed_wiki: int | None = None
    max_batches_per_shard: int | None = None
    max_random_skip_batches: int = 0
    max_random_row_offset: int = 0
    rows_per_shard_limit: int | None = None
    min_tokens: int = 16
    shard_split_threshold: int | None = 30
    log_loss_every: int = 2000  # 이 스텝마다 디버그 출력


# ----- JSONL/Parquet Iterable (지금은 Parquet + tokenized_input 사용) -----
class JsonlTextIterableDataset(IterableDataset):
    def __init__(self, path: str, tokenizer, max_length: int, max_char_len: int, use_unicode_escape: bool, min_tokens: int):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_char_len = max_char_len
        self.use_unicode_escape = use_unicode_escape
        self.min_tokens = min_tokens

    def parse_file(self) -> Iterable[List[int]]:
        with open(self.path, "r", encoding="utf-8") as f:
            batch_texts: List[str] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text")
                if not text:
                    continue
                batch_texts.append(text)
                if len(batch_texts) >= 256:
                    for ids in tokenize_batch_texts(
                        batch_texts, self.tokenizer, self.max_length, self.max_char_len, self.use_unicode_escape, self.min_tokens
                    ):
                        yield ids
                    batch_texts = []
            if batch_texts:
                for ids in tokenize_batch_texts(
                    batch_texts, self.tokenizer, self.max_length, self.max_char_len, self.use_unicode_escape, self.min_tokens
                ):
                    yield ids

    def __iter__(self):
        return self.parse_file()


class ParquetTextIterableDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        text_col: str = "text",
        batch_rows: int = 500,
        shuffle_buffer: int = 0,
        seed: int = 42,
        log_fragments: bool = True,
        log_once: bool = True,
        infinite_epochs: bool = True,
        max_char_len: int = 20000,
        use_unicode_escape: bool = False,
        tokenized_input: bool = False,
        max_batches_per_shard: int | None = None,
        max_random_skip_batches: int = 0,
        max_random_row_offset: int = 0,
        rows_per_shard_limit: int | None = None,
        min_tokens: int = 2,
        fragment_filter=None,
    ):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.batch_rows = batch_rows
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.log_fragments = log_fragments
        self.log_once = log_once
        self.infinite_epochs = infinite_epochs
        self.max_char_len = max_char_len
        self.use_unicode_escape = use_unicode_escape
        self.tokenized_input = tokenized_input
        self.max_batches_per_shard = max_batches_per_shard
        self.max_random_skip_batches = max_random_skip_batches
        self.max_random_row_offset = max_random_row_offset
        self.rows_per_shard_limit = rows_per_shard_limit
        self.min_tokens = min_tokens
        self.fragment_filter = fragment_filter

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        is_main_worker = worker_info is None or worker_info.id == 0
        dataset = ds.dataset(self.path, format="parquet")
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        fragments = list(dataset.get_fragments())
        if self.fragment_filter is not None:
            fragments = [f for f in fragments if self.fragment_filter(getattr(f, "path", None))]
        if len(fragments) == 0:
            return
        seen = set()
        buffer: list[list[int]] = []

        while True:
            order = torch.randperm(len(fragments), generator=rng).tolist()
            for idx in order:
                frag = fragments[idx]
                frag_path = getattr(frag, "path", None)
                if self.log_fragments and is_main_worker:
                    key = str(frag_path) if frag_path is not None else str(frag)
                    if not self.log_once or key not in seen:
                        print(f"[data] reading fragment: {key}")
                        seen.add(key)

                columns = None if not self.tokenized_input else ["input_ids"]
                skip_batches = 0
                if self.max_random_skip_batches > 0:
                    skip_batches = int(torch.randint(self.max_random_skip_batches + 1, (1,), generator=rng).item())

                skip_rows = 0
                if self.max_random_row_offset < 0:
                    try:
                        frag_rows = frag.count_rows()
                    except Exception:
                        frag_rows = None
                    if frag_rows is not None:
                        if self.rows_per_shard_limit is not None and self.rows_per_shard_limit > 0:
                            offset_max = max(0, frag_rows - self.rows_per_shard_limit)
                        else:
                            offset_max = max(0, frag_rows - 1)
                        if offset_max > 0:
                            skip_rows = int(torch.randint(offset_max + 1, (1,), generator=rng).item())
                elif self.max_random_row_offset > 0:
                    skip_rows = int(torch.randint(self.max_random_row_offset + 1, (1,), generator=rng).item())
                rows_budget = None if (self.rows_per_shard_limit is None or self.rows_per_shard_limit == 0) else self.rows_per_shard_limit

                batch_cnt = 0
                for batch in frag.to_batches(batch_size=self.batch_rows, columns=columns):
                    if rows_budget is not None and rows_budget <= 0:
                        break
                    if skip_batches > 0:
                        skip_batches -= 1
                        batch_cnt += 1
                        continue

                    start = 0
                    end = batch.num_rows
                    if skip_rows > 0:
                        if skip_rows >= end:
                            skip_rows -= end
                            batch_cnt += 1
                            continue
                        start = skip_rows
                        skip_rows = 0

                    if rows_budget is not None:
                        take = min(rows_budget, end - start)
                        end = start + take
                    else:
                        take = end - start
                    if take <= 0:
                        batch_cnt += 1
                        continue

                    if self.tokenized_input:
                        if "input_ids" not in batch.schema.names:
                            batch_cnt += 1
                            continue
                        ids_list = batch.column("input_ids").to_pylist()[start:end]
                        for ids in ids_list:
                            if not ids or len(ids) < self.min_tokens:
                                continue
                            # 길이가 max_length를 넘으면 여러 윈도우로 쪼개어 모두 사용
                            for s in range(0, len(ids), self.max_length):
                                part = ids[s : s + self.max_length]
                                if len(part) >= self.min_tokens:
                                    yield part
                    else:
                        if self.text_col not in batch.schema.names:
                            batch_cnt += 1
                            continue
                        texts = batch.column(self.text_col).to_pylist()[start:end]
                        tokenized = tokenize_batch_texts(
                            texts,
                            tokenizer=self.tokenizer,
                            max_length=self.max_length,
                            max_char_len=self.max_char_len,
                            use_unicode_escape=self.use_unicode_escape,
                            min_tokens=self.min_tokens,
                        )
                        if self.shuffle_buffer > 0:
                            for ids in tokenized:
                                buffer.append(ids)
                                if len(buffer) >= self.shuffle_buffer:
                                    b_idx = torch.randint(len(buffer), (1,), generator=rng).item()
                                    yield buffer.pop(b_idx)
                        else:
                            for ids in tokenized:
                                yield ids

                    if rows_budget is not None:
                        rows_budget -= take
                    batch_cnt += 1
                    if self.max_batches_per_shard is not None and batch_cnt >= self.max_batches_per_shard:
                        break

            if self.shuffle_buffer > 0:
                while buffer:
                    b_idx = torch.randint(len(buffer), (1,), generator=rng).item()
                    yield buffer.pop(b_idx)

            if not self.infinite_epochs:
                break


def collate_batch(batch: List, pad_id: int, max_length: int) -> Dict[str, torch.Tensor]:
    cleaned = []
    for item in batch:
        if isinstance(item, tuple) and len(item) >= 1:
            ids = item[0]
            lbl = None
        else:
            ids, lbl = item, None
        if ids is None or len(ids) < 2:
            continue
        cleaned.append((ids, lbl))
    if not cleaned:
        raise ValueError("배치 내 유효한 시퀀스가 없습니다")

    input_ids = []
    labels = []
    for ids, lbl in cleaned:
        if lbl is None:
            inp = ids[:-1]
            lbl = ids[1:]
        else:
            inp = ids
        target_len = max_length
        if len(inp) > target_len:
            inp = inp[:target_len]
            lbl = lbl[:target_len]
        pad_len = target_len - len(inp)
        inp = inp + [pad_id] * pad_len
        lbl = lbl + [-100] * pad_len
        input_ids.append(inp)
        labels.append(lbl)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ----- 생성 샘플 (간단 프리뷰용) -----
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    generated: torch.Tensor | None = None,
) -> int:
    if generated is not None and repetition_penalty != 1.0:
        logits = logits.clone()
        for tok in set(generated.view(-1).tolist()):
            if logits[tok] > 0:
                logits[tok] /= repetition_penalty
            else:
                logits[tok] *= repetition_penalty
    else:
        logits = logits.clone()

    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    logits = logits / temperature

    if top_k is not None and top_k > 0 and top_k < logits.size(0):
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, indices, values)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        if mask.any():
            first_mask = mask.int().argmax(dim=-1)
            sorted_logits[first_mask + 1 :] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(
    model: RWKVMemoryLM,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], device=device, dtype=torch.long)
    mem_state = None

    with torch.inference_mode():
        for _ in range(max_new_tokens):
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
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return text


def preview_generation(model: RWKVMemoryLM, tokenizer, device: torch.device, prompt: str = "안녕", max_new_tokens: int = 30):
    text = generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return text


def find_latest_checkpoint(ckpt_dir: str) -> Path | None:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None

    def step_num(p: Path) -> int:
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    ckpts = sorted([p for p in ckpt_path.glob("step_*.pt") if p.is_file()], key=step_num)
    return ckpts[-1] if ckpts else None


def train(cfg: TrainConfig):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    base_dir = Path(__file__).resolve().parent
    log_dir = Path(cfg.log_dir)
    ckpt_dir = Path(cfg.ckpt_dir)
    if not log_dir.is_absolute():
        log_dir = base_dir / log_dir
    if not ckpt_dir.is_absolute():
        ckpt_dir = base_dir / ckpt_dir

    device = torch.device(cfg.device)
    model = RWKVMemoryLM(vocab_size=len(tokenizer), pad_id=pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    # 체크포인트 로드
    start_step = 0
    latest = find_latest_checkpoint(str(ckpt_dir))
    if latest:
        add_safe_globals([TrainConfig])
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 0)
        print(f"[ckpt] loaded: {latest} (start_step={start_step})")

    # 단일 소스(wiki)만 사용 + shard 번호로 3:1 비율 분리
    class MixedIterable(IterableDataset):
        def __iter__(self_inner):
            import re

            def shard_num(path: str | None) -> int | None:
                if path is None:
                    return None
                m = re.search(r"shard_(\d+)", str(path))
                return int(m.group(1)) if m else None

            # shard split (low/high 3:1)
            if cfg.shard_split_threshold is not None:
                thresh = cfg.shard_split_threshold

                def filt_low(p):
                    n = shard_num(p)
                    return n is not None and n < thresh

                def filt_high(p):
                    n = shard_num(p)
                    return n is not None and n >= thresh

                gen_low = iter_parquet_round_robin(
                    cfg.data_path,
                    max_length=cfg.max_length,
                    batch_rows=cfg.arrow_batch_rows,
                    text_col=cfg.text_col,
                    tokenizer=tokenizer,
                    shuffle_buffer=0,
                    seed=cfg.data_seed_code,
                    log_fragments=False,
                    max_char_len=cfg.max_char_len,
                    use_unicode_escape=cfg.use_unicode_escape,
                    tokenized_input=cfg.tokenized_input,
                    max_batches_per_shard=cfg.max_batches_per_shard,
                    max_random_skip_batches=cfg.max_random_skip_batches,
                    max_random_row_offset=cfg.max_random_row_offset,
                    rows_per_shard_limit=cfg.rows_per_shard_limit,
                    min_tokens=cfg.min_tokens,
                    fragment_filter=filt_low,
                )
                gen_high = iter_parquet_round_robin(
                    cfg.data_path,
                    max_length=cfg.max_length,
                    batch_rows=cfg.arrow_batch_rows,
                    text_col=cfg.text_col,
                    tokenizer=tokenizer,
                    shuffle_buffer=0,
                    seed=cfg.data_seed_code,
                    log_fragments=False,
                    max_char_len=cfg.max_char_len,
                    use_unicode_escape=cfg.use_unicode_escape,
                    tokenized_input=cfg.tokenized_input,
                    max_batches_per_shard=cfg.max_batches_per_shard,
                    max_random_skip_batches=cfg.max_random_skip_batches,
                    max_random_row_offset=cfg.max_random_row_offset,
                    rows_per_shard_limit=cfg.rows_per_shard_limit,
                    min_tokens=cfg.min_tokens,
                    fragment_filter=filt_high,
                )
                gens = [gen_low, gen_high]
                pattern = [0, 0, 0, 1]  # low:high = 3:1
                ptr = 0
                while gens:
                    idx_gen = pattern[ptr % len(pattern)]
                    if idx_gen >= len(gens):
                        idx_gen = 0
                    try:
                        item = next(gens[idx_gen])
                        yield item
                        ptr += 1
                    except StopIteration:
                        gens.pop(idx_gen)
                        if not gens:
                            break
                        if ptr >= len(pattern):
                            ptr %= len(pattern)
                return

            # threshold를 끄면 단일 wiki만 사용
            gen1 = iter_parquet_round_robin(
                cfg.data_path,
                max_length=cfg.max_length,
                batch_rows=cfg.arrow_batch_rows,
                text_col=cfg.text_col,
                tokenizer=tokenizer,
                shuffle_buffer=0,
                seed=cfg.data_seed_code,
                log_fragments=False,
                max_char_len=cfg.max_char_len,
                use_unicode_escape=cfg.use_unicode_escape,
                tokenized_input=cfg.tokenized_input,
                max_batches_per_shard=cfg.max_batches_per_shard,
                max_random_skip_batches=cfg.max_random_skip_batches,
                max_random_row_offset=cfg.max_random_row_offset,
                rows_per_shard_limit=cfg.rows_per_shard_limit,
                min_tokens=cfg.min_tokens,
            )
            for item in gen1:
                yield item

    print(f"[data] using mixed parquet: {cfg.data_path}" + (f" + {cfg.data_path2}" if cfg.data_path2 else ""))
    train_ds = MixedIterable()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        collate_fn=lambda b: collate_batch(b, pad_id, cfg.max_length),
        num_workers=cfg.num_workers_parquet,
        prefetch_factor=2 if cfg.num_workers_parquet > 0 else None,
        pin_memory=True,
        persistent_workers=False,
    )

    global_step = start_step
    coverage_logged = False  # 유효 토큰 통계를 한 번만 찍기

    for batch in train_loader:
        global_step += 1
        tokens = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits, _ = model(tokens)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # TensorBoard 기록
        writer.add_scalar("train/loss", loss.item(), global_step)

        # 유효 토큰 커버리지 통계 (1회)
        if not coverage_logged:
            valid = (labels != -100)
            valid_per_seq = valid.sum(dim=1)
            print("=== COVERAGE DEBUG (once) ===")
            print(f"batch_size={labels.size(0)}, seq_len={labels.size(1)}")
            print("valid tokens per seq:", valid_per_seq.tolist())
            print(f"mean valid tokens: {valid_per_seq.float().mean().item():.2f}")
            coverage_logged = True

        # 강한 디버그: ce_model / ce_rand / token_acc + 토큰/레이블 내용 확인
        if cfg.log_loss_every and global_step % cfg.log_loss_every == 0:
            with torch.no_grad():
                logits_det = logits.detach()
                vocab_size = logits_det.size(-1)

                ce_model = F.cross_entropy(
                    logits_det.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                ).item()

                rand_logits = torch.randn_like(logits_det)
                ce_rand = F.cross_entropy(
                    rand_logits.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                ).item()

                preds = logits_det.argmax(dim=-1)
                mask = labels != -100
                num_tokens = int(mask.sum().item())
                if num_tokens > 0:
                    correct = (preds[mask] == labels[mask]).sum().item()
                    token_acc = 100.0 * correct / num_tokens
                else:
                    correct = 0
                    token_acc = 0.0

                print(
                    f"[DEBUG] step {global_step} "
                    f"loss={loss.item():.6f} "
                    f"ce_model={ce_model:.6f} ce_rand={ce_rand:.6f} "
                    f"token_acc={token_acc:.2f}% num_tokens={num_tokens}"
                )

                # 첫 시퀀스 디코드/분포 확인
                b0 = 0
                inp0 = tokens[b0].detach().cpu()
                lbl0 = labels[b0].detach().cpu()
                pred0 = preds[b0].detach().cpu()
                valid_mask0 = lbl0 != -100

                print("=== DEBUG SAMPLE (seq 0) ===")
                print("input_ids[0,:64] =", inp0[:64].tolist())
                print("labels[0,:64]    =", lbl0[:64].tolist())
                print("preds [0,:64]    =", pred0[:64].tolist())
                print("valid_mask[:64]  =", valid_mask0[:64].tolist())

                ids_text = inp0[valid_mask0].tolist()
                try:
                    decoded = tokenizer.decode(ids_text, skip_special_tokens=True)
                    print("decoded input[0]:", repr(decoded[:300]))
                except Exception as e:
                    print("decode error:", e)

                valid_lbl = lbl0[valid_mask0]
                if valid_lbl.numel() > 0:
                    uniq, cnt = torch.unique(valid_lbl, return_counts=True)
                    topk = torch.topk(cnt, k=min(10, cnt.numel()))
                    print("=== LABEL HIST (seq0, top 10) ===")
                    for i in range(topk.values.numel()):
                        tok_id = int(uniq[topk.indices[i]])
                        c = int(topk.values[i])
                        print(f"token_id={tok_id} count={c}")
                else:
                    print("=== LABEL HIST (seq0) === no valid labels ===")

        # 체크포인트 저장
        if global_step % cfg.save_every == 0:
            ckpt_path = ckpt_dir / f"step_{global_step}.pt"
            torch.save(
                {
                    "step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"[ckpt] saved: {ckpt_path}")

        # 프리뷰 생성
        if cfg.enable_preview and global_step % cfg.preview_every == 0:
            try:
                preview = preview_generation(
                    model,
                    tokenizer,
                    device,
                    prompt=cfg.prompt_text,
                    max_new_tokens=cfg.preview_max_new_tokens,
                )
                writer.add_text("preview", preview, global_step)
                print(f"[preview] step {global_step}: {preview[:200]}")
            except Exception as e:
                print(f"[preview error] {e}")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
