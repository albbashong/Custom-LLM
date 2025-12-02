import os
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

OUTPUT_DIR = "/home/test/final_155gb_stable_data/code"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 5_000
ROWS_PER_FILE = 300_000
file_index = 15
row_count = 0

writer = None

# ⭐ The Stack v1 C++ 데이터셋 (streaming=True 정상)
ds = load_dataset("bigcode/the-stack", data_dir="data/c-sharp", split="train",streaming=True)

buffer_source = []
buffer_text = []

def flush_buffer():
    global writer, file_index
    if not buffer_source:
        return

    table = pa.Table.from_pydict({
        "source": buffer_source,
        "text": buffer_text,
    })

    if writer is None:
        out_path = os.path.join(OUTPUT_DIR, f"part-{file_index:05d}.parquet")
        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        print(f"[OPEN] {out_path}")

    writer.write_table(table)

    buffer_source.clear()
    buffer_text.clear()

for example in ds:

    # v1 구조
    text = example.get("content", "")
    source = example.get("repository_name", "unknown")

    # 빈 파일 제외 가능 (원하면)
    # if not text.strip():
    #     continue

    buffer_source.append(source)
    buffer_text.append(text)
    row_count += 1

    if len(buffer_source) >= CHUNK_SIZE:
        flush_buffer()

    if row_count >= ROWS_PER_FILE:
        flush_buffer()
        writer.close()
        print(f"[CLOSE] part-{file_index:05d}.parquet (rows = {row_count})")

        writer = None
        row_count = 0
        file_index += 1

flush_buffer()
if writer:
    writer.close()
    print(f"[CLOSE] part-{file_index:05d}.parquet (rows = {row_count})")
