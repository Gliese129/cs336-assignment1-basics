import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import regex as re
from typing import BinaryIO, Tuple


# training_file = "data/TinyStoriesV2-GPT4-train.txt"
training_file = "data/smoke.txt"
corpus_file = "data/corpus.txt"
num_processes = 4
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_one_chunk(file: str, start: int, end: int, special_token: str) -> dict:
    cnt = 0
    with open(file, "rb") as fin:
        fin.seek(start)
        chunk = fin.read(end - start).decode("utf-8", errors="ignore")
        chunk = chunk.replace(special_token, "")
        corp = {}
        for match in re.finditer(PAT, chunk):
            word = match[0]
            if word not in corp:
                corp[word] = 1
                cnt += 1
            else:
                corp[word] += 1
    return corp


if __name__ == "__main__":
    corpus = {}
    all_boundaries = []

    pool = ProcessPoolExecutor(max_workers=num_processes)
    with open(training_file, "rb") as f:

        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            all_boundaries.append((start, end))
            # Run pre-tokenization on your chunk and store the counts for each pre-token

    ftr = []
    for start, end in all_boundaries:
        ftr.append(pool.submit(process_one_chunk, training_file, start, end, "<|endoftext|>"))

    for ft in as_completed(ftr):
        single_corp = ft.result()
        for w, c in single_corp.items():
            if w not in corpus:
                corpus[w] = c
            else:
                corpus[w] += c

    pool.shutdown()
    with open(corpus_file, "w") as f:
        json.dump(corpus, f)