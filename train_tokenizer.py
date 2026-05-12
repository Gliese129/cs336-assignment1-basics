import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from cs336_basics.pretokenization import find_chunk_boundaries, process_one_chunk

# training_file = "data/TinyStoriesV2-GPT4-train.txt"
training_file = "data/smoke.txt"
corpus_file = "data/corpus.txt"
tmp_folder = "tmp/"
num_processes = 4


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

    os.makedirs(tmp_folder, exist_ok=True)
    ftr = []
    for idx, (start, end) in enumerate(all_boundaries):
        tmp_file = f"{tmp_folder}corp_part_{idx}.tmp"
        ftr.append(pool.submit(process_one_chunk, training_file, start, end, "<|endoftext|>", tmp_file))

    for ft in as_completed(ftr):
        cnt_part, tmp_file = ft.result()
        with open(tmp_file, "r") as f:
            single_corp: dict = json.loads(f.read())
        for w, c in single_corp.items():
            if w not in corpus:
                corpus[w] = c
            else:
                corpus[w] += c

    pool.shutdown()
    with open(corpus_file, "w") as f:
        json.dump(corpus, f)
