from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from cs336_basics.pretokenization import find_chunk_boundaries, process_one_chunk

training_file = "data/TinyStoriesV2-GPT4-train.txt"
corpus_file = "data/corpus.txt"
num_processes = 4


if __name__ == "__main__":
    corpus = {}

    with open(training_file, "rb") as f, ProcessPoolExecutor(max_workers=num_processes) as pool:

        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        ftr = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            ftr.append(pool.submit(process_one_chunk, chunk, "<|endoftext|>"))

        for ft in as_completed(ftr):
            single_res = ft.result()
            for w, c in single_res:
                if w not in corpus:
                    corpus[w] = c
                else:
                    corpus[w] += c

    with open(corpus_file, "w") as f:
        json.dump(corpus, f)
