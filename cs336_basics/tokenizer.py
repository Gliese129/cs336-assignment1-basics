import json
from dataclasses import dataclass
from typing import Tuple, TypedDict
import regex as re


class Tokenizer:
    vocab_size: int
    special_tokens: list[str]
    vocab: dict[int, str]
    token_to_id: dict[str, int]

    def __init__(self, vocab: dict[int, str], special_tokens: list[str]):
        self.special_tokens = special_tokens
        self.vocab = vocab

    def from_file(self, file: str):
        with open(file, "r") as f:
            data: dict = json.loads(f.read())
            self.special_tokens = data.get("special_tokens", [])
            self.vocab = data.get("vocab", {})
        self.vocab_size = len(self.vocab)
        assert self.vocab_size > 0

    def save(self, file: str):
        data = {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab
        }
        with open(file, "w") as f:
            json.dump(data, f)


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> Tuple[dict[int,bytes], list[tuple[bytes, bytes]]]:
    corpus_ = _pre_tokenization(input_path)
    merge_record = []
    class PairRecord(TypedDict):
        cnt: int
        usage: set[tuple[int]]

    corpus: dict[tuple[int], int] = {
        tuple(w.encode("utf-8")): c
        for w, c in corpus_.items()
    }
    cnt = len(special_tokens)
    cnt += 256
    # merge
    while cnt < vocab_size:
        records: dict[Tuple[int, int], PairRecord] = {}
        for w, c in corpus.items():
            for i in range(len(w) - 1):
                p = (w[i], w[i+1])
                if p not in records:
                    records[p] = {
                        'cnt': 0,
                        'usage': set()
                    }
                records[p]['cnt'] += c
                records[p]['usage'].add(w)
        best_pair, best_record = max(
            records.items(),
            key=lambda item: (item[1]["cnt"], item[0])
        )
        merge_record.append(best_pair)
        for w_old in records[best_pair]['usage']:
            w_new = []
            for i in range(len(w_old) - 1):
                if (w_old[i], w_old[i+1]) == best_pair:
                    w_new.append(cnt)
                else:
                    w_new.append(w_old[i])
            corpus[tuple(w_new)] = corpus[w_old]
            del corpus[w_old]
        cnt += 1
    # rebuild vocab
    vocab: dict[int, bytes] = {}
    l = 0
    for st in special_tokens:
        vocab[l] = st.encode("utf-8")
        l += 1
    for _ in range(256):
        vocab[l] = bytes([i])
        l += 1
    for r in merge_record:
        # in order, there will not be further id appear before its idx
        a, b = vocab[r[0]], vocab[r[1]]
        vocab[l] = a + b
        l += 1

    return vocab, [
        (vocab[r[0]], vocab[r[1]]) for r in merge_record
    ]


def _pre_tokenization(input_file: str) -> dict[str, int]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from cs336_basics.pretokenization import find_chunk_boundaries, process_one_chunk

    num_processes = 8
    corpus = {}
    all_boundaries = []

    pool = ProcessPoolExecutor(max_workers=num_processes)
    with open(input_file, "rb") as f:

        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            all_boundaries.append((start, end))

    ftr = []
    for start, end in all_boundaries:
        ftr.append(pool.submit(process_one_chunk, input_file, start, end, "<|endoftext|>"))

    for ft in as_completed(ftr):
        single_corp = ft.result()
        for w, c in single_corp.items():
            if w not in corpus:
                corpus[w] = c
            else:
                corpus[w] += c

    pool.shutdown()
    return corpus

if __name__ == '__main__':
    v, m = train_bpe("../data/smoke.txt", 300, [])
    print(v, m)