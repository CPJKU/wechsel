from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from dataclasses import dataclass
from datasets import load_dataset
from tqdm.auto import tqdm
import json
from pathlib import Path
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
from scipy.linalg import orthogonal_procrustes
import gc
import datasets


@dataclass
class Args:
    model_name: str
    dataset_name: str
    dataset_config_name: str
    output_dir: str
    source_word_vector_path: str
    target_word_vector_path: str
    align_dict_path: str
    skip_data_download: bool = False
    reduce_tokenizer_train_size: bool = False
    subsample_size_mb: int = 1024
    valid_percent = 0.1
    max_n_word_vectors: int = None
    new_tokenizer_name: str = None


def softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def get_subword_embeddings_in_word_embedding_space(
    tokenizer, model, max_n_word_vectors=None, use_subword_info=False
):
    words, freqs = model.get_words(include_freq=True, on_unicode_error="ignore")

    if max_n_word_vectors is None:
        max_n_word_vectors = len(words)

    sources = {}
    embs_matrix = np.zeros((len(tokenizer), model.get_dimension()))

    if use_subword_info:
        for i in range(len(tokenizer)):
            token = tokenizer.decode(i).strip()

            # `get_word_vector` returns zeros if not able to interpolate
            embs_matrix[i] = model.get_word_vector(token)
    else:
        embs = {value: [] for value in tokenizer.get_vocab().values()}

        for i, word in tqdm(
            enumerate(words[:max_n_word_vectors]), total=max_n_word_vectors
        ):
            for tokenized in [
                tokenizer.encode(word, add_special_tokens=False),
                tokenizer.encode(" " + word, add_special_tokens=False),
            ]:
                for token_id in set(tokenized):
                    embs[token_id].append(i)

        for i in range(len(embs_matrix)):
            if len(embs[i]) == 0:
                continue

            weight = np.array([freqs[idx] for idx in embs[i]])
            weight = weight / weight.sum()

            vectors = [model.get_word_vector(words[idx]) for idx in embs[i]]

            sources[tokenizer.convert_ids_to_tokens([i])[0]] = embs[i]
            embs_matrix[i] = (np.stack(vectors) * weight[:, np.newaxis]).sum(axis=0)

    return embs_matrix, sources


def create_target_embeddings(
    source_subword_embeddings,
    target_subword_embeddings,
    source_tokenizer,
    target_tokenizer,
    source_matrix,
    neighbors=10,
    temperature=0.1,
):
    def get_n_closest(token_id, similarities, top_k):
        if (target_subword_embeddings[token_id] == 0).all():
            return None

        best_indices = np.argpartition(similarities, -top_k)[-top_k:]
        best_tokens = source_tokenizer.convert_ids_to_tokens(best_indices)

        best = sorted(
            [
                (token, similarities[idx])
                for token, idx in zip(best_tokens, best_indices)
            ],
            key=lambda x: -x[1],
        )

        return best

    source_vocab = source_tokenizer.vocab

    target_matrix = np.zeros((len(target_tokenizer), source_matrix.shape[1]))

    mean, std = (
        source_matrix.mean(0),
        source_matrix.std(0),
    )

    random_fallback_matrix = np.random.RandomState(1234).normal(
        mean, std, (len(target_tokenizer.vocab), source_matrix.shape[1])
    )

    batch_size = 1024
    n_matched = 0

    not_found = []
    sources = {}

    for i in tqdm(range(int(math.ceil(len(target_matrix) / batch_size)))):
        start, end = (
            i * batch_size,
            min((i + 1) * batch_size, len(target_matrix)),
        )

        similarities = cosine_similarity(
            target_subword_embeddings[start:end], source_subword_embeddings
        )
        for token_id in range(start, end):
            closest = get_n_closest(token_id, similarities[token_id - start], neighbors)

            if closest is not None:
                tokens, sims = zip(*closest)
                weights = softmax(np.array(sims) / temperature, 0)

                sources[target_tokenizer.convert_ids_to_tokens(token_id)] = (
                    tokens,
                    weights,
                    sims,
                )

                emb = np.zeros(target_matrix.shape[1])

                for i, close_token in enumerate(tokens):
                    emb += source_matrix[source_vocab[close_token]] * weights[i]

                target_matrix[token_id] = emb

                n_matched += 1
            else:
                target_matrix[token_id] = random_fallback_matrix[token_id]
                not_found.append(target_tokenizer.convert_ids_to_tokens([token_id])[0])

    for token in source_tokenizer.special_tokens_map.values():
        if isinstance(token, str):
            token = [token]

        for t in token:
            if t in target_tokenizer.vocab:
                target_matrix[target_tokenizer.vocab[t]] = source_matrix[
                    source_tokenizer.vocab[t]
                ]

    print(f"Matching token found for {n_matched} of {len(target_matrix)} tokens.")
    return target_matrix, not_found, sources


EMBEDDING_PATHS = {
    "roberta-base": ("embeddings", "word_embeddings"),
    "roberta-large": ("embeddings", "word_embeddings"),
    "gpt2": ("wte",),
    "tau/splinter-base-qass": ("embeddings", "word_embeddings"),
}

if __name__ == "__main__":
    parser = HfArgumentParser([Args])

    (args,) = parser.parse_args_into_dataclasses()

    subsample_size = 1024 * 1024 * args.subsample_size_mb

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # create configuration
    config = AutoConfig.from_pretrained(args.model_name)
    config.save_pretrained(output_dir)

    if not args.skip_data_download:
        dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, split="train", streaming=True
        )
        dataset_iter = iter(dataset)

        with open(output_dir / "train.json", "w") as f:
            size = 0
            bar = tqdm(total=subsample_size)

            while size < subsample_size:
                entry = next(dataset_iter)

                entry_size = len(entry["text"].encode("utf-8"))
                size += entry_size

                bar.update(entry_size)

                f.write(f"{json.dumps(entry)}\n")

        with open(output_dir / "valid.json", "w") as f:
            size = 0
            bar = tqdm(total=subsample_size * args.valid_percent)

            while size < subsample_size * args.valid_percent:
                entry = next(dataset_iter)

                entry_size = len(entry["text"].encode("utf-8"))
                size += entry_size

                bar.update(entry_size)

                f.write(f"{json.dumps(entry)}\n")

    dataset = datasets.load_dataset(
        "json", data_files=str(output_dir / "train.json"), split="train"
    )
    if args.reduce_tokenizer_train_size:
        dataset = dataset.filter(lambda _, i: i % 2 == 0, with_indices=True)

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    # train tokenizer
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.new_tokenizer_name
        if args.new_tokenizer_name is not None
        else args.model_name
    )
    target_tokenizer.additional_special_tokens = ()  # bug in splinter
    if "additional_special_tokens" in target_tokenizer.init_kwargs:
        target_tokenizer.init_kwargs.pop("additional_special_tokens")  # bug in splinter
    target_tokenizer = target_tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=len(target_tokenizer)
    )
    target_tokenizer.save_pretrained(output_dir)

    source_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, add_prefix_space=False
    )

    source_word_model = fasttext.load_model(args.source_word_vector_path)
    target_word_model = fasttext.load_model(args.target_word_vector_path)

    correspondences = []

    for line in open(args.align_dict_path):
        line = line.strip()
        try:
            source_word, target_word = line.split("\t")
        except ValueError:
            source_word, target_word = line.split()

        source_word = source_word.lower()
        target_word = target_word.lower()

        for src_w in (source_word, source_word.title()):
            for trg_w in (target_word, target_word.title()):
                src_id = source_word_model.get_word_id(src_w)
                trg_id = target_word_model.get_word_id(trg_w)

                if src_id != -1 and trg_id != -1:
                    correspondences.append(
                        [
                            source_word_model.get_word_vector(src_w),
                            target_word_model.get_word_vector(trg_w),
                        ]
                    )

    correspondences = np.array(correspondences)
    align_matrix, _ = orthogonal_procrustes(
        correspondences[:, 0], correspondences[:, 1]
    )

    source_model = AutoModel.from_pretrained(args.model_name)

    for key in EMBEDDING_PATHS[args.model_name]:
        source_model = getattr(source_model, key)

    source_matrix = source_model.weight.detach().cpu().numpy()

    for use_subword_info in (False, True):
        (
            source_subword_embeddings,
            source_subword_sources,
        ) = get_subword_embeddings_in_word_embedding_space(
            source_tokenizer,
            source_word_model,
            use_subword_info=use_subword_info,
            max_n_word_vectors=args.max_n_word_vectors,
        )
        (
            target_subword_embeddings,
            target_subword_sources,
        ) = get_subword_embeddings_in_word_embedding_space(
            target_tokenizer,
            target_word_model,
            use_subword_info=use_subword_info,
            max_n_word_vectors=args.max_n_word_vectors,
        )

        source_subword_embeddings = np.dot(
            source_subword_embeddings,
            align_matrix,
        )
        source_subword_embeddings /= (
            np.linalg.norm(source_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        target_subword_embeddings /= (
            np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )

        for neighbors, temperature in ((1, 1), (10, 0.1), (10, 1), (50, 0.1), (50, 1)):
            target_matrix, not_found, sources = create_target_embeddings(
                source_subword_embeddings,
                target_subword_embeddings,
                source_tokenizer,
                target_tokenizer,
                source_matrix.copy(),
                neighbors=neighbors,
                temperature=temperature,
            )

            np.save(
                output_dir
                / f"embeddings_{use_subword_info}_{neighbors}_{temperature}.npy",
                target_matrix,
            )
            np.save(
                output_dir
                / f"embeddings_info_{use_subword_info}_{neighbors}_{temperature}.npy",
                (not_found, sources),
            )

