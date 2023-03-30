# WECHSEL
Code for WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models published at NAACL2022.

Paper: https://aclanthology.org/2022.naacl-main.293/

<p align="center">
<img src="https://user-images.githubusercontent.com/13353204/165908328-3f3217ad-f08e-4051-8648-3e49b26f3b71.png" width="400"  />
</p>

Models from the paper are available on the HuggingFace Hub:

- [`roberta-base-wechsel-french`](https://huggingface.co/benjamin/roberta-base-wechsel-french)
- [`roberta-base-wechsel-german`](https://huggingface.co/benjamin/roberta-base-wechsel-german)
- [`roberta-base-wechsel-chinese`](https://huggingface.co/benjamin/roberta-base-wechsel-chinese)
- [`roberta-base-wechsel-swahili`](https://huggingface.co/benjamin/roberta-base-wechsel-swahili)
- [`gpt2-wechsel-french`](https://huggingface.co/benjamin/gpt2-wechsel-french)
- [`gpt2-wechsel-german`](https://huggingface.co/benjamin/gpt2-wechsel-german)
- [`gpt2-wechsel-chinese`](https://huggingface.co/benjamin/gpt2-wechsel-chinese)
- [`gpt2-wechsel-swahili`](https://huggingface.co/benjamin/gpt2-wechsel-swahili)

## Installation

We distribute a Python Package via PyPI:

```
pip install wechsel
```

Alternatively, clone the repository, install `requirements.txt` and run the code in `wechsel/`.

## Example usage

Transferring English `roberta-base` to Swahili:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from wechsel import WECHSEL, load_embeddings

source_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

target_tokenizer = source_tokenizer.train_new_from_iterator(
    load_dataset("oscar", "unshuffled_deduplicated_sw", split="train")["text"],
    vocab_size=len(source_tokenizer)
)

wechsel = WECHSEL(
    load_embeddings("en"),
    load_embeddings("sw"),
    bilingual_dictionary="swahili"
)

target_embeddings, info = wechsel.apply(
    source_tokenizer,
    target_tokenizer,
    model.get_input_embeddings().weight.detach().numpy(),
)

model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
model.config.vocab_size = len(target_embeddings)

# use `model` and `target_tokenizer` to continue training in Swahili!
```

## Bilingual dictionaries

We distribute 3276 bilingual dictionaries from English to other languages for use with WECHSEL in `dicts/`.

## Citation

Please cite WECHSEL as

```
@inproceedings{minixhofer-etal-2022-wechsel,
    title = "{WECHSEL}: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models",
    author = "Minixhofer, Benjamin  and
      Paischer, Fabian  and
      Rekabsaz, Navid",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.293",
    pages = "3992--4006",
    abstract = "Large pretrained language models (LMs) have become the central building block of many NLP applications. Training these models requires ever more computational resources and most of the existing models are trained on English text only. It is exceedingly expensive to train these models in other languages. To alleviate this problem, we introduce a novel method {--} called WECHSEL {--} to efficiently and effectively transfer pretrained LMs to new languages. WECHSEL can be applied to any model which uses subword-based tokenization and learns an embedding for each subword. The tokenizer of the source model (in English) is replaced with a tokenizer in the target language and token embeddings are initialized such that they are semantically similar to the English tokens by utilizing multilingual static word embeddings covering English and the target language. We use WECHSEL to transfer the English RoBERTa and GPT-2 models to four languages (French, German, Chinese and Swahili). We also study the benefits of our method on very low-resource languages. WECHSEL improves over proposed methods for cross-lingual parameter transfer and outperforms models of comparable size trained from scratch with up to 64x less training effort. Our method makes training large language models for new languages more accessible and less damaging to the environment. We make our code and models publicly available.",
}
```

## Acknowledgments

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC). We thank Andy Koh and Artus Krohn-Grimberghe for providing additional computational resources. The ELLIS Unit Linz, the LIT AI Lab, the Institute for Machine Learning, are supported by the Federal State Upper Austria. We thank the project INCONTROL-RL (FFG-881064).
