# WECHSEL
Code for WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models.

ArXiv: https://arxiv.org/abs/2112.06598

Models from the paper will be available on the Huggingface Hub.

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

# use `model` and `target_tokenizer` to continue training in Swahili!
```

## Bilingual dictionaries

We distribute 3276 bilingual dictionaries from English to other languages for use with WECHSEL in `dicts/`.

## Citation

Please cite WECHSEL as

```
@misc{minixhofer2021wechsel,
      title={WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models}, 
      author={Benjamin Minixhofer and Fabian Paischer and Navid Rekabsaz},
      year={2021},
      eprint={2112.06598},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
