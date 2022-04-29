# WECHSEL
Code for WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models to appear in NAACL2022.

arXiv: https://arxiv.org/abs/2112.06598

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

## Acknowledgments

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC). We thank Andy Koh and Artus Krohn-Grimberghe for providing additional computational resources. The ELLIS Unit Linz, the LIT AI Lab, the Institute for Machine Learning, are supported by the Federal State Upper Austria. We thank the project INCONTROL-RL (FFG-881064).
