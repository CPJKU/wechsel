---
language: <language>
license: mit
---

# <model_name>

Model trained with WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models.

See the code here: https://github.com/CPJKU/wechsel

And the paper here: https://arxiv.org/abs/2112.06598

## Performance

### RoBERTa

| Model | NLI Score | NER Score | Avg Score |
|---|---|---|---|
| `roberta-base-wechsel-french` | **82.43** | **90.88** | **86.65** |
| `camembert-base`   | 80.88 | 90.26 | 85.57 |


| Model | NLI Score | NER Score | Avg Score |
|---|---|---|---|
| `roberta-base-wechsel-german` | **81.79** | **89.72** | **85.76** |
| `deepset/gbert-base`   | 78.64 | 89.46 | 84.05 |

| Model | NLI Score | NER Score | Avg Score |
|---|---|---|---|
| `roberta-base-wechsel-chinese` | **78.32** | 80.55 | **79.44** |
| `bert-base-chinese`   | 76.55 | **82.05** | 79.30 |

| Model | NLI Score | NER Score | Avg Score |
|---|---|---|---|
| `roberta-base-wechsel-swahili` | **75.05** | **87.39** | **81.22** |
| `xlm-roberta-base`   | 69.18 | 87.37 | 78.28 |

### GPT2

| Model | PPL |
|---|---|
| `gpt2-wechsel-french` | **19.71** |
| `gpt2` (retrained from scratch)  | 20.47 |

| Model | PPL |
|---|---|
| `gpt2-wechsel-german` | **26.8** |
| `gpt2` (retrained from scratch)  | 27.63 |

| Model | PPL |
|---|---|
| `gpt2-wechsel-chinese` | **51.97** |
| `gpt2` (retrained from scratch)  | 52.98 |

| Model | PPL |
|---|---|
| `gpt2-wechsel-swahili` | **10.14** |
| `gpt2` (retrained from scratch)  | 10.58 |

See our paper for details.

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