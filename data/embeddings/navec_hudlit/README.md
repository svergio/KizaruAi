---
tags:
- embeddings
- static-embeddings
pipeline_tag: feature-extraction
library_name: sentence-transformers
license: mit
language:
- ru
---

# Natasha/Navec hudlit_v1 Model Card

This repository provides a Sentence-Transformers version of the **Natasha/Navec** `hudlit_v1` word embeddings model.

The underlying Navec embedding weights are unchanged. This revision adds an explicit `Normalize` module after `StaticEmbedding`, so `model.encode(...)` returns L2-normalized sentence embeddings by default.

## Source

The original word embeddings come from the **Navec** project:

- Repository: https://github.com/natasha/navec
- Authors: Natasha NLP project
- License: MIT License

Navec is a compact and efficient set of Russian word embeddings trained on Russian corpora.

## Usage

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BorisTM/natasha_navec_hudlit_v1_12B_500K_300d_100q")

sentences = [
    "Сегодня хорошая погода.",
    "На улице солнечно.",
    "Команда выиграла матч.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 300)
print(np.linalg.norm(embeddings, axis=1))  # close to 1.0

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```

## Results

Results on [MTEB (rus, v1.1)](https://github.com/embeddings-benchmark/mteb), evaluated with normalized embeddings. Scores are percentages.

| Task | `navec_hudlit_v1_12B_500K_300d_100q` | [navec_news_v1_1B_250K_300d_100q](https://huggingface.co/BorisTM/natasha_navec_news_v1_1B_250K_300d_100q) |
|------|----------------------------------|--------------------------------|
| **Mean (Task, 23 tasks)** | 36.94 | 36.31 |
| **Mean (Task Type)** | 34.59 | 34.23 |
| CEDRClassification | 34.14 | 32.81 |
| GeoreviewClassification | 33.09 | 34.03 |
| GeoreviewClusteringP2P | 34.10 | 28.78 |
| HeadlineClassification | 55.33 | 63.47 |
| InappropriatenessClassification | 53.39 | 53.15 |
| KinopoiskClassification | 45.25 | 44.89 |
| MassiveIntentClassification | 48.54 | 43.86 |
| MassiveScenarioClassification | 55.20 | 49.88 |
| MIRACLReranking | 10.88 | 10.88 |
| MIRACLRetrievalHardNegatives.v2 | 1.75 | 1.60 |
| RiaNewsRetrievalHardNegatives.v2 | 15.43 | 23.47 |
| RuBQReranking | 38.00 | 37.71 |
| RuBQRetrieval | 5.80 | 5.09 |
| RUParaPhraserSTS | 41.38 | 41.12 |
| RuReviewsClassification | 49.35 | 48.80 |
| RuSciBenchGRNTIClassification | 43.63 | 40.54 |
| RuSciBenchGRNTIClusteringP2P | 40.94 | 38.43 |
| RuSciBenchOECDClassification | 35.62 | 32.70 |
| RuSciBenchOECDClusteringP2P | 36.89 | 33.98 |
| SensitiveTopicsClassification | 19.49 | 18.57 |
| STS22 | 50.20 | 51.57 |
| TERRa | 52.57 | 53.99 |
| RuSTSBenchmarkSTS | 48.59 | 45.84 |

Evaluation artifacts for this update are stored locally in the article workspace under `data/metrics/navec_baselines_mteb_rus_v1_1_summary.csv` and `data/metrics/navec_baselines_mteb_rus_v1_1_task_scores.csv`.

## License

MIT

## Contact

quelquemath@gmail.com

tg: @btmalov
