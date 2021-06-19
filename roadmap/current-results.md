# Current results

Keep up-to-date and mark best approach bold.

| Approach | mAP (strict) | mAP (relaxed) |
| --- | --- | --- |
| term overlap (no preprocessing) | 0.52 | 0.70 |
| term overlap (stemming) | 0.60 | 0.75 |
| term overlap (stemming, stop words) | **0.64** | 0.80 |
| term overlap (stemming, stop words, synonyms, antonyms) | **0.64** | **0.82** |
| regression (C=1, TF/IDF) | 0.32 | 0.55 |
| regression (C=14, BOW) | 0.44 | 0.70 |
| regression (C=14, BOW, POS) | 0.47 | 0.66 |
| SVC (BOW) | 0.46 | 0.70 |
| SVC (BOW, POS) | 0.48 | 0.74 |
| ensemble (LG=0.55, SVC=0.45, BOW) | 0.43 | 0.64 |
| ensemble (LG=0.55, SVC=0.45, BOW, POS) | 0.50 | 0.70 |
| ensemble (LG=0.45, SVC=0.55, BOW) | 0.45 | 0.65 |
| ensemble (LG=0.45, SVC=0.55, BOW, POS) | 0.51 | 0.71 |
| BiLSTM (GloVe embeddings) | 0.27 | 0.50 |