# Digital Eye: original vs fine-tuned recogniser

Numbers only. Same documents, same Digital Eye code and options; only `paddle.rec_model_dir` differs.

Documents: 45

| Metric | Original | Fine-tuned | Better is |
|---|---:|---:|---|
| False acceptance (wrong value returned as trustworthy) | 0.4% | 0.0% | lower |
| Field exact match | 84.0% | 98.7% | higher |
| Critical fields exact match (permit, plot, area) | 71.3% | 97.2% | higher |
| Required-field coverage | 87.2% | 99.5% | higher |
| Wrong values that were flagged | 97.2% | 100.0% | higher |
| Correct values still flagged for review | 14.3% | 14.9% | lower |
| Coordinate values within tolerance | 96.1% | 96.1% | higher |
| Page text CER | — | — | lower |
| Seconds per document | 3.74 | 3.969 | lower |

| Language | Fields | Correct (orig → tuned) | False acceptances (orig → tuned) | Flagged (orig → tuned) |
|---|---:|---:|---:|---:|
| ar | 72 | 37 → 70 | 1 → 0 | 38 → 14 |
| mixed | 72 | 72 → 72 | 0 → 0 | 2 → 0 |
| en | 72 | 71 → 71 | 0 → 0 | 13 → 13 |
