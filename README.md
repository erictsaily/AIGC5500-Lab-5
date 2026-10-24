# Attention Against Recurrence
Attention built from scratch, then tested against recurrence, positions
and normalization on small tasks with known answers.

## Setup
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Run
Open lab05.ipynb, select the python environment, ensure you have installed all the librearies, and run all the cells from top to bottom

## Prediction (written before running the mixed-length training)
I predict mixed-length training will improve accuracy at length 32 for both learned and sinusoidal positions because the models will see more sequence lengths. Additionally, I think sinusoidal positions will improve more than learned.

## Results
### Verification: what matched PyTorch, and the parameter counts.
My scaled dot-product attention matched PyTorch with and without a causal mask for seeds 0, 1, and 2. My multi-head attention also matched PyTorch for self-attention, causal self-attention, and cross-attention.
The parameter counts were: 4224 for 1, 2, 4, and 8 heads.

### Recall: accuracy per seed at each length, and the weight on step 0.
| Length | Seed | Attention Accuracy | Weight on Step 0 | LSTM Accuracy |
|---|---|---|---|---|
| 30 | 0 | 1.000 | 0.730 | 1.000 |
| 30 | 1 | 1.000 | 0.742 | 1.000 |
| 30 | 2 | 1.000 | 0.674 | 1.000 |
| 100 | 0 | 1.000 | 0.738 | 0.252 |
| 100 | 1 | 1.000 | 0.746 | 1.000 |
| 100 | 2 | 1.000 | 0.658 | 0.499 |
| 400 | 0 | 1.000 | 0.741 | 0.245 |
| 400 | 1 | 1.000 | 0.746 | 0.244 |
| 400 | 2 | 1.000 | 0.663 | 0.245 |

![Delayed Recall Accuracy](recall.png)  

As seen in the graph above, attention achieved 100% accuracy across all lengths and seeds, while the LSTM became less reliable at longer lengths.

### Positions: accuracy per seed at each length, before and after mixing lengths.
Fixed-Length Training (Length 16)
| Positions | Seed | Accuracy at 16 | Accuracy at 32 |
|---|---|---|---|
| None | 0 | 0.552 | 0.503 |
| None | 1 | 0.523 | 0.484 |
| None | 2 | 0.447 | 0.498 |
| Learned | 0 | 1.000 | 0.620 |
| Learned | 1 | 1.000 | 0.601 |
| Learned | 2 | 1.000 | 0.623 |
| Sinusoidal | 0 | 1.000 | 0.643 |
| Sinusoidal | 1 | 1.000 | 0.664 |
| Sinusoidal | 2 | 1.000 | 0.750 |

Mixed-Length Training (Lengths 8, 16, 24)
| Positions | Seed | Accuracy at 16 | Accuracy at 32 | Accuracy at 48 |
|---|---|---|---|---|
| None | 0 | 0.494 | 0.516 | 0.493 |
| None | 1 | 0.512 | 0.479 | 0.508 |
| None | 2 | 0.539 | 0.507 | 0.481 |
| Learned | 0 | 0.975 | 0.703 | 0.558 |
| Learned | 1 | 1.000 | 0.718 | 0.549 |
| Learned | 2 | 1.000 | 0.706 | 0.543 |
| Sinusoidal | 0 | 1.000 | 0.893 | 0.615 |
| Sinusoidal | 1 | 1.000 | 0.901 | 0.580 |
| Sinusoidal | 2 | 1.000 | 0.929 | 0.750 |

As predicted, mixed-length training improved accuracy at length 32 for both learned and sinusoidal positions. For sinusoidal, all the seeds were roughly 20% more accurate at 32, while learned was roughly 10% more accurate.

### Stacks and timing: accuracy per seed, and the timing table.
Normalization
| Norm | Warmup | Seed | Accuracy at 16 |
|---|---|---|---|
| Post | 0 | 0 | 0.552 |
| Post | 0 | 1 | 0.552 |
| Post | 0 | 2 | 0.448 |
| Post | 100 | 0 | 1.000 |
| Post | 100 | 1 | 0.997 |
| Post | 100 | 2 | 1.000 |
| Pre | 0 | 0 | 1.000 |
| Pre | 0 | 1 | 1.000 |
| Pre | 0 | 2 | 1.000 |

Timing
| Length | Seed | LSTM (seconds) | Attention (seconds) |
|---|---|---|---|
| 100 | 0 | 0.0064 | 0.0116 |
| 100 | 1 | 0.0046 | 0.0105 |
| 100 | 2 | 0.0048 | 0.0104 |
| 200 | 0 | 0.0100 | 0.0306 |
| 200 | 1 | 0.0102 | 0.0283 |
| 200 | 2 | 0.0081 | 0.0308 |
| 400 | 0 | 0.0169 | 0.1102 |
| 400 | 1 | 0.0194 | 0.1139 |
| 400 | 2 | 0.0214 | 0.1165 |
| 800 | 0 | 0.0380 | 0.4576 |
| 800 | 1 | 0.0460 | 0.4599 |
| 800 | 2 | 0.0435 | 0.4327 |

## Conclusion
Attention performed better than the LSTM on longer recall sequences because it could directly access earlier information without passing it through every intermediate step. Pre-norm and post-norm with warmup also trained successfully, while post-norm without warmup remained around 50% accuraty.

However, the cost was that attention was significantly slower than the LSTM at every tested length, with the gap increasing as sequences grew longer due to attention's computational cost.

## Known limitations
These experiments only used small synthetic tasks, three random seeds, and fixed training budgets. The results may not generalize to larger datasets or real-world problems. Additonally, the timing measurements were timed on my relatively old laptop, so they be a lot slower than running the code on a more modern system.

With more time and resources, I would test more seeds, longer sequences, and more complex datasets. I would also compare different model sizes and optimized attention implementations to better understand their performance and efficiency.