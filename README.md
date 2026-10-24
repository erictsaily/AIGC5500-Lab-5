# Attention Against Recurrence
Attention built from scratch, then tested against recurrence, positions
and normalization on small tasks with known answers.

## Setup
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Run
Open and run lab05.ipynb from top to bottom

## Prediction (written before running the mixed-length training)
What I expected for learned and sinusoidal positions at length 32, and why.

## Results
Verification: what matched PyTorch, and the parameter counts.
Recall: accuracy per seed at each length, and the weight on step 0.
Positions: accuracy per seed at each length, before and after mixing lengths.
Stacks and timing: accuracy per seed, and the timing table.

## Conclusion
What attention removed, what it costs, and what these measurements
do and do not establish.

## Known limitations
Anything you are aware of that does not work, or that you would improve with more time.