# Working Memory Task Code

This directory hosts the source code for the working memory (WM) task used in our paper.

## Reproducing the Figures

1. Run `run_multiple_trial.py` to train the model over multiple trials.
2. Then run `analyze_multiple_run.py` to aggregate the results and generate the figures.

Running these scripts sequentially will reproduce the figures presented in the paper.

## Required Environment

- Python 3.8 or later
- TensorFlow 2.x
- NumPy
- pandas
- matplotlib
- seaborn
- SciPy

Using the GPU-enabled TensorFlow package speeds up training. For example:

```bash
pip install tensorflow
```

Install the other libraries via `pip` as well.

## Files Overview
- `tasks.py` – Generates input/output pairs for various tasks.
- `rnnmodel.py` – Implementation of the leaky RNN model.
- `mine.py` – Mutual information neural estimator.
- `wmtask_train.py` – Main Training loop for the RNN and the MI network.
- Parameter files: `gp_wm.py`, `gp_mfs.py`, `gp_rossler.py`, and `parameters.py`.
- Utility scripts: `run_once.py`, `run_multiple_trial.py`, `analyze_multiple_run.py`.
