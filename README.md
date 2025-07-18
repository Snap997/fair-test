# Fair-Test Thesis Code

This repository contains the source code used in my thesis work. The accompanying document can be found as [thesis.pdf](thesis.pdf).

The implementation is based on [Flower](https://flower.ai/) and [scikit-learn](https://scikit-learn.org/) to run federated fairness experiments using logistic regression on datasets such as COMPAS.

## Repository layout

- `fair-test/` – Python package containing the Flower server and client apps.
- `test.py` – Standalone example showing local dataset weighting logic.
- `LICENSE` – Project license.
- `thesis.pdf` – Thesis document (add the file here).

## Installation

1. Create a virtual environment (optional but recommended).
2. Install the project in editable mode along with its dependencies:

```bash
pip install -e ./fair-test
```

## Running the simulation

Inside the `fair-test` directory execute:

```bash
flwr run .
```

This launches a local simulation with the configuration defined in `pyproject.toml`. You can modify the parameters (number of rounds, reweighting strategy, etc.) in that file.

## Reproducing results

The code in `test.py` demonstrates how fairness weights can be computed locally on the COMPAS dataset. See the comments in the file for details.

For more information, please refer to `thesis.pdf`.
