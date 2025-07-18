# Fair-Test Thesis Project

This repository contains the code used for my thesis research on fairness in federated learning. The full thesis document is available in [`Thesis.pdf`](Thesis.pdf).

## Project structure

- `fair-test/` – Python package implementing experiments with [Flower](https://flower.ai/) and scikit-learn.
- `test.py` – Stand-alone script demonstrating fairness metrics on the COMPAS dataset.
- `Thesis.pdf` – The thesis report.

## Installation

Install the package and its dependencies using `pip`:

```bash
pip install -e fair-test
```

Or change into the folder first:

```bash
cd fair-test
pip install -e .
```

## Running the Flower simulation

From inside the `fair-test` directory run:

```bash
flwr run .
```

This launches a local simulation that trains a logistic regression model and evaluates several fairness metrics.

## Running the standalone example

Run the example script directly from the repository root:

```bash
python test.py
```

The script loads the COMPAS dataset, applies optional reweighting strategies and reports fairness statistics.

## License

This project is released under the terms of the [MIT License](LICENSE).
