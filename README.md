# From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models

## ✨ Introduction

This repository contains the official implementation of our NeurIPS submission, demonstrating how TabPFN-v2 achieves superior performance in time series forecasting compared to specialized models. Our goal is to ensure complete reproducibility of our results and provide the research community with accessible tools for further exploration.

## 📋 Key Features

- Complete implementation of TabPFN-TS
- A Jupyter notebook for demonstration of TabPFN-TS
- Detailed documentation for reproducing our experiments

## 🚀 Getting Started

Simply install the dependencies via:

```bash
pip install -r requirements.txt
```

Now, you are ready to use TabPFN-TS for your time series forecasting tasks!
We have provided a Jupyter notebook for demonstration of TabPFN-TS (see `demo.ipynb`).

## 📊 Reproducing our experiments
### Setup
After installing the dependencies for TabPFN-TS, we will need to install additional dependencies as well as download the GIFT-Eval benchmark.

To do so, run the following command:

```bash
cd gift_eval

# download the data
./download_data.sh                          

# install the dependencies
pip install -r requirements-gift-eval.txt  
```

Note 1: downloading the data should take less than 5 minutes (a total of 1.66 GB).

Note 2: you might run into the following error, but it is fine. You can ignore it.
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tabpfn-client 0.1.7 requires pandas<=2.2.3,>=2.1.2, but you have pandas 2.0.0 which is incompatible.
```

### Running the experiments

As mentioned in the paper, in GIFT-Eval, there are a total of **97 benchmarking tasks**. Each benchmarking task in GIFT-Eval corresponds to a unique combination of dataset, prediction horizon (short-, medium-, or long-term), and sampling frequency (where applicable).

We have provided two helpful scripts here:
- `gift_eval/list_datasets.py`: list all the available benchmarking tasks in GIFT-Eval
- `gift_eval/evaluate.py`: run given benchmarking task

For example, to run the evaluation on `bizitobs_l2c/H`, predicting a long-term horizon, you can run the following command:

```bash
python gift_eval/evaluate.py --dataset bizitobs_l2c/H --horizon long
```

To run all experiments efficiently, you could optimize running many evaluation jobs in parallel, depending on your computational resources.

Note: TabPFN-TS supports multi-GPU inference - it internally distributes the inference workload across all available GPUs.

### Results

To reproduce the paper’s normalized and aggregated metrics, simply open and run the provided Jupyter notebook (results.ipynb), which guides you step-by-step. It includes full baseline and TabPFN-TS results, and you can also easily load your own outputs into the same workflow to regenerate the aggregated tables.

## 📋 Repository Structure

The codebase is organized into several key components:

### Core Implementation (`tabpfn_time_series/`)
- **Predictor**: Main forecasting interface
- **Features**: Comprehensive feature engineering pipeline
  - Calendar features
  - Auto-seasonal detection
  - Running index features

### Evaluation Framework (`gift_eval/`)
- Standardized evaluation on GIFT benchmark
