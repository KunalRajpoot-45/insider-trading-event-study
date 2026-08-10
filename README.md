# Insider Trading Event Study

When a CEO or director buys or sells shares of their own company, they have to report the trade to the SEC, and the information becomes public soon after. This project looks at a simple question: does the market react when these insider trades are disclosed?

I use an event study to measure how stock prices behave around the time of these trades and then use regression analysis to see which factors matter most—such as who made the trade, how large it was, and whether they bought or sold. The project is a practical application of statistical inference and regression to study how markets respond to insider trading.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You can install the required packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Contains the Jupyter notebooks for executing the project pipeline.
- `src/`: Python modules used by the notebooks for data downloading and event study logic.
- `report/`: Output directory for generated plots.

### Usage Instructions

The research pipeline is split into four Jupyter Notebooks that must be run sequentially:

1. **`notebooks/01_data_collection.ipynb`**: 
   Downloads SEC Form 4 insider transaction bulk files and matching daily stock prices via `yfinance`. Run this notebook first to populate `data/raw/` and `data/processed/`.

2. **`notebooks/02_market_model.ipynb`**: 
   Estimates the market model regression (alpha and beta) for each firm using the S&P 500 benchmark over an estimation window.

3. **`notebooks/03_event_study.ipynb`**: 
   Computes Cumulative Abnormal Returns (CARs) for the event window [-5, +5] days relative to the transaction dates and runs statistical hypothesis tests (t-tests, Wilcoxon). This will also generate a CAR plot in the `report/` folder.

4. **`notebooks/04_cross_sectional_regression.ipynb`**: 
   Performs a cross-sectional OLS regression to analyze the relationship between CAR and various trade/firm characteristics, including robust diagnostic checks (Breusch-Pagan for heteroskedasticity, VIF for multicollinearity).

To launch the notebooks, simply navigate to the project root and run:

```bash
jupyter notebook
```

Navigate into the `notebooks/` directory inside Jupyter and execute them in order.
