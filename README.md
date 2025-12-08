# Haier Europe Datathon: Sales Forecasting Pipeline

A comprehensive machine learning pipeline for forecasting product and category-level sales using advanced ensemble methods and time series cross-validation.

## Overview

This project implements a sophisticated sales forecasting system designed for Haier Europe's datathon challenge. The pipeline leverages multiple state-of-the-art machine learning models including LightGBM, XGBoost, CatBoost, and LSTM neural networks, combined through weighted ensemble strategies to deliver accurate predictions.

### Key Features

- **Ensemble Methods**: Combines LightGBM, XGBoost, CatBoost, and LSTM models for robust predictions
- **Time Series Cross-Validation**: Expanding window strategy for proper temporal evaluation
- **Automatic Hyperparameter Tuning**: Best-iteration detection for final model refit
- **Class-Based Architecture**: Clean, maintainable design with state management
- **Vectorized Recursive Forecasting**: Efficient multi-step ahead predictions
- **Weighted Ensemble Strategy**: Combines models based on OOF (Out-of-Fold) performance
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations

## Project Structure

```
├── model.py                          # Main forecasting pipeline
├── eda.ipynb                         # Exploratory data analysis notebook
├── pyproject.toml                    # Project configuration and dependencies
├── data/
│   ├── train.csv                     # Training dataset
│   ├── product_master.csv            # Product metadata
│   └── submission.csv                # Submission template
├── submissions/                      # Model predictions and submissions
├── summaries/                        # EDA summary reports
│   ├── eda_category_summary.csv
│   ├── eda_market_summary.csv
│   ├── eda_product_lifecycle.csv
│   └── eda_sku_abc_classification.csv
└── visualizations/                   # Generated plots and charts
```

## Installation

### Requirements
- Python 3.12+
- Virtual environment (recommended)

### Setup

1. **Clone the repository and navigate to the project directory**
   ```bash
   cd /path/to/haier-europe-datathon
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

## Dependencies

- **Machine Learning**: 
  - `lightgbm>=4.6.0` - Gradient boosting framework
  - `xgboost>=3.1.2` - Extreme gradient boosting
  - `catboost>=1.2.8` - Categorical feature-aware boosting
  - `scikit-learn>=1.7.2` - Preprocessing and utilities

- **Deep Learning**:
  - `torch>=2.9.1` - LSTM neural network implementation

- **Data Processing**:
  - `pandas>=2.3.3` - Data manipulation
  - `numpy` - Numerical computing

- **Visualization**:
  - `matplotlib>=3.10.7` - Plotting library
  - `seaborn>=0.13.2` - Statistical visualization

- **Time Series**:
  - `statsmodels>=0.14.5` - Statistical modeling

- **Utilities**:
  - `tqdm>=4.67.1` - Progress bars
  - `nbformat>=5.10.4` - Notebook support

## Configuration

The `PipelineConfig` dataclass in `model.py` controls all pipeline parameters:

```python
@dataclass
class PipelineConfig:
    # File Paths
    TRAIN_FILE: str = 'data/train.csv'
    PRODUCT_MASTER_FILE: str = 'data/product_master.csv'
    SUBMISSION_FILE: str = 'data/submission.csv'
    OUTPUT_FILE: str = "submissions/submission_ensemble_cv.csv"
    
    # Dates
    CALENDAR_START: pd.Timestamp = pd.to_datetime("2022-01-01")
    CALENDAR_END: pd.Timestamp = pd.to_datetime("2024-10-01")
    FORECAST_START: pd.Timestamp = pd.to_datetime("2024-11-01")
    
    # CV Settings
    CV_FOLDS: int = 3
    CV_VAL_MONTHS: int = 1
    CV_GAP_MONTHS: int = 0
    
    # Feature Engineering
    LAGS: List[int] = [1, 2, 3, 6, 12]
    ROLLING_WINDOWS: List[int] = [3, 6, 12]
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CV_FOLDS` | Number of time series cross-validation folds | 3 |
| `CV_VAL_MONTHS` | Validation set size in months | 1 |
| `LAGS` | Lag features for historical values | [1, 2, 3, 6, 12] |
| `ROLLING_WINDOWS` | Rolling window sizes for features | [3, 6, 12] |
| `ENSEMBLE_STRATEGY` | Aggregation method for ensemble | weighted_average |

## Usage

### Running the Pipeline

```bash
python model.py
```

### Exploratory Data Analysis

Open and run the Jupyter notebook:
```bash
jupyter notebook eda.ipynb
```

The EDA includes:
- Dataset overview and statistics
- Sales trends and seasonality analysis
- Product lifecycle analysis
- Market and category analysis
- ABC classification of SKUs
- Temporal patterns visualization
- Missing data inspection

## Pipeline Architecture

### Data Processing

1. **Load & Validate**: Reads training data, product master, and submission template
2. **Feature Engineering**: Creates lag, rolling window, and temporal features
3. **Encoding**: Label encodes categorical variables (market, product, category)
4. **Scaling**: Standardizes features for gradient-based models

### Model Training

The pipeline uses time series cross-validation with expanding windows:

```
Fold 1: Train [2022-01-01 : 2024-08-01] → Validate [2024-09-01 : 2024-10-01]
Fold 2: Train [2022-01-01 : 2024-09-01] → Validate [2024-10-01 : 2024-11-01]
Fold 3: Train [2022-01-01 : 2024-10-01] → Validate [2024-11-01 : 2024-12-01]
```

### Model Components

- **LightGBM**: Fast, efficient gradient boosting
- **XGBoost**: Regularized gradient boosting with better generalization
- **CatBoost**: Optimized for categorical features
- **LSTM**: Deep learning for temporal sequence patterns

### Ensemble Strategy

Models are combined using weighted averaging based on out-of-fold (OOF) validation performance:

$$\text{Prediction} = \sum_{i=1}^{n} w_i \cdot \text{Model}_i(X)$$

where weights $w_i$ are proportional to each model's validation performance.

### Forecasting

Recursive multi-step ahead forecasting:
- Generates predictions for forecast period
- Uses previous predictions as features for subsequent steps
- Maintains temporal consistency

## Output

The pipeline generates:

1. **Submissions CSV**: `submissions/submission_ensemble_cv.csv`
   - Contains product-level predictions for forecast period
   - Format: [product_id, date, predicted_sales]

2. **Summary Reports**: CSV files in `summaries/` directory
   - Category summaries
   - Market analysis
   - Product lifecycle stages
   - SKU ABC classifications

3. **Visualizations**: Generated plots in `visualizations/` directory
   - Trend analysis
   - Seasonality patterns
   - Model performance comparisons

## Evaluation Metrics

The pipeline uses **Regularized WMAPE (Weighted Mean Absolute Percentage Error)**:

$$\text{RWMAPE} = \frac{\sum |y_{true} - y_{pred}| + \lambda |\sum y_{true} - \sum y_{pred}|}{{\sum |y_{true}| + \gamma \sum |y_{pred}| + \epsilon}}$$

where:
- $\lambda = 0.2$: Balancing term weight
- $\gamma = 0.8$: Prediction scaling factor
- $\epsilon = 10^{-9}$: Numerical stability

## Performance Optimization

- **Memory Management**: Garbage collection after model training
- **Logging**: Comprehensive logging for debugging and monitoring
- **Progress Tracking**: TQDM progress bars for long operations
- **Vectorization**: NumPy/Pandas operations for efficiency

## Best Practices

1. **Data Validation**: Always inspect data shape and missing values before processing
2. **Feature Scaling**: Applied before model training for consistent predictions
3. **Temporal Order**: Cross-validation respects time order (no data leakage)
4. **Hyperparameter Tuning**: Controlled through config file for reproducibility
5. **Ensemble Weights**: Computed from validation performance, not test data

## Troubleshooting

### Out of Memory
- Reduce `CV_FOLDS` or batch size
- Increase `CV_GAP_MONTHS` to use less data per fold

### Slow Execution
- Reduce number of LSTM epochs in config
- Disable less important models if forecasting time is critical
- Use a subset of data for initial testing

### Poor Predictions
- Check data quality and missing values
- Adjust lag and rolling window sizes
- Increase ensemble diversity with different random seeds
- Consider market-specific models instead of global model

## Contributing

When modifying the pipeline:
1. Update configuration in `PipelineConfig`
2. Add appropriate logging statements
3. Test with smaller CV folds initially
4. Document changes in docstrings
5. Verify results match expectations

## License

This project is part of the Haier Europe Datathon competition.

## Contact & Support

For issues or questions about this pipeline, please refer to the project documentation and comments in `model.py`.

---

**Last Updated**: December 2025
**Python Version**: 3.12+
**Status**: Active Development
