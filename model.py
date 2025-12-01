"""
Sales Forecasting Pipeline with Ensemble Models & Time Series Cross-Validation
============================================================================
Optimized pipeline for forecasting product and category-level sales 
using ensemble methods (LightGBM, XGBoost, CatBoost, LSTM).

Key Features:
- Time Series Cross-Validation (Expanding Window).
- Average Best-Iteration detection for final refit.
- Class-based architecture for state management.
- Vectorized recursive forecasting.
- Weighted ensemble strategy based on OOF (Out-of-Fold) performance.
"""

import gc
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import copy

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    # File Paths
    TRAIN_FILE: str = 'data/train.csv'
    PRODUCT_MASTER_FILE: str = 'data/product_master.csv'
    SUBMISSION_FILE: str = 'data/submission.csv'
    OUTPUT_FILE: str = "submissions/submission_ensemble_cv.csv"
    
    # Dates
    # Note: validation cutoff is now determined dynamically by CV folds
    CALENDAR_START: pd.Timestamp = pd.to_datetime("2022-01-01")
    CALENDAR_END: pd.Timestamp = pd.to_datetime("2024-10-01")
    FORECAST_START: pd.Timestamp = pd.to_datetime("2024-11-01")
    
    # CV Settings
    CV_FOLDS: int = 3           # Number of validation folds
    CV_VAL_MONTHS: int = 1      # Size of validation set in months (usually 1 for monthly forecast)
    CV_GAP_MONTHS: int = 0      # Gap between train and val (if needed)
    
    # Model Settings
    ENSEMBLE_STRATEGY: str = "weighted_average"
    USE_PRODUCT_ENSEMBLE: bool = True
    USE_CATEGORY_ENSEMBLE: bool = True
    
    # Feature Config
    LAGS: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [3, 6, 12])

    RUN_EDA: bool = True

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# METRICS & HELPERS
# ============================================================================

def rwmape(y_true: np.ndarray, y_pred: np.ndarray, gamma: float = 0.8, lam: float = 0.2, eps: float = 1e-9) -> float:
    """Regularized WMAPE metric."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    num = np.sum(np.abs(y_true - y_pred)) + lam * np.abs(np.sum(y_true) - np.sum(y_pred))
    den = np.sum(np.abs(y_true)) + gamma * np.sum(np.abs(y_pred)) + eps
    return float(num / den)

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

class SalesEDA:
    """Handles visualization and statistical analysis of the dataset."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_global_sales(self):
        """Plots total aggregated sales over time."""
        logger.info("Plotting global sales trend...")
        daily_sales = self.df.groupby('date')['quantity'].sum().reset_index()
        
        plt.figure()
        plt.plot(daily_sales['date'], daily_sales['quantity'], label='Total Sales', alpha=0.6)
        # Add trend line
        daily_sales['MA_3'] = daily_sales['quantity'].rolling(window=3).mean()
        plt.plot(daily_sales['date'], daily_sales['MA_3'], color='red', label='3-Month Moving Avg')
        
        plt.title('Global Sales Trend')
        plt.legend()
        plt.tight_layout()
        plt.savefig("visualizations/global_sales_trend.png")

    def plot_market_segments(self):
        """Plots sales distribution by Market and Category."""
        logger.info("Plotting market segments...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Market Share
        market_sales = self.df.groupby('market')['quantity'].sum().sort_values(ascending=False)
        sns.barplot(x=market_sales.index, y=market_sales.values, ax=axes[0], palette="viridis")
        axes[0].set_title("Total Quantity by Market")
        
        # Top 10 Categories
        cat_sales = self.df.groupby('category')['quantity'].sum().sort_values(ascending=False).head(10)
        sns.barplot(x=cat_sales.values, y=cat_sales.index, ax=axes[1], palette="magma")
        axes[1].set_title("Top 10 Categories by Volume")
        
        plt.tight_layout()
        plt.savefig("visualizations/market_segments.png")

    def plot_seasonality(self):
        """Plots boxplots of sales by month to check seasonality."""
        logger.info("Plotting seasonality...")
        df_temp = self.df.copy()
        df_temp['month'] = df_temp['date'].dt.month
        
        # Aggregate to avoid noise from item-level granularity
        monthly_agg = df_temp.groupby(['date', 'month'])['quantity'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='month', y='quantity', data=monthly_agg, palette="coolwarm")
        plt.title("Seasonality: Global Sales Distribution by Month")
        plt.savefig("visualizations/seasonality_boxplot.png")

    def plot_feature_correlations(self, feature_df: pd.DataFrame, target_col: str = 'quantity'):
        """Plots correlation heatmap for engineered features."""
        logger.info("Plotting feature correlations...")
        
        # Select numeric columns only
        cols = [c for c in feature_df.columns if 'lag' in c or 'roll' in c or c == target_col]
        
        if len(cols) < 2:
            logger.warning("Not enough features generated to plot correlation.")
            return

        corr = feature_df[cols].corr()
        
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu', center=0, square=True)
        plt.title(f"Feature Correlation Matrix (Target: {target_col})")
        plt.savefig("visualizations/feature_correlation_heatmap.png")

# ============================================================================
# CUSTOM MODELS
# ============================================================================

class LSTMRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible PyTorch LSTM wrapper."""
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.2, 
                 lr=0.01, epochs=15, batch_size=1024, random_state=42, device=None):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.model = None
        self.scaler = StandardScaler()
        self.input_dim = None
        self._is_fitted = False

    class _LSTMModule(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim, 
                hidden_size=hidden_dim, 
                num_layers=num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_dim, 1)
            self.activation = nn.Softplus() 

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :] 
            out = self.fc(out)
            return self.activation(out)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        # Handle DataFrame input
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        self.input_dim = X_scaled.shape[1]
        
        # Convert to Tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = self._LSTMModule(
            self.input_dim, self.hidden_dim, self.num_layers, self.dropout
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")
        
        if hasattr(X, "values"):
            X = X.values
            
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
            
        return preds

# ============================================================================
# ENSEMBLE MANAGER WITH CV
# ============================================================================

class EnsembleModelManager:
    """Manages training, weighting, and prediction of multiple models using CV."""
    
    def __init__(self, strategy: str = "weighted_average"):
        self.strategy = strategy
        self.models = {}
        self.weights = {}
        self.avg_best_iterations = {} # Store avg best iterations for final refit
        self.is_fitted = False
        self.feature_names = [] 
        self.model_type = "product"
        
    def _get_model_factory(self, model_type: str, seed: int = 42):
        """Returns a dictionary of FRESH model instances."""
        n_jobs = -1
        
        lstm_params = {
            'hidden_dim': 128 if model_type == 'product' else 64,
            'num_layers': 2,
            'epochs': 15, 
            'batch_size': 1024,
            'lr': 0.005,
            'random_state': seed
        }
        
        if model_type == "product":
            return {
                'lgb': lgb.LGBMRegressor(
                    objective='tweedie', tweedie_variance_power=1.5,
                    learning_rate=0.03, num_leaves=31, n_estimators=2000, # High n_est for early stopping
                    random_state=seed, n_jobs=n_jobs, verbosity=-1
                ),
                'xgb': xgb.XGBRegressor(
                    objective='reg:squarederror', learning_rate=0.05,
                    max_depth=8, n_estimators=2000, random_state=seed, n_jobs=n_jobs,
                    enable_categorical=True
                ),
                'catboost': CatBoostRegressor(
                    loss_function='Tweedie:variance_power=1.5', learning_rate=0.05,
                    depth=6, iterations=2000, random_seed=seed, verbose=False, thread_count=n_jobs,
                    allow_writing_files=False
                ),
                'rf': RandomForestRegressor(
                    n_estimators=100, max_depth=15, min_samples_split=5,
                    random_state=seed, n_jobs=n_jobs
                ),
                'lstm': LSTMRegressor(**lstm_params)
            }
        else:
            return {
                'lgb': lgb.LGBMRegressor(
                    objective='tweedie', tweedie_variance_power=1.5,
                    learning_rate=0.05, num_leaves=63, n_estimators=1500,
                    random_state=seed, n_jobs=n_jobs
                ),
                'xgb': xgb.XGBRegressor(
                    objective='reg:squarederror', learning_rate=0.1,
                    max_depth=6, n_estimators=1500, random_state=seed, n_jobs=n_jobs
                ),
                'lstm': LSTMRegressor(**lstm_params)
            }

    def fit_cv(self, 
               df: pd.DataFrame, 
               splits: List[Tuple[pd.Timestamp, pd.Timestamp]], 
               feature_cols: List[str],
               target_col: str,
               cat_features: List[str] = [],
               model_type: str = "product") -> pd.DataFrame:
        """
        Performs Time Series Cross Validation.
        
        Args:
            df: Full dataframe containing history.
            splits: List of tuples (train_end_date, val_end_date).
            feature_cols: List of feature names.
            target_col: Name of target column.
            cat_features: List of categorical feature names.
            model_type: "product" or "category".
            
        Returns:
            pd.DataFrame: Out-of-fold predictions.
        """
        self.model_type = model_type
        self.feature_names = feature_cols
        
        # Store scores per model per fold
        fold_scores = {} # {model_name: [score1, score2, ...]}
        best_iters = {}  # {model_name: [iter1, iter2, ...]}
        
        # Identify categorical indices
        cat_indices = None
        if cat_features:
            cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features]

        # Init result storage
        oof_preds = []

        logger.info(f"Starting {len(splits)}-fold CV for {model_type} models...")

        for fold_idx, (train_end_dt, val_end_dt) in enumerate(splits):
            logger.info(f"Fold {fold_idx+1}/{len(splits)}: Train <= {train_end_dt.date()}, Val <= {val_end_dt.date()}")
            
            # 1. Create Masks
            train_mask = df['date'] <= train_end_dt
            val_mask = (df['date'] > train_end_dt) & (df['date'] <= val_end_dt)
            
            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, target_col]
            X_val = df.loc[val_mask, feature_cols]
            y_val = df.loc[val_mask, target_col]
            
            # Store metadata for OOF return
            val_meta = df.loc[val_mask, ['date', 'market', 'unique_code']].copy()
            
            # 2. Initialize Fresh Models
            current_models = self._get_model_factory(model_type, seed=42 + fold_idx)
            
            fold_predictions = {}
            
            for name, model in current_models.items():
                # initialize lists if first fold
                if name not in fold_scores: 
                    fold_scores[name] = []
                    best_iters[name] = []
                    
                try:
                    # --- TRAINING ---
                    if name == 'lgb':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            eval_metric='rmse',
                            categorical_feature=cat_indices or 'auto',
                            callbacks=[lgb.early_stopping(50, verbose=False)]
                        )
                        if hasattr(model, 'best_iteration_'):
                            best_iters[name].append(model.best_iteration_)
                            
                    elif name == 'catboost':
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_val, y_val),
                            early_stopping_rounds=50,
                            cat_features=cat_indices,
                            verbose=False
                        )
                        if hasattr(model, 'get_best_iteration'):
                            best_iters[name].append(model.get_best_iteration())
                            
                    elif name == 'xgb':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                        if hasattr(model, 'best_iteration'):
                            best_iters[name].append(model.best_iteration)
                    
                    else:
                        # LSTM, RF (No early stopping)
                        model.fit(X_train, y_train)

                    # --- PREDICTION ---
                    if name == 'lgb' and hasattr(model, 'best_iteration_'):
                        pred = model.predict(X_val, num_iteration=model.best_iteration_)
                    elif name == 'catboost' and hasattr(model, 'get_best_iteration'):
                         # Catboost predict uses best iteration automatically if eval_set provided
                        pred = model.predict(X_val)
                    elif name == 'xgb' and hasattr(model, 'best_iteration'):
                        pred = model.predict(X_val, iteration_range=(0, model.best_iteration + 1))
                    else:
                        pred = model.predict(X_val)
                    
                    pred = np.maximum(pred, 0)
                    fold_predictions[name] = pred
                    
                    # --- SCORING ---
                    score = rwmape(y_val.values, pred)
                    fold_scores[name].append(score)
                    
                except Exception as e:
                    logger.error(f"Fold {fold_idx+1} model {name} failed: {e}")
                    fold_scores[name].append(float('inf'))
                    fold_predictions[name] = np.zeros(len(y_val))

            # Store OOF preds for this fold (using simple average for now, just for tracking)
            # Actual OOF logic would just return the DF, weight calc happens after loop
            val_meta['pred_avg'] = 0
            for n, p in fold_predictions.items():
                val_meta[f'pred_{n}'] = p
            oof_preds.append(val_meta)
            
            # Clean up to save memory
            del X_train, y_train, X_val, y_val, current_models
            gc.collect()

        # --- AGGREGATE RESULTS ---
        
        # 1. Average Iterations
        for name, iters in best_iters.items():
            if len(iters) > 0:
                self.avg_best_iterations[name] = int(np.mean(iters))
                logger.info(f"Model {name} Avg Best Iteration: {self.avg_best_iterations[name]}")
        
        # 2. Calculate Weights based on Average Score across folds
        avg_scores = {k: np.mean(v) for k, v in fold_scores.items()}
        logger.info(f"Average CV Scores: {avg_scores}")
        
        inv_scores = {k: 1.0/(v + 1e-9) for k, v in avg_scores.items()}
        total_inv = sum(inv_scores.values())
        if total_inv > 0:
            self.weights = {k: v/total_inv for k, v in inv_scores.items()}
        else:
            self.weights = {k: 1.0/len(avg_scores) for k in avg_scores}
            
        logger.info(f"Calculated Ensemble Weights: {self.weights}")
        self.is_fitted = True
        
        return pd.concat(oof_preds)

    def refit_full(self, X: pd.DataFrame, y: pd.Series, cat_features: List[str] = []):
        """Retrain models on full dataset using averaged best iterations."""
        logger.info("Refitting models on full history...")
        
        # Get fresh models
        self.models = self._get_model_factory(self.model_type, seed=999)
        
        cat_indices = [i for i, c in enumerate(X.columns) if c in cat_features] if cat_features else None
        
        for name, model in self.models.items():
            # Apply iteration limits if available
            params = model.get_params()
            avg_iter = self.avg_best_iterations.get(name)
            
            if avg_iter:
                # Add a small buffer (e.g. 10%) as we have more data now
                final_iter = int(avg_iter * 1.1)
                
                if name in ['lgb', 'xgb']:
                    params['n_estimators'] = final_iter
                elif name == 'catboost':
                    params['iterations'] = final_iter
            
            try:
                # Re-instantiate with new params
                if name == 'lgb':
                    new_model = lgb.LGBMRegressor(**params)
                    new_model.fit(X, y, categorical_feature=cat_indices or 'auto')
                elif name == 'catboost':
                    # Clean params for catboost constructor
                    clean_params = {k: v for k, v in params.items() 
                                  if k not in ['n_jobs', 'n_estimators', 'verbosity', 'random_state', 'cat_features']}
                    clean_params['random_seed'] = params.get('random_state', 42)
                    clean_params['iterations'] = params.get('iterations', 1000)
                    new_model = CatBoostRegressor(**clean_params)
                    new_model.fit(X, y, cat_features=cat_indices, verbose=False)
                elif name == 'xgb':
                    new_model = xgb.XGBRegressor(**params)
                    new_model.fit(X, y) # categorical support in newer xgb handles auto
                elif name == 'lstm':
                    new_model = LSTMRegressor(**params)
                    new_model.fit(X, y)
                else:
                    new_model = model.__class__(**params)
                    new_model.fit(X, y)
                
                self.models[name] = new_model
            except Exception as e:
                logger.error(f"Error refitting {name}: {e}")

    def _predict_single(self, name: str, X: pd.DataFrame) -> np.ndarray:
        model = self.models[name]
        # Handle models that require specific iteration handling if not handled inside wrapper
        return model.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("Models not fitted. Run fit_cv or refit_full first.")
            
        preds = np.zeros(len(X))
        
        if self.strategy == "weighted_average":
            total_weight = sum(self.weights.values())
            for name, weight in self.weights.items():
                if weight > 0.001:
                    p = self._predict_single(name, X)
                    # Normalize weight just in case
                    preds += p * (weight / total_weight)
        else:
            # Simple average
            for name in self.models:
                preds += self._predict_single(name, X)
            preds /= len(self.models)
            
        return np.maximum(preds, 0) # Clip negative predictions

# ============================================================================
# PIPELINE CLASS
# ============================================================================

class SalesForecastingPipeline:
    """Main pipeline with Time Series Cross-Validation."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.df: pd.DataFrame = pd.DataFrame()
        self.sub_df: pd.DataFrame = pd.DataFrame()
        self.prod_ensemble: EnsembleModelManager = EnsembleModelManager()
        self.cat_ensemble: EnsembleModelManager = EnsembleModelManager()
        self.encoders = {}
        self.val_preds: pd.DataFrame = pd.DataFrame() # Stores OOF predictions
        
    def load_data(self):
        logger.info("Loading datasets...")
        sales = pd.read_csv(self.cfg.TRAIN_FILE, parse_dates=['date'])
        product_master = pd.read_csv(self.cfg.PRODUCT_MASTER_FILE, 
                                        parse_dates=['start_production_date', 'end_production_date'])
        self.sub_df = pd.read_csv(self.cfg.SUBMISSION_FILE, parse_dates=['date'])
        
        self.df = pd.merge(sales, product_master, on='product_code', how='left')
        self.df['quantity'] = self.df['quantity'].clip(lower=0)

    def prepare_monthly_grids(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Creating monthly grids...")
        self.df['date'] = self.df['date'].dt.to_period('M').dt.to_timestamp()
        all_months = pd.date_range(self.cfg.CALENDAR_START, self.cfg.CALENDAR_END, freq='MS')
        
        # --- Product Grid ---
        prod_attrs = ['market', 'product_code', 'category', 'brand', 'factory', 
                      'start_production_date', 'end_production_date']
        unique_prods = self.df[prod_attrs].drop_duplicates()
        grid_prod = unique_prods.merge(pd.DataFrame({'date': all_months}), how='cross')
        monthly_sales = self.df.groupby(['market', 'product_code', 'date'])['quantity'].sum().reset_index()
        full_prod = pd.merge(grid_prod, monthly_sales, on=['market', 'product_code', 'date'], how='left')
        full_prod['quantity'] = full_prod['quantity'].fillna(0.0)
        
        # --- Category Grid ---
        unique_cats = self.df[['market', 'category']].drop_duplicates()
        grid_cat = unique_cats.merge(pd.DataFrame({'date': all_months}), how='cross')
        monthly_cat = self.df.groupby(['market', 'category', 'date'])['quantity'].sum().reset_index()
        full_cat = pd.merge(grid_cat, monthly_cat, on=['market', 'category', 'date'], how='left')
        full_cat['quantity'] = full_cat['quantity'].fillna(0.0)
        
        # Aggregates
        cat_counts = self.df.groupby(['market', 'category'])['product_code'].nunique().reset_index()
        cat_counts.columns = ['market', 'category', 'cat_product_count']
        full_cat = full_cat.merge(cat_counts, on=['market', 'category'], how='left').fillna(0)
        
        brand_counts = self.df.groupby(['market', 'category'])['brand'].nunique().reset_index()
        brand_counts.columns = ['market', 'category', 'cat_brand_diversity']
        full_cat = full_cat.merge(brand_counts, on=['market', 'category'], how='left').fillna(0)

        # Encoders
        for col in ['brand', 'factory']:
            full_prod[col] = full_prod[col].fillna('NA')
            le = LabelEncoder()
            full_prod[f'{col}_le'] = le.fit_transform(full_prod[col])
            self.encoders[col] = le
            
        # Add Unique Code for tracking
        full_prod['unique_code'] = full_prod['market'] + '-' + full_prod['product_code']
        full_cat['unique_code'] = full_cat['market'] + '-' + full_cat['category']
            
        return full_prod, full_cat

    def engineer_features(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        df = df.sort_values(group_cols + ['date']).reset_index(drop=True)
        
        # Lags & Rolling
        for lag in self.cfg.LAGS:
            df[f'lag_{lag}'] = df.groupby(group_cols)['quantity'].shift(lag).fillna(0)
            
        for w in self.cfg.ROLLING_WINDOWS:
            df[f'roll_mean_{w}'] = df.groupby(group_cols)['quantity'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            ).fillna(0)
            
        # Date Features
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Advanced ratios
        if 'lag_1' in df.columns and 'lag_3' in df.columns:
            df['lag_ratio_1_3'] = df['lag_1'] / (df['lag_3'] + 1e-9)
        if 'roll_mean_3' in df.columns and 'roll_mean_12' in df.columns:
            df['roll_ratio_3_12'] = df['roll_mean_3'] / (df['roll_mean_12'] + 1e-9)
            
        # Production Age
        if 'start_production_date' in df.columns:
            prod_start = pd.to_datetime(df['start_production_date'])
            diff = (df['date'].dt.year - prod_start.dt.year) * 12 + (df['date'].dt.month - prod_start.dt.month)
            df['start_prod_months'] = diff.fillna(0)
            
        # Cleanup
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        
        return df

    def generate_cv_splits(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Generates (train_end, val_end) tuples for expanding window CV.
        Working backwards from FORECAST_START.
        """
        splits = []
        
        # The last known data point is FORECAST_START - 1 month
        last_data_point = self.cfg.FORECAST_START - pd.DateOffset(months=1)
        
        for i in range(self.cfg.CV_FOLDS):
            # Calculate validation end date for this fold
            # Fold 0 (Latest): Ends at last_data_point
            # Fold 1: Ends at last_data_point - 1 month
            val_end = last_data_point - pd.DateOffset(months=i * self.cfg.CV_VAL_MONTHS)
            
            # Calculate train end date
            train_end = val_end - pd.DateOffset(months=self.cfg.CV_VAL_MONTHS)
            
            # Check sanity
            if train_end < self.cfg.CALENDAR_START:
                logger.warning(f"Fold {i} skipped: Train end {train_end} before calendar start.")
                continue
                
            splits.append((train_end, val_end))
            
        # Return reversed so we train from smallest window to largest (optional, but good for logging flow)
        return splits[::-1]

    def generate_recursive_forecast(self, 
                                  history_df: pd.DataFrame, 
                                  test_template: pd.DataFrame, 
                                  model_mgr: EnsembleModelManager,
                                  group_cols: List[str],
                                  feature_cols: List[str]) -> pd.DataFrame:
        logger.info(f"Recursive forecasting for {len(test_template)} rows...")
        
        # Filter history to relevant series
        test_keys = test_template[group_cols].drop_duplicates()
        work_df = pd.merge(history_df, test_keys, on=group_cols, how='inner')
        
        # Prepare Future Rows
        static_cols = [c for c in history_df.columns if c not in 
                      ['date', 'quantity'] + [f for f in history_df.columns if 'lag' in f or 'roll' in f]]
        
        future_dates = pd.date_range(self.cfg.FORECAST_START, periods=12, freq='MS')
        last_hist = work_df.sort_values('date').groupby(group_cols).last().reset_index()
        
        future_rows = []
        for date in future_dates:
            temp = last_hist[group_cols + [c for c in static_cols if c not in group_cols]].copy()
            temp['date'] = date
            temp['quantity'] = np.nan
            future_rows.append(temp)
            
        full_work = pd.concat([work_df, pd.concat(future_rows, ignore_index=True)], ignore_index=True).sort_values(group_cols + ['date'])
        
        # Loop
        for date in tqdm(future_dates, desc="Forecast Steps"):
            mask_date = full_work['date'] == date
            
            # Update Features (Localized Re-engineering)
            lookback_start = date - pd.DateOffset(months=max(self.cfg.LAGS) + 1)
            slice_mask = (full_work['date'] >= lookback_start) & (full_work['date'] <= date)
            slice_df = full_work.loc[slice_mask].copy()
            slice_df = self.engineer_features(slice_df, group_cols)
            
            X_pred = slice_df[slice_df['date'] == date][feature_cols].fillna(0)
            
            if not X_pred.empty:
                preds = model_mgr.predict(X_pred)
                full_work.loc[mask_date, 'quantity'] = np.round(preds)

        # Output
        forecast_df = full_work[full_work['date'] >= self.cfg.FORECAST_START][group_cols + ['date', 'quantity']]
        
        if 'product_code' in group_cols:
            forecast_df['unique_code'] = forecast_df['market'] + '-' + forecast_df['product_code']
        else:
            forecast_df['unique_code'] = forecast_df['market'] + '-' + forecast_df['category']
            
        return forecast_df
    
    def run_eda_suite(self, prod_grid: pd.DataFrame):
        """Executes the EDA steps."""
        eda = SalesEDA(self.df) # Use raw loaded df for general trends
        
        # 1. Global Trend
        eda.plot_global_sales()
        
        # 2. Market/Category Breakdown
        eda.plot_market_segments()
        
        # 3. Seasonality
        eda.plot_seasonality()
        
        # 4. Correlation (Uses the engineered grid)
        # We sample the grid to avoid memory issues with huge plots
        sample_grid = prod_grid.sample(n=min(50000, len(prod_grid)), random_state=42)
        eda.plot_feature_correlations(sample_grid)
    
    def run(self):
        # 1. Load
        self.load_data()
        prod_grid, cat_grid = self.prepare_monthly_grids()
        
        # 2. Features
        logger.info("Engineering features...")
        prod_grid = self.engineer_features(prod_grid, ['market', 'product_code'])
        cat_grid = self.engineer_features(cat_grid, ['market', 'category'])
        
        if self.cfg.RUN_EDA:
            logger.info("--- Starting EDA ---")
            self.run_eda_suite(prod_grid)
            logger.info("--- EDA Complete ---")

        # Select Features
        exclude = ['quantity', 'date', 'product_code', 'market', 'category', 'unique_code',
                   'start_production_date', 'end_production_date', 'brand', 'factory']
        prod_feats = [c for c in prod_grid.select_dtypes(include=np.number).columns if c not in exclude]
        cat_feats = [c for c in cat_grid.select_dtypes(include=np.number).columns if c not in exclude]
        
        # 3. Generate Splits
        cv_splits = self.generate_cv_splits()
        
        # 4. Product Ensemble CV
        if self.cfg.USE_PRODUCT_ENSEMBLE:
            logger.info("--- Product Ensemble CV ---")
            self.prod_ensemble = EnsembleModelManager(self.cfg.ENSEMBLE_STRATEGY)
            
            # Run CV
            prod_oof = self.prod_ensemble.fit_cv(
                prod_grid, cv_splits, prod_feats, 'quantity', 
                cat_features=['brand_le', 'factory_le'], model_type="product"
            )
            self.val_preds = pd.concat([self.val_preds, prod_oof])
            
            # Final Refit
            full_mask = prod_grid['date'] < self.cfg.FORECAST_START
            self.prod_ensemble.refit_full(
                prod_grid.loc[full_mask, prod_feats],
                prod_grid.loc[full_mask, 'quantity'],
                cat_features=['brand_le', 'factory_le']
            )

        # 5. Category Ensemble CV
        if self.cfg.USE_CATEGORY_ENSEMBLE:
            logger.info("--- Category Ensemble CV ---")
            self.cat_ensemble = EnsembleModelManager(self.cfg.ENSEMBLE_STRATEGY)
            
            cat_oof = self.cat_ensemble.fit_cv(
                cat_grid, cv_splits, cat_feats, 'quantity',
                model_type="category"
            )
            self.val_preds = pd.concat([self.val_preds, cat_oof])
            
            # Final Refit
            full_mask = cat_grid['date'] < self.cfg.FORECAST_START
            self.cat_ensemble.refit_full(
                cat_grid.loc[full_mask, cat_feats],
                cat_grid.loc[full_mask, 'quantity']
            )
            
        # 6. Forecast
        logger.info("Generating final forecasts...")
        all_forecasts = []
        
        sub = self.sub_df.copy()
        is_cat = sub['unique_code'].str.contains("CAT")
        prod_sub = sub[~is_cat].copy()
        cat_sub = sub[is_cat].copy()
        
        # Add metadata for template matching
        if not prod_sub.empty:
            prod_sub['market'] = prod_sub['unique_code'].str.split('-').str[0]
            prod_sub['product_code'] = prod_sub['unique_code'].str.split('-').str[1]
            
            hist = prod_grid[prod_grid['date'] < self.cfg.FORECAST_START]
            preds = self.generate_recursive_forecast(
                hist, prod_sub, self.prod_ensemble, ['market', 'product_code'], prod_feats
            )
            all_forecasts.append(preds)
            
        if not cat_sub.empty:
            cat_sub['market'] = cat_sub['unique_code'].str.split('-').str[0]
            cat_sub['category'] = cat_sub['unique_code'].str.split('-').str[1]
            
            hist = cat_grid[cat_grid['date'] < self.cfg.FORECAST_START]
            preds = self.generate_recursive_forecast(
                hist, cat_sub, self.cat_ensemble, ['market', 'category'], cat_feats
            )
            all_forecasts.append(preds)
            
        # 7. Save
        if all_forecasts:
            final = pd.concat(all_forecasts)
            submission = self.sub_df[['ID', 'unique_code', 'date']].merge(
                final[['unique_code', 'date', 'quantity']], on=['unique_code', 'date'], how='left'
            )
            submission['quantity'] = submission['quantity'].fillna(0)
            submission.to_csv(self.cfg.OUTPUT_FILE, index=False)
            logger.info(f"Saved to {self.cfg.OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        # 1. Initialize Configuration
        config = PipelineConfig()
        
        # 2. Instantiate Pipeline
        pipeline = SalesForecastingPipeline(config)
        
        # 3. Run
        pipeline.run()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise