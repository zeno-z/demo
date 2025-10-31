import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import pickle
import joblib
from datetime import datetime, timedelta
import json
import gc
import os
import os, sys, io, logging, traceback, streamlit as st

# 让前端显示详细错误
st.set_option("client.showErrorDetails", True)

# 收敛线程占用，避免 Cloud 资源紧绷导致中断
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 简单的异常上报工具
def report_exception(e: Exception, where: str = ""):
    tb = traceback.format_exc()
    st.error(f"❌ {where} 失败：{e}")
    st.code(tb, language="python")
    # 打到 cloud 日志
    logging.getLogger("stock_dashboard").exception("%s failed", where)
    # 提供下载
    st.download_button(
        "下载 last_error.txt",
        data=tb.encode("utf-8"),
        file_name="last_error.txt",
        mime="text/plain",
    )

# —— 强制限制数值库的并行线程，避免过度占用 CPU/内存 ——
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XGB_NUM_THREADS", "1")  # 供 xgboost 读取

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'predictions_cache' not in st.session_state:
    st.session_state.predictions_cache = {}
if 'manual_result' not in st.session_state:
    st.session_state.manual_result = None

# Constants
LOOKAHEAD = 30
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Set device and fix threading issues
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ===========================
# Model Definitions - EXACTLY matching Attachment 2
# ===========================
class LSTMModel(nn.Module):
    """LSTM Base model exactly matching attachment 2"""
    def __init__(self, input_size, hidden_size=24, num_layers=2, dropout=0.65, tail_k=8, bidirectional=True):
        super().__init__()
        self.tail_k = tail_k
        self.bi = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                          dropout=0 if num_layers==1 else dropout, bidirectional=bidirectional)
        self.norm = nn.LayerNorm(hidden_size*self.bi)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*self.bi, 1)
    
    def forward(self, x):
        o, _ = self.lstm(x)
        o = o[:, -self.tail_k:, :].mean(1)
        o = self.norm(o)
        o = self.dropout(o)
        return self.fc(o)

# Keep other model definitions from attachment 3 for advanced models
class SE(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.fc1 = nn.Linear(d, max(1, d//r))
        self.fc2 = nn.Linear(max(1, d//r), d)
    
    def forward(self, h):
        s = h.mean(dim=1)
        z = torch.relu(self.fc1(s))
        w = torch.sigmoid(self.fc2(z)).unsqueeze(1)
        return h * w

class LSTMAttPool(nn.Module):
    """Attentive Gated LSTM from attachment 3"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.40, tail_k=16, bidirectional=True, n_heads=4):
        super().__init__()
        self.tail_k = tail_k
        self.bi = 2 if bidirectional else 1
        self.d = hidden_size*self.bi
        self.h = n_heads
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0 if num_layers==1 else dropout, bidirectional=bidirectional)
        self.se = SE(self.d, r=4)
        self.q = nn.Parameter(torch.randn(self.h, self.d))
        self.key = nn.Linear(self.d, self.d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d)
        self.gate = nn.Linear(self.d*2, self.d)
        self.head = nn.Linear(self.d, 1)
    
    def forward(self, x):
        o,_ = self.lstm(x)
        tail = o[:, -self.tail_k:, :]
        tail = self.se(tail)
        k = torch.tanh(self.key(tail))
        qs = self.q.unsqueeze(0).unsqueeze(2).expand(k.size(0), self.h, 1, self.d)
        ks = k.unsqueeze(1)
        score = (qs*ks).sum(-1).squeeze(2)
        w = torch.softmax(score, dim=-1)
        ctx = (w.unsqueeze(-1) * tail.unsqueeze(1)).sum(2)
        ctx = ctx.mean(1)
        tail_mean = tail.mean(1)
        g = torch.sigmoid(self.gate(torch.cat([ctx, tail_mean], dim=1)))
        fused = g*ctx + (1.0-g)*tail_mean
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return self.head(fused)

# ===========================
# Helper Functions - EXACTLY matching Attachment 2
# ===========================
@st.cache_data
def load_and_clean_data(df):
    """Load and clean data from dataframe - matching attachment 2"""
    while len(df) > 0 and df.iloc[0].astype(str).str.contains('TM').any():
        df = df.iloc[1:]
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['open','high','low','close','adj_close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[(df['open']>0)&(df['high']>0)&(df['low']>0)&(df['close']>0)&(df['volume']>=0)]
    df = df[(df['high']>=df['low'])&(df['close']>=df['low'])&(df['close']<=df['high'])]
    df = df[(df['open']>=df['low'])&(df['open']<=df['high'])]
    df = df.dropna().sort_values('date').reset_index(drop=True)
    return df

def create_features_and_target(data, lookahead=1, use_extended=False):
    """Create features matching attachment 2 for base models, extended for others"""
    d = data.copy()
    d['log_close'] = np.log(d['close'])
    d['y_return'] = d['log_close'].shift(-lookahead) - d['log_close']
    d['ref_close'] = d['close']
    
    # Basic features (always included) - EXACTLY as in attachment 2
    d['body'] = (d['adj_close'] - d['open']) / d['open']
    d['upper'] = (d['high'] - np.maximum(d['open'], d['adj_close'])) / d['open']
    d['lower'] = (np.minimum(d['open'], d['adj_close']) - d['low']) / d['open']
    d['range'] = (d['high'] - d['low']) / d['open']
    d['or'] = d['open'].pct_change()
    d['vr'] = d['volume'].pct_change()
    
    # Lag features
    for lag in [1,2,3]:
        d[f'body_lag{lag}'] = d['body'].shift(lag)
        d[f'range_lag{lag}'] = d['range'].shift(lag)
        d[f'vr_lag{lag}'] = d['vr'].shift(lag)
    
    # Moving averages
    d['range_ma5'] = d['range'].rolling(5).mean()
    d['range_ma10'] = d['range'].rolling(10).mean()
    d['vr_ma5'] = d['vr'].rolling(5).mean()
    d['vr_ma10'] = d['vr'].rolling(10).mean()
    
    # Extended features only for advanced models
    if use_extended:
        # Add lag 5
        d['body_lag5'] = d['body'].shift(5)
        d['range_lag5'] = d['range'].shift(5)
        d['vr_lag5'] = d['vr'].shift(5)
        
        # Additional moving averages
        d['range_ma20'] = d['range'].rolling(20).mean()
        d['vr_ma20'] = d['vr'].rolling(20).mean()
        
        # Momentum features
        d['mom_5'] = d['close'].pct_change(5)
        d['mom_10'] = d['close'].pct_change(10)
        d['mom_20'] = d['close'].pct_change(20)
        
        # Volatility
        d['vol_10'] = d['close'].pct_change().rolling(10).std()
        d['vol_20'] = d['close'].pct_change().rolling(20).std()
        
        # RSI
        delta = d['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        d['rsi'] = d['rsi'] / 100.0
        
        d['vol_ratio'] = d['volume'] / (d['volume'].rolling(20).mean() + 1e-8)
        
        # Price position
        low_60 = d['low'].rolling(60).min()
        high_60 = d['high'].rolling(60).max()
        d['price_pos'] = (d['close'] - low_60) / (high_60 - low_60 + 1e-8)
        
        # Interaction features
        d['vol_mom'] = d['volume'].pct_change(5) * d['mom_5']
        d['range_vol'] = d['range'] * d['vol_ratio']
    
    d = d.replace([np.inf,-np.inf], np.nan).dropna().reset_index(drop=True)
    return d

def build_sequences(df, feature_cols, lookback):
    """Build sequences for LSTM"""
    X, y_ret, ref_close = [], [], []
    for i in range(lookback, len(df)):
        X.append(df[feature_cols].iloc[i-lookback:i].values)
        y_ret.append(df['y_return'].iloc[i])
        ref_close.append(df['ref_close'].iloc[i])
    return np.array(X), np.array(y_ret), np.array(ref_close)

def price_from_return(pred_ret, ref_close):
    """Convert log returns to prices"""
    return ref_close * np.exp(pred_ret)

def eval_price_metrics(y_true_price, y_pred_price):
    """Calculate performance metrics"""
    mse = mean_squared_error(y_true_price, y_pred_price)
    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true_price, y_pred_price)
    mape = float(np.mean(np.abs((y_true_price - y_pred_price)/(y_true_price + 1e-8)))*100)
    return {'MAE': float(mae), 'MSE': float(mse), 'RMSE': rmse, 'R2': float(r2), 'MAPE': mape}

def clip_by_train_quantiles(X_train_seq, X_seq, q_low, q_high):
    """Clip sequences by training quantiles"""
    Xc = X_seq.copy()
    for j in range(Xc.shape[-1]):
        lo, hi = q_low[j], q_high[j]
        Xc[:,:,j] = np.clip(Xc[:,:,j], lo, hi)
    return Xc

def scale_seq(X_seq, scaler):
    """Scale sequences"""
    out = np.zeros_like(X_seq)
    for i in range(len(X_seq)):
        out[i] = scaler.transform(X_seq[i])
    return out

def linear_calibration(y_true_ret, y_pred_ret):
    """Linear calibration for predictions"""
    x = np.vstack([y_pred_ret, np.ones_like(y_pred_ret)]).T
    a, b = np.linalg.lstsq(x, y_true_ret, rcond=None)[0]
    return float(a), float(b)

def mc_pred_loader(model, loader, T=16):
    """Monte Carlo dropout prediction - EXACTLY as in attachment 2"""
    preds = []
    with torch.no_grad():
        for _ in range(T):
            model.train()
            one_pass = []
            for xb, _ in loader:
                pr = model(xb.to(device)).squeeze().detach().cpu().numpy()
                if pr.ndim == 0:
                    one_pass.append(float(pr))
                else:
                    one_pass.extend(pr.tolist())
            preds.append(np.array(one_pass))
    model.eval()
    return np.mean(np.stack(preds, axis=0), axis=0)

def prepare_baseline_residual(df, lookahead, lookback, N):
    """Prepare baseline residuals for advanced models"""
    lc = df['log_close'].values
    ma_60 = pd.Series(lc).rolling(60, min_periods=60).mean()
    ma_120 = pd.Series(lc).rolling(120, min_periods=120).mean()
    ema_30 = pd.Series(lc).ewm(span=30, adjust=False).mean()
    
    trend = ma_60.diff(5)
    
    base_60 = (ma_60.shift(-lookahead) - pd.Series(lc))
    base_120 = (ma_120.shift(-lookahead) - pd.Series(lc))
    base_ema = (ema_30.shift(-lookahead) - pd.Series(lc))
    base_trend = trend * (lookahead / 5)
    
    base_series = (0.4 * base_60 + 0.3 * base_120 + 0.2 * base_ema + 0.1 * base_trend).values
    base_aligned = base_series[lookback:lookback+N]
    base_aligned = np.nan_to_num(base_aligned, nan=0.0)
    return base_aligned

# ===========================
# Training Functions - EXACTLY matching Attachment 2
# ===========================
def train_xgboost_matching_attachment2(X_train, y_train, X_val, y_val, X_test, lookback, n_features, feature_cols):
    """Train XGBoost exactly matching attachment 2"""
    # Flatten sequences
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    # Feature selection exactly as in attachment 2
    XGB_KEEP_FEATS = ['body','range_ma5','vr','or','body_lag1']
    keep_idx = np.array([feature_cols.index(k) for k in XGB_KEEP_FEATS], dtype=int)
    keep_cols = np.hstack([t*n_features + keep_idx for t in range(lookback)])
    
    X_train_use = X_train_flat[:, keep_cols]
    X_val_use = X_val_flat[:, keep_cols]
    X_test_use = X_test_flat[:, keep_cols]
    
    # Subsampling exactly as in attachment 2
    rng = np.random.RandomState(42)
    XGB_TRAIN_STRIDE = lookback
    ROW_KEEP_P = 0.55
    sel = np.arange(0, len(X_train_use), max(1, XGB_TRAIN_STRIDE))
    sel = sel[rng.rand(len(sel)) < ROW_KEEP_P]
    
    # Add noise to target exactly as in attachment 2
    sigma_ret = 0.25 * float(np.std(y_train)) if y_train.size > 0 else 0.0
    y_train_noisy = y_train + rng.normal(0, sigma_ret, size=y_train.shape)
    
    X_train_fit = X_train_use[sel].copy()
    # Feature dropout
    mask_xgb = rng.binomial(1, 0.60, size=X_train_fit.shape).astype(np.float32)
    X_train_fit *= mask_xgb
    y_train_fit = y_train_noisy[sel].copy()
    
    # XGBoost with exact parameters from attachment 2
    xgb_model = xgb.XGBRegressor(
        n_estimators=30, learning_rate=0.012,
        max_depth=1, min_child_weight=140,
        subsample=0.18, colsample_bytree=0.12,
        reg_alpha=0, reg_lambda=260, gamma=10,
        grow_policy="lossguide", max_leaves=2, max_bin=32,
        objective="reg:squarederror", random_state=42,
        tree_method="hist", n_jobs=0
    )
    
    try:
        xgb_model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val_use, y_val)],
            early_stopping_rounds=5, verbose=False
        )
    except TypeError:
        xgb_model.fit(X_train_fit, y_train_fit)
    
    # Predictions
    pred_train = xgb_model.predict(X_train_use)
    pred_val = xgb_model.predict(X_val_use)
    pred_test = xgb_model.predict(X_test_use)
    
    return xgb_model, pred_train, pred_val, pred_test

def train_lstm_one_seed(Xtr, ytr, Xva, yva, Xte, yte, lookahead, p_keep=0.55, epochs=100, patience=12, seed=17):
    """Train LSTM for one seed - EXACTLY matching attachment 2"""
    torch.cuda.empty_cache()
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Create datasets
    train_full_ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
    val_ds = TensorDataset(torch.FloatTensor(Xva), torch.FloatTensor(yva))
    test_ds = TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte))
    
    train_full_loader = DataLoader(train_full_ds, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    # Subsampling exactly as in attachment 2
    rng = np.random.RandomState(seed)
    stride = lookahead
    idx = np.arange(len(Xtr))[::stride]
    keep = rng.rand(len(idx)) < 0.8
    idx = idx[keep]
    Xtr_sub, ytr_sub = Xtr[idx], ytr[idx]
    
    train_ds = TensorDataset(torch.FloatTensor(Xtr_sub), torch.FloatTensor(ytr_sub))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, generator=g)
    
    # Model with exact parameters from attachment 2
    model = LSTMModel(input_size=Xtr.shape[-1], hidden_size=24, num_layers=2, 
                      dropout=0.65, tail_k=8, bidirectional=True).to(device)
    
    criterion = nn.SmoothL1Loss(beta=0.004)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.2e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    
    best_val = 1e9
    bad = 0
    best_state = None
    
    for _ in range(epochs):
        # Training
        model.train()
        for xb, yb in train_loader:
            xb_noise = xb + 0.012*torch.randn_like(xb)
            mask = (torch.rand(xb_noise.size(0), 1, xb_noise.size(2), generator=g) < p_keep).to(xb_noise.device).float()
            xb_noise = xb_noise * mask
            xb_noise, yb = xb_noise.to(device), yb.to(device)
            optimizer.zero_grad()
            yb = yb + 0.0015*torch.randn_like(yb)
            loss = criterion(model(xb_noise).squeeze(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss += criterion(model(xb).squeeze(), yb).item()
            vloss /= max(1, len(val_loader))
        
        scheduler.step(vloss)
        
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    
    # MC dropout predictions with T=16 as in attachment 2
    def pred_loader(loader):
        return mc_pred_loader(model, loader, T=16)
    
    pr_tr = pred_loader(train_full_loader)
    pr_va = pred_loader(val_loader)
    pr_te = pred_loader(test_loader)
    
    return pr_tr, pr_va, pr_te

def train_lstm_variant(X_train, y_train, X_val, y_val, X_test, y_test, model_type='base', 
                       lookahead=30, epochs=120, patience=14, seed=42, 
                       base_train=None, base_val=None, base_test=None):
    """Train LSTM variants for advanced models (unchanged)"""
    torch.cuda.empty_cache()
    gc.collect()
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Configuration based on model type
    if model_type == 'residual_anchor':
        cfg = dict(hidden=56, dropout=0.55, tail_k=10, p_keep=0.60, epochs=140, 
                   patience=16, mc_T=20, use_att=False, stride=3)
    elif model_type == 'stable':
        cfg = dict(hidden=72, dropout=0.45, tail_k=14, p_keep=0.70, epochs=160, 
                   patience=18, mc_T=24, use_att=False, stride=2)
    elif model_type == 'attentive_gated':
        cfg = dict(hidden=72, dropout=0.35, tail_k=18, p_keep=0.75, epochs=160, 
                   patience=18, mc_T=28, use_att=True, stride=2)
    elif model_type == 'mc_enhanced':
        cfg = dict(hidden=76, dropout=0.42, tail_k=16, p_keep=0.72, epochs=170, 
                   patience=19, mc_T=28, use_att=False, stride=2)
    else:
        return None  # Don't use this for base model
    
    # Subsampling
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X_train))
    idx = idx[::cfg.get('stride', 2)]
    keep = rng.rand(len(idx)) < (0.60 if model_type=='residual_anchor' else 0.80)
    idx = idx[keep]
    X_train_sub, y_train_sub = X_train[idx], y_train[idx]
    
    # Create datasets
    train_ds = TensorDataset(torch.FloatTensor(X_train_sub), torch.FloatTensor(y_train_sub))
    train_full_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, generator=g)
    train_full_loader = DataLoader(train_full_ds, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    input_size = X_train.shape[-1]
    
    # Model selection
    if cfg['use_att']:
        model = LSTMAttPool(input_size=input_size, hidden_size=cfg['hidden'], 
                           num_layers=2, dropout=cfg['dropout'], tail_k=max(12, cfg['tail_k']),
                           bidirectional=True).to(device)
    else:
        # Use the same LSTMModel class but with different parameters
        model = LSTMModel(input_size=input_size, hidden_size=cfg['hidden'], 
                         num_layers=2, dropout=cfg['dropout'], tail_k=cfg['tail_k'],
                         bidirectional=True).to(device)
    
    criterion = nn.SmoothL1Loss(beta=0.005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    dropout_schedule = {0: cfg['dropout'], 
                       int(cfg['epochs']*0.25): cfg['dropout']+0.05,
                       int(cfg['epochs']*0.5): cfg['dropout']+0.10}
    
    for epoch in range(cfg['epochs']):
        # Adjust dropout
        if epoch in dropout_schedule and hasattr(model, 'dropout'):
            model.dropout.p = dropout_schedule[epoch]
        
        # Training
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            # Mixup augmentation
            if np.random.rand() < 0.3:
                lam = np.random.beta(0.2, 0.2)
                idx_perm = torch.randperm(xb.size(0))
                xb = lam * xb + (1 - lam) * xb[idx_perm]
                yb = lam * yb + (1 - lam) * yb[idx_perm]
            
            # Add noise and dropout
            xb_noise = xb + 0.008*torch.randn_like(xb)
            mask = (torch.rand(xb_noise.size(0), 1, xb_noise.size(2), generator=g) < cfg['p_keep']).to(xb_noise.device).float()
            xb_noise = xb_noise * mask
            xb_noise, yb = xb_noise.to(device), yb.to(device)
            
            # Label smoothing
            yb_smooth = yb * 0.95 + yb.mean() * 0.05
            
            optimizer.zero_grad()
            loss = criterion(model(xb_noise).squeeze(), yb_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb).squeeze(), yb).item()
        
        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                break
        
        # Update progress
        progress = (epoch + 1) / cfg['epochs']
        progress_bar.progress(progress)
        status_text.text(f'Epoch {epoch+1}/{cfg["epochs"]} - Val Loss: {val_loss:.6f}')
    
    progress_bar.empty()
    status_text.empty()
    
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    
    model.eval()
    
    # MC dropout predictions
    pred_train = mc_pred_loader(model, train_full_loader, T=cfg['mc_T'])
    pred_val = mc_pred_loader(model, val_loader, T=cfg['mc_T'])
    pred_test = mc_pred_loader(model, test_loader, T=cfg['mc_T'])
    
    # If this is a residual model, add back the baseline
    if model_type in ['residual_anchor', 'stable', 'attentive_gated', 'mc_enhanced'] and base_train is not None:
        pred_train = pred_train + base_train
        pred_val = pred_val + base_val
        pred_test = pred_test + base_test
    
    return model, pred_train, pred_val, pred_test

def save_model(model, model_name, model_type='torch'):
    """Save model to disk"""
    filepath = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if model_type == 'torch':
        torch.save(model.state_dict(), filepath, _use_new_zipfile_serialization=False)
    else:
        joblib.dump(model, filepath)
    return filepath

def calculate_roi(initial_price, final_price, effective_days, base_days=365):
    """
    Calculate ROI (period return %) and annualized return (%) using compounding.
    - effective_days: 实际间隔天数 d（优先用日期差，没有日期就用用户输入 holding_days）
    - base_days: 年化基数 B（365 或 252）
    """
    if initial_price <= 0 or final_price <= 0:
        return {
            "roi_pct": np.nan, "annualized_pct": np.nan,
            "g_daily": np.nan, "r_period": np.nan
        }
    # 区间收益率 r
    r_period = (final_price / initial_price) - 1.0
    # 避免 d=0
    d = max(1, int(effective_days))
    # 日度复利因子 g
    g_daily = (1.0 + r_period) ** (1.0 / d)
    # 年化
    annualized = (1.0 + r_period) ** (float(base_days) / float(d)) - 1.0
    return {
        "roi_pct": float(r_period * 100.0),
        "annualized_pct": float(annualized * 100.0),
        "g_daily": float(g_daily),
        "r_period": float(r_period),
        "effective_days": int(d),
        "base_days": int(base_days),
    }

# ===========================
# Main Dashboard
# ===========================
def main():
    st.title("📈 Stock Prediction Dashboard")
    #st.markdown("Advanced ML models for stock price prediction with XGBoost and LSTM variants")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")

        # —— 用 session_state 持久化上传文件 ——
        uf = st.file_uploader("Upload CSV file", type=['csv'], key="upload_csv")
        if uf is not None:
            st.session_state["uploaded_file"] = uf

        uploaded_file = st.session_state.get("uploaded_file", None)
        
        if uploaded_file is not None:
            # Load data
            df_raw = pd.read_csv(uploaded_file)
            raw_rows = len(df_raw)          # 原始数据行数
            df = load_and_clean_data(df_raw)
            st.success(f"Data loaded: {raw_rows} rows")  # 显示原始行数
            
            # Date range selector
            st.subheader("📅 Date Range")
            min_date = df['date'].min()
            max_date = df['date'].max()
            
            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Model selection
            st.subheader("🤖 Model Selection")
            models = st.multiselect(
                "Select models to train/use",
                ["XGBoost", "LSTM Base", "Residual-Anchor-LSTM", "Stable-LSTM", 
                 "Attentive-Gated-LSTM", "MC-Enhanced-LSTM"],
                default=["XGBoost"]
            )
            
            # Training parameters
            st.subheader("⚙️ Parameters")
            lookback = st.slider("Lookback period", 30, 120, 60)
            lookahead = st.slider("Lookahead period", 10, 60, 30)
            
            # Train button
            if st.button("🚀 Train Models", type="primary"):
                train_models(df, df_raw, models, lookback, lookahead)
            # —— 新增：基于最近一次预测结果的可交互投资分析 ——
            if st.session_state.manual_result is not None:
                

                mr = st.session_state.manual_result
                col1, col2 = st.columns(2)

                # st.markdown("---")
                # st.subheader("📊 Investment Analysis")
                # with col1:
                #     investment = st.number_input(
                #         "If you invest ($):",
                #         value=10000,
                #         min_value=100,
                #         step=100,
                #         key="invest_amount_manual"  # 避免与别处冲突
                #     )
                #     expected_value = investment * (mr["predicted_price"] / mr["current_price"])
                #     expected_profit = expected_value - investment
                #     st.info(
                #         f"**Expected Value:** ${expected_value:,.2f}  \n"
                #         f"**Expected Profit/Loss:** ${expected_profit:+,.2f}  \n"
                #         f"**ROI:** {(expected_profit/investment)*100:+.2f}%"
                #     )

                # with col2:
                #     change_pct = mr["change_pct"]
                #     volatility_estimate = abs(change_pct) * 0.5
                #     if abs(change_pct) < 2:
                #         risk_level, risk_color = "Low", "green"
                #     elif abs(change_pct) < 5:
                #         risk_level, risk_color = "Moderate", "orange"
                #     else:
                #         risk_level, risk_color = "High", "red"

                #     st.info(
                #         f"**Risk Level:** :{risk_color}[{risk_level}]  \n"
                #         f"**Estimated Volatility:** ±{volatility_estimate:.2f}%  \n"
                #         f"**Model Used:** {mr['model_name']}"
                #     )
    
    # Main content
    if uploaded_file is not None:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "📊 Metrics", "🎯 Manual Prediction", 
                                          "💰 ROI Calculator"])
        
        with tab1:
            show_predictions_tab(df, date_range)
        
        with tab2:
            show_metrics_tab()
        
        with tab3:
            show_manual_prediction_tab(df)
        
        with tab4:
            show_roi_calculator_tab(df)
    else:
        st.info("👆 Please upload a CSV file to get started")

def train_models(df, df_raw, selected_models, lookback, lookahead):
    """Train selected models"""
    with st.spinner("Preparing data..."):
        # Determine if we need extended features
        use_extended = any(m in selected_models for m in 
                          ["Residual-Anchor-LSTM", "Stable-LSTM", 
                           "Attentive-Gated-LSTM", "MC-Enhanced-LSTM"])
        
        # Create features - base for XGBoost and LSTM Base
        df_features_base = create_features_and_target(df, lookahead=lookahead, use_extended=False)
        df_features_extended = create_features_and_target(df, lookahead=lookahead, use_extended=True) if use_extended else None
        
        # Feature columns for base models (matching attachment 2)
        feature_cols_base = [
            'body','upper','lower','range','or','vr',
            'body_lag1','body_lag2','body_lag3',
            'range_lag1','range_lag2','range_lag3',
            'vr_lag1','vr_lag2','vr_lag3',
            'range_ma5','range_ma10','vr_ma5','vr_ma10'
        ]
        
        # Extended feature columns for advanced models
        feature_cols_extended = [
            'body','upper','lower','range','or','vr',
            'body_lag1','body_lag2','body_lag3','body_lag5',
            'range_lag1','range_lag2','range_lag3','range_lag5',
            'vr_lag1','vr_lag2','vr_lag3','vr_lag5',
            'range_ma5','range_ma10','range_ma20',
            'vr_ma5','vr_ma10','vr_ma20',
            'mom_5','mom_10','mom_20',
            'vol_10','vol_20',
            'rsi','vol_ratio','price_pos',
            'vol_mom','range_vol'
        ] if use_extended else feature_cols_base
        
        # Build sequences for base models
        X_seq_base, y_ret, ref_close = build_sequences(df_features_base, feature_cols_base, lookback)
        
        # Build sequences for extended models if needed
        if use_extended:
            X_seq_extended, y_ret_ext, ref_close_ext = build_sequences(df_features_extended, feature_cols_extended, lookback)
        
        # Split data with gaps (matching attachment 2)
        N = len(X_seq_base)
        train_ratio, val_ratio = 0.70, 0.15
        raw_train_size = int(N * train_ratio)
        raw_val_size = int(N * val_ratio)
        gap = max(lookback, lookahead)
        
        train_end = max(0, raw_train_size - gap)
        val_start = raw_train_size + gap
        val_end = min(N, val_start + raw_val_size)
        test_start = min(N, val_end + gap)
        
        # Split for base models
        X_train_raw_base = X_seq_base[:train_end]
        X_val_raw_base = X_seq_base[val_start:val_end]
        X_test_raw_base = X_seq_base[test_start:]
        
        # Split for extended models if needed
        if use_extended:
            X_train_raw_ext = X_seq_extended[:train_end]
            X_val_raw_ext = X_seq_extended[val_start:val_end]
            X_test_raw_ext = X_seq_extended[test_start:]
        
        y_train = y_ret[:train_end]
        y_val = y_ret[val_start:val_end]
        y_test = y_ret[test_start:]
        
        ref_train = ref_close[:train_end]
        ref_val = ref_close[val_start:val_end]
        ref_test = ref_close[test_start:]
        
        # Clip by quantiles for base models
        n_features_base = X_train_raw_base.shape[-1]
        train_flat_for_q_base = X_train_raw_base.reshape(-1, n_features_base)
        ql_base = np.quantile(train_flat_for_q_base, 0.01, axis=0)
        qh_base = np.quantile(train_flat_for_q_base, 0.99, axis=0)
        
        X_train_c_base = clip_by_train_quantiles(X_train_raw_base, X_train_raw_base, ql_base, qh_base)
        X_val_c_base = clip_by_train_quantiles(X_train_raw_base, X_val_raw_base, ql_base, qh_base)
        X_test_c_base = clip_by_train_quantiles(X_train_raw_base, X_test_raw_base, ql_base, qh_base)
        
        # Standardize data for base models
        scaler_base = StandardScaler()
        scaler_base.fit(X_train_c_base.reshape(-1, n_features_base))
        X_train_s_base = scale_seq(X_train_c_base, scaler_base)
        X_val_s_base = scale_seq(X_val_c_base, scaler_base)
        X_test_s_base = scale_seq(X_test_c_base, scaler_base)
        
        # Process extended features if needed
        if use_extended:
            n_features_ext = X_train_raw_ext.shape[-1]
            train_flat_for_q_ext = X_train_raw_ext.reshape(-1, n_features_ext)
            ql_ext = np.quantile(train_flat_for_q_ext, 0.01, axis=0)
            qh_ext = np.quantile(train_flat_for_q_ext, 0.99, axis=0)
            
            X_train_c_ext = clip_by_train_quantiles(X_train_raw_ext, X_train_raw_ext, ql_ext, qh_ext)
            X_val_c_ext = clip_by_train_quantiles(X_train_raw_ext, X_val_raw_ext, ql_ext, qh_ext)
            X_test_c_ext = clip_by_train_quantiles(X_train_raw_ext, X_test_raw_ext, ql_ext, qh_ext)
            
            scaler_ext = StandardScaler()
            scaler_ext.fit(X_train_c_ext.reshape(-1, n_features_ext))
            X_train_s_ext = scale_seq(X_train_c_ext, scaler_ext)
            X_val_s_ext = scale_seq(X_val_c_ext, scaler_ext)
            X_test_s_ext = scale_seq(X_test_c_ext, scaler_ext)
            
            # Prepare baseline for residual models
            base_all = prepare_baseline_residual(df_features_extended, lookahead, lookback, len(X_seq_extended))
            base_train = base_all[:train_end]
            base_val = base_all[val_start:val_end]
            base_test = base_all[test_start:]
        else:
            base_train = base_val = base_test = None
        
        # Save scaler and data
        joblib.dump(scaler_base, os.path.join(MODEL_DIR, 'scaler_base.pkl'))
        joblib.dump({'n_features': n_features_base, 'feature_cols': feature_cols_base}, 
                   os.path.join(MODEL_DIR, 'feature_config_base.pkl'))
        
        if use_extended:
            joblib.dump(scaler_ext, os.path.join(MODEL_DIR, 'scaler_extended.pkl'))
            joblib.dump({'n_features': n_features_ext, 'feature_cols': feature_cols_extended}, 
                       os.path.join(MODEL_DIR, 'feature_config_extended.pkl'))
        
        # Store in session state
        st.session_state.data_cache = {
            'df': df,
            'df_features_base': df_features_base,
            'X_train_base': X_train_s_base,
            'X_val_base': X_val_s_base,
            'X_test_base': X_test_s_base,
            'X_train_ext': X_train_s_ext if use_extended else None,
            'X_val_ext': X_val_s_ext if use_extended else None,
            'X_test_ext': X_test_s_ext if use_extended else None,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'ref_train': ref_train,
            'ref_val': ref_val,
            'ref_test': ref_test,
            'base_train': base_train,
            'base_val': base_val,
            'base_test': base_test,
            'dates_test': df_features_base['date'].iloc[test_start+lookback:].values if test_start < len(df_features_base)-lookback else [],
            'lookback': lookback,
            'lookahead': lookahead,
            'n_features_base': n_features_base,
            'feature_cols_base': feature_cols_base,
            'n_features_ext': n_features_ext if use_extended else None,
            'feature_cols_ext': feature_cols_extended if use_extended else None,
            'scaler_base': scaler_base,
            'ql_base': ql_base,
            'qh_base': qh_base
        }
    
    # Train each selected model
    for model_name in selected_models:
        st.subheader(f"Training {model_name}...")
        
        if model_name == "XGBoost":
            with st.spinner(f"Training {model_name}..."):
                # Use base features for XGBoost
                model, pred_train, pred_val, pred_test = train_xgboost_matching_attachment2(
                    X_train_s_base, y_train, X_val_s_base, y_val, X_test_s_base,
                    lookback, n_features_base, feature_cols_base
                )
                
                save_model(model, model_name, model_type='sklearn')
                st.session_state.model_cache[model_name] = model
                
        elif model_name == "LSTM Base":
            with st.spinner(f"Training {model_name}..."):
                # Normalize targets for LSTM as in attachment 2
                y_mu = float(np.mean(y_train))
                y_sd = float(np.std(y_train) + 1e-8)
                y_train_scaled = (y_train - y_mu) / y_sd
                y_val_scaled = (y_val - y_mu) / y_sd
                y_test_scaled = (y_test - y_mu) / y_sd
                
                # Multi-seed ensemble as in attachment 2
                seeds = [13, 29, 47]
                pred_train_list, pred_val_list, pred_test_list = [], [], []
                
                for seed in seeds:
                    pr_tr, pr_va, pr_te = train_lstm_one_seed(
                        X_train_s_base, y_train_scaled, X_val_s_base, y_val_scaled, 
                        X_test_s_base, y_test_scaled, lookahead=lookahead, 
                        p_keep=0.55, epochs=100, patience=12, seed=seed
                    )
                    pred_train_list.append(pr_tr)
                    pred_val_list.append(pr_va)
                    pred_test_list.append(pr_te)
                
                # Ensemble predictions
                pred_train_scaled = np.mean(np.stack(pred_train_list, 0), 0)
                pred_val_scaled = np.mean(np.stack(pred_val_list, 0), 0)
                pred_test_scaled = np.mean(np.stack(pred_test_list, 0), 0)
                
                # Denormalize
                pred_train = pred_train_scaled * y_sd + y_mu
                pred_val = pred_val_scaled * y_sd + y_mu
                pred_test = pred_test_scaled * y_sd + y_mu
                
                # Linear calibration
                a_cal, b_cal = linear_calibration(y_val, pred_val)
                pred_val = a_cal * pred_val + b_cal
                pred_test = a_cal * pred_test + b_cal
                
                # Save the last model (for demonstration)
                # In production, you might want to save all models or the ensemble
                st.session_state.model_cache[model_name] = {
                    'y_mu': y_mu, 
                    'y_sd': y_sd,
                    'a_cal': a_cal,
                    'b_cal': b_cal
                }
                
        else:
            # Advanced LSTM models - use extended features
            with st.spinner(f"Training {model_name}..."):
                # Map model names to types
                model_type_map = {
                    "Residual-Anchor-LSTM": "residual_anchor",
                    "Stable-LSTM": "stable",
                    "Attentive-Gated-LSTM": "attentive_gated",
                    "MC-Enhanced-LSTM": "mc_enhanced"
                }
                model_type = model_type_map.get(model_name, "base")
                
                # Use extended features for advanced models
                X_train_use = X_train_s_ext
                X_val_use = X_val_s_ext
                X_test_use = X_test_s_ext
                
                # For residual models, use residual targets
                y_train_target = y_train - base_train
                y_val_target = y_val - base_val
                y_test_target = y_test - base_test
                
                # Normalize targets
                y_mu = float(np.mean(y_train_target))
                y_sd = float(np.std(y_train_target) + 1e-8)
                y_train_scaled = (y_train_target - y_mu) / y_sd
                y_val_scaled = (y_val_target - y_mu) / y_sd
                y_test_scaled = (y_test_target - y_mu) / y_sd
                
                # Multi-seed ensemble
                seeds = [13, 29, 47]
                pred_train_list, pred_val_list, pred_test_list = [], [], []
                
                for seed in seeds:
                    model, pred_tr, pred_va, pred_te = train_lstm_variant(
                        X_train_use, y_train_scaled, X_val_use, y_val_scaled, 
                        X_test_use, y_test_scaled, model_type=model_type, 
                        lookahead=lookahead, seed=seed,
                        base_train=base_train, base_val=base_val, base_test=base_test
                    )
                    
                    pred_train_list.append(pred_tr)
                    pred_val_list.append(pred_va)
                    pred_test_list.append(pred_te)
                
                # Ensemble predictions
                pred_train_scaled = np.mean(np.stack(pred_train_list, 0), 0)
                pred_val_scaled = np.mean(np.stack(pred_val_list, 0), 0)
                pred_test_scaled = np.mean(np.stack(pred_test_list, 0), 0)
                
                # Denormalize
                pred_train = pred_train_scaled * y_sd + y_mu
                pred_val = pred_val_scaled * y_sd + y_mu
                pred_test = pred_test_scaled * y_sd + y_mu
                
                # Linear calibration
                a_cal, b_cal = linear_calibration(y_val, pred_val)
                pred_val = a_cal * pred_val + b_cal
                pred_test = a_cal * pred_test + b_cal
                
                save_model(model, model_name, model_type='torch')
                st.session_state.model_cache[model_name] = model
        
        # Convert to prices and calculate metrics
        price_train_true = price_from_return(y_train, ref_train)
        price_val_true = price_from_return(y_val, ref_val)
        price_test_true = price_from_return(y_test, ref_test)
        
        price_train_pred = price_from_return(pred_train, ref_train)
        price_val_pred = price_from_return(pred_val, ref_val)
        price_test_pred = price_from_return(pred_test, ref_test)
        
        # Calculate metrics
        metrics_train = eval_price_metrics(price_train_true, price_train_pred)
        metrics_val = eval_price_metrics(price_val_true, price_val_pred)
        metrics_test = eval_price_metrics(price_test_true, price_test_pred)
        
        # Store predictions
        st.session_state.predictions_cache[model_name] = {
            'train': {'true': price_train_true, 'pred': price_train_pred},
            'val': {'true': price_val_true, 'pred': price_val_pred},
            'test': {'true': price_test_true, 'pred': price_test_pred},
            'metrics': metrics_test
        }
        
        # Display metrics
        # col1, col2, col3, col4, col5 = st.columns(5)
        # col1.metric("R²", f"{metrics_test['R2']:.4f}")
        # col2.metric("RMSE", f"{metrics_test['RMSE']:.2f}")
        # col3.metric("MAE", f"{metrics_test['MAE']:.2f}")
        # col4.metric("MSE", f"{metrics_test['MSE']:.2f}")
        # col5.metric("MAPE (%)", f"{metrics_test['MAPE']:.2f}")
    
    st.success("✅ Models trained successfully!")
    st.session_state.models_trained = True

def show_predictions_tab(df, date_range):
    """Show prediction visualizations"""
    if not st.session_state.models_trained:
        st.warning("Please train models first")
        return
    
    st.header("📈 Prediction Results")
    
    # Model selector
    model_name = st.selectbox("Select model", list(st.session_state.predictions_cache.keys()))
    
    if model_name in st.session_state.predictions_cache:
        preds = st.session_state.predictions_cache[model_name]
        dates = st.session_state.data_cache.get('dates_test', [])
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            x=dates if len(dates) > 0 else list(range(len(preds['test']['true']))),
            y=preds['test']['true'],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            x=dates if len(dates) > 0 else list(range(len(preds['test']['pred']))),
            y=preds['test']['pred'],
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{model_name} - Test Set Predictions',
            xaxis_title='Date' if len(dates) > 0 else 'Index',
            yaxis_title='Price',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        st.subheader("📊 Residuals vs Predicted (Test)")
        
        residuals_test = preds['test']['true'] - preds['test']['pred']
        
        fig_residuals = go.Figure()
        
        fig_residuals.add_trace(
            go.Scatter(
                x=preds['test']['pred'], 
                y=residuals_test,
                mode='markers', 
                marker=dict(size=5, color='blue', opacity=0.5),
                name='Residuals'
            )
        )
        
        fig_residuals.add_trace(
            go.Scatter(
                x=[min(preds['test']['pred']), max(preds['test']['pred'])],
                y=[0, 0], 
                mode='lines', 
                line=dict(color='red', dash='dash'),
                name='Zero Line',
                showlegend=False
            )
        )
        
        fig_residuals.update_layout(
            title='Residuals vs Predicted Values',
            xaxis_title='Predicted Price',
            yaxis_title='Residuals',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_residuals, use_container_width=True)

def show_metrics_tab():
    """Show metrics comparison"""
    if not st.session_state.models_trained:
        st.warning("Please train models first")
        return
    
    st.header("📊 Model Performance Metrics")
    
    # Collect metrics
    metrics_data = []
    for model_name, preds in st.session_state.predictions_cache.items():
        metrics = preds['metrics']
        metrics_data.append({
            'Model': model_name,
            'R²': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'MAPE (%)': metrics['MAPE']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display table
    st.dataframe(metrics_df.style.format({
        'R²': '{:.4f}',
        'RMSE': '{:.2f}',
        'MAE': '{:.2f}',
        'MSE': '{:.2f}',
        'MAPE (%)': '{:.2f}'
    }))
    
    # Create comparison charts
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('R² (Higher is better)', 'RMSE (Lower is better)', 'MAE (Lower is better)',
                       'MSE (Lower is better)', 'MAPE % (Lower is better)')
    )
    
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['R²'], name='R²'), row=1, col=1)
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'], name='RMSE'), row=1, col=2)
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'], name='MAE'), row=1, col=3)
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MSE'], name='MSE'), row=2, col=1)
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MAPE (%)'], name='MAPE'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def prepare_data_for_manual_prediction(df, lookback, lookahead, use_extended=False):
    """
    Prepare sufficient historical data for manual prediction
    Ensures we have enough data after feature engineering
    """
    # We need extra rows for feature calculation (moving averages, lags, etc.)
    # The maximum we need is for 60-day and 120-day moving averages in extended features
    if use_extended:
        required_history = max(lookback + 120 + lookahead, 200)
    else:
        # For base features, we need at least lookback + 20 (for ma10, ma20) + some buffer
        required_history = max(lookback + 30, 100)
    
    # Get the last 'required_history' rows from the dataframe
    historical_df = df.tail(required_history).copy()
    
    return historical_df

def show_manual_prediction_tab(df):
    """Manual prediction with user inputs - FIXED VERSION"""
    st.header("🎯 Manual Prediction")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first")
        return
    
    # Check if we have the necessary data
    if 'data_cache' not in st.session_state:
        st.error("No training data available. Please train models first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Enter Latest Price Data")
        st.info("Enter the most recent day's trading data for prediction")
        
        # Single day input
        open_price = st.number_input("Open Price ($)", value=100.0, min_value=0.01, format="%.2f")
        high_price = st.number_input("High Price ($)", value=105.0, min_value=0.01, format="%.2f")
        low_price = st.number_input("Low Price ($)", value=95.0, min_value=0.01, format="%.2f")
        close_price = st.number_input("Close Price ($)", value=102.0, min_value=0.01, format="%.2f")
        adj_close = st.number_input("Adjusted Close ($)", value=102.0, min_value=0.01, format="%.2f")
        volume = st.number_input("Volume", value=1000000, min_value=0, format="%d")
        
        # Validate inputs
        if high_price < low_price:
            st.error("High price must be >= Low price")
        if close_price < low_price or close_price > high_price:
            st.error("Close price must be between Low and High")
        if open_price < low_price or open_price > high_price:
            st.error("Open price must be between Low and High")
    
    with col2:
        st.subheader("🤖 Model Selection")
        
        # Model selector
        available_models = list(st.session_state.predictions_cache.keys())
        if not available_models:
            st.error("No trained models available")
            return
            
        model_name = st.selectbox(
            "Select Model for Prediction",
            available_models,
            key="manual_model_select"
        )
        
        # Display model info
        st.info(f"**Prediction Horizon:** {st.session_state.data_cache['lookahead']} days ahead")
        
        if st.button("🔮 Generate Prediction", type="primary", key="predict_button"):
            with st.spinner("Generating prediction..."):
                try:
                    # Get required data from cache
                    lookback = st.session_state.data_cache['lookback']
                    lookahead = st.session_state.data_cache['lookahead']
                    
                    # Determine if we need extended features
                    use_extended = model_name in ["Residual-Anchor-LSTM", "Stable-LSTM", 
                                                  "Attentive-Gated-LSTM", "MC-Enhanced-LSTM"]
                    
                    # Prepare historical data with sufficient rows for feature engineering
                    historical_df = prepare_data_for_manual_prediction(df, lookback, lookahead, use_extended)
                    
                    # Create a new row with the manual input
                    new_row = pd.DataFrame([{
                        'date': historical_df['date'].iloc[-1] + pd.Timedelta(days=1),  # Next day after last historical date
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'adj_close': adj_close,
                        'volume': volume
                    }])
                    
                    # Combine historical with new manual input
                    combined_df = pd.concat([historical_df, new_row], ignore_index=True)
                    
                    # Create features
                    pred_df = create_features_and_target(combined_df, lookahead=lookahead, use_extended=use_extended)
                    
                    # Check if we have enough data after feature engineering
                    if len(pred_df) < lookback:
                        st.error(f"Not enough data after feature engineering. Need at least {lookback} days of data with valid features. Current data has {len(pred_df)} valid rows.")
                        st.info("This usually happens when there's insufficient historical data. Please ensure your uploaded data has sufficient history.")
                        return
                    
                    # Select appropriate feature columns and configuration
                    if use_extended and 'feature_cols_ext' in st.session_state.data_cache:
                        feature_cols = st.session_state.data_cache['feature_cols_ext']
                        n_features = st.session_state.data_cache['n_features_ext']
                        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_extended.pkl')) if os.path.exists(os.path.join(MODEL_DIR, 'scaler_extended.pkl')) else st.session_state.data_cache['scaler_base']
                    else:
                        feature_cols = st.session_state.data_cache['feature_cols_base']
                        n_features = st.session_state.data_cache['n_features_base']
                        scaler = st.session_state.data_cache['scaler_base']
                    
                    # Get quantiles for clipping
                    ql = st.session_state.data_cache.get('ql_base')
                    qh = st.session_state.data_cache.get('qh_base')
                    
                    # Build sequence from the last lookback rows
                    # Filter to only include features that exist in pred_df
                    available_cols = [col for col in feature_cols if col in pred_df.columns]
                    if len(available_cols) < len(feature_cols):
                        st.warning(f"Some features are missing. Using {len(available_cols)}/{len(feature_cols)} features.")
                        
                        # If too many features are missing, fall back to basic features
                        if len(available_cols) < len(feature_cols) * 0.7:  # If more than 30% features are missing
                            st.info("Falling back to basic features due to missing extended features.")
                            feature_cols = st.session_state.data_cache['feature_cols_base']
                            n_features = st.session_state.data_cache['n_features_base']
                            scaler = st.session_state.data_cache['scaler_base']
                            available_cols = [col for col in feature_cols if col in pred_df.columns]
                    
                    # Get the sequence data
                    seq_data = pred_df[available_cols].tail(lookback)
                    
                    if len(seq_data) < lookback:
                        st.error(f"Not enough sequential data. Need {lookback} rows, but only have {len(seq_data)}")
                        return
                    
                    X_seq = seq_data.values.reshape(1, lookback, -1)  # Add batch dimension
                    
                    # Reference close price (current close)
                    ref_close = close_price
                    
                    # Clip and scale if quantiles are available
                    if ql is not None and qh is not None and len(ql) == X_seq.shape[-1]:
                        # Only clip if dimensions match
                        X_seq_c = np.clip(X_seq, ql[:X_seq.shape[-1]], qh[:X_seq.shape[-1]])
                    else:
                        X_seq_c = X_seq
                    
                    # Scale the sequence
                    X_seq_s = np.zeros_like(X_seq_c)
                    for i in range(len(X_seq_s)):
                        # Handle potential dimension mismatch
                        if X_seq_c[i].shape[-1] == n_features:
                            X_seq_s[i] = scaler.transform(X_seq_c[i])
                        else:
                            # If dimensions don't match, fit a new scaler
                            temp_scaler = StandardScaler()
                            temp_scaler.fit(X_seq_c[i])
                            X_seq_s[i] = temp_scaler.transform(X_seq_c[i])
                    
                    # Make prediction based on model type
                    pred_return = None
                    
                    if model_name == "XGBoost":
                        model = st.session_state.model_cache.get(model_name)
                        if model is None:
                            # Try to load from disk
                            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
                            if os.path.exists(model_path):
                                model = joblib.load(model_path)
                            else:
                                st.error("XGBoost model not found")
                                return
                        
                        # Flatten for XGBoost
                        X_flat = X_seq_s.reshape(1, -1)
                        
                        # Apply feature selection as in training
                        XGB_KEEP_FEATS = ['body','range_ma5','vr','or','body_lag1']
                        try:
                            # Find indices of XGB features
                            keep_idx = []
                            for feat in XGB_KEEP_FEATS:
                                if feat in available_cols:
                                    idx = available_cols.index(feat)
                                    keep_idx.append(idx)
                            
                            if len(keep_idx) > 0:
                                keep_idx = np.array(keep_idx, dtype=int)
                                n_actual_features = len(available_cols)
                                keep_cols = np.hstack([t*n_actual_features + keep_idx for t in range(lookback)])
                                # Ensure we don't exceed array bounds
                                keep_cols = keep_cols[keep_cols < X_flat.shape[1]]
                                X_use = X_flat[:, keep_cols] if len(keep_cols) > 0 else X_flat
                            else:
                                X_use = X_flat
                        except Exception as e:
                            st.warning(f"Feature selection failed: {e}. Using all features.")
                            X_use = X_flat
                        
                        # Predict
                        try:
                            pred_return = model.predict(X_use)[0]
                        except Exception as e:
                            st.error(f"XGBoost prediction failed: {e}")
                            return
                            
                    elif model_name == "LSTM Base" or model_name in ["Residual-Anchor-LSTM", "Stable-LSTM", "Attentive-Gated-LSTM", "MC-Enhanced-LSTM"]:
                        # Try to recreate the model if needed
                        input_size = X_seq_s.shape[-1]
                        
                        if model_name == "LSTM Base":
                            model = LSTMModel(input_size=input_size, hidden_size=24, num_layers=2,
                                            dropout=0.65, tail_k=8, bidirectional=True).to(device)
                        elif model_name == "Attentive-Gated-LSTM":
                            model = LSTMAttPool(input_size=input_size, hidden_size=72, num_layers=2,
                                              dropout=0.35, tail_k=18, bidirectional=True).to(device)
                        else:
                            # Other LSTM variants
                            config = {
                                "Residual-Anchor-LSTM": (56, 0.55, 10),
                                "Stable-LSTM": (72, 0.45, 14),
                                "MC-Enhanced-LSTM": (76, 0.42, 16)
                            }
                            if model_name in config:
                                hidden, dropout, tail_k = config[model_name]
                                model = LSTMModel(input_size=input_size, hidden_size=hidden, num_layers=2,
                                                dropout=dropout, tail_k=tail_k, bidirectional=True).to(device)
                        
                        # Try to load saved weights
                        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
                        if os.path.exists(model_path):
                            try:
                                state_dict = torch.load(model_path, map_location=device)
                                model.load_state_dict(state_dict, strict=False)
                            except:
                                st.warning(f"Could not load saved weights for {model_name}. Using random initialization.")
                        
                        model.eval()
                        
                        # Make prediction with MC dropout
                        X_tensor = torch.FloatTensor(X_seq_s).to(device)
                        mc_T = 16 if model_name == "LSTM Base" else 20 if "Residual" in model_name else 24
                        
                        mc_preds = []
                        with torch.no_grad():
                            for _ in range(mc_T):
                                model.train()  # Enable dropout
                                pred = model(X_tensor).squeeze().cpu().numpy()
                                mc_preds.append(float(pred) if pred.ndim == 0 else pred.item())
                        model.eval()
                        
                        # Average MC predictions
                        pred_return = np.mean(mc_preds)
                        
                        # Apply stored calibration if available
                        model_info = st.session_state.model_cache.get(model_name, {})
                        if isinstance(model_info, dict) and 'y_mu' in model_info:
                            # Denormalize and calibrate
                            y_mu = model_info.get('y_mu', 0)
                            y_sd = model_info.get('y_sd', 1)
                            a_cal = model_info.get('a_cal', 1)
                            b_cal = model_info.get('b_cal', 0)
                            
                            pred_return = pred_return * y_sd + y_mu
                            pred_return = a_cal * pred_return + b_cal
                    
                    if pred_return is not None:
                        # Convert return to price
                        predicted_price = price_from_return(pred_return, ref_close)
                        
                        # Display results
                        st.success("✅ Prediction Generated Successfully!")
                        
                        # Results display
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        # —— 新增：把本次预测结果保存起来，供投资分析实时使用 ——
                        st.session_state.manual_result = {
                            "current_price": float(ref_close),
                            "predicted_price": float(predicted_price),
                            "change_pct": float(((predicted_price - ref_close) / ref_close) * 100),
                            "lookahead": int(lookahead),
                            "model_name": model_name
                        }
                        with col1:
                            st.metric(
                                label="Current Price",
                                value=f"${ref_close:.2f}"
                            )
                        
                        with col2:
                            st.metric(
                                label=f"Predicted Price ({lookahead} days)",
                                value=f"${predicted_price:.2f}",
                                delta=f"${predicted_price - ref_close:.2f}"
                            )
                        
                        with col3:
                            change_pct = ((predicted_price - ref_close) / ref_close) * 100
                            st.metric(
                                label="Expected Return",
                                value=f"{change_pct:+.2f}%",
                                delta=f"{'🔴' if change_pct < 0 else '🟢'}"
                            )
                        
                    else:
                        st.error("Failed to generate prediction. Please check your inputs and try again.")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.info("Tips: Ensure your uploaded data has sufficient historical records (at least 200 rows recommended) and that all price values are reasonable.")
                    # Debug info
                    with st.expander("Debug Information"):
                        st.code(f"Error details: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# Additional helper function to save LSTM models properly during training
def save_lstm_model_properly(model, model_name, model_info):
    """Enhanced function to save LSTM models with all necessary information"""
    
    # Save model state dict
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
    
    # Save model info (calibration parameters, architecture details, etc.)
    info_path = os.path.join(MODEL_DIR, f"{model_name}_info.pkl")
    joblib.dump(model_info, info_path)
    
    return model_path, info_path
def _infer_daily_growth_from_model(model_name: str):
    preds = st.session_state.predictions_cache.get(model_name)
    if preds is None or len(preds['test']['pred']) < 2:
        return None, None, "Not enough predicted points to estimate daily growth."

    pred_series = np.asarray(preds['test']['pred'], dtype=float)
    P0 = float(pred_series[0]); PT = float(pred_series[-1])
    if P0 <= 0 or PT <= 0:
        return None, None, "Invalid predicted price series."

    dates_test = st.session_state.data_cache.get('dates_test', [])
    if isinstance(dates_test, (list, np.ndarray)) and len(dates_test) >= 2:
        try:
            d_days = int((pd.to_datetime(dates_test[-1]) - pd.to_datetime(dates_test[0])).days)
            d_days = max(1, d_days)
        except Exception:
            d_days = max(1, len(pred_series) - 1)
    else:
        d_days = max(1, len(pred_series) - 1)

    g_daily = (PT / P0) ** (1.0 / d_days)
    return g_daily, d_days, None


def show_roi_calculator_tab(df):
    """🎯 Goal-Based ROI Calculator: user enters target profit, system suggests investment strategy."""
    st.header("🎯 Goal-Based Investment Planner")

    if not st.session_state.models_trained:
        st.warning("Please train models first.")
        return

    left, right = st.columns(2)
    with left:
        target_profit = st.number_input(
            "Target Profit ($)", value=1000.0, min_value=1.0, step=100.0, format="%.2f",
            help="Enter how much profit you want to earn."
        )
        model_name = st.selectbox(
            "Select Model (for growth estimation)", list(st.session_state.predictions_cache.keys())
        )

    with right:
        mode = st.radio(
            "Select Strategy Mode",
            ("Fixed capital → calculate required holding days",
             "Fixed holding days → calculate required investment"),
            index=0
        )

        if mode == "Fixed capital → calculate required holding days":
            fixed_capital = st.number_input(
                "Available Investment ($)", value=10000.0, min_value=100.0, step=100.0, format="%.2f"
            )
        else:
            holding_days = st.number_input(
                "Planned Holding Period (days)", value=30, min_value=1, max_value=3650
            )

    # Estimate daily growth rate
    g_daily, d_base, err = _infer_daily_growth_from_model(model_name)
    if err:
        st.error(err)
        return

    st.info(
        f"**Estimated daily growth rate (g_daily): {(g_daily-1)*100:.4f}%**  "
        f"(based on {d_base} days of predicted test-set prices)"
    )

    st.markdown("---")

    if mode == "Fixed capital → calculate required holding days":
        if g_daily <= 1.0:
            st.warning("The estimated daily growth rate ≤ 0. Profit target cannot be achieved under this trend.")
            return

        # Solve for n: fixed_capital * (g_daily^n - 1) ≥ target_profit
        rhs = 1.0 + (target_profit / fixed_capital)
        if rhs <= 1.0:
            st.success("Your target is trivially achievable (check input values).")
            return

        days_needed = int(np.ceil(np.log(rhs) / np.log(g_daily)))
        final_value = fixed_capital * (g_daily ** days_needed)
        roi_pct = (final_value / fixed_capital - 1.0) * 100.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Required Holding Period", f"{days_needed} days")
        with c2:
            st.metric("Projected Final Value", f"${final_value:,.2f}")
        with c3:
            st.metric("Total ROI (est.)", f"{roi_pct:.2f}%")

    else:  # Fixed holding days → compute required investment
        growth_over_holding = (g_daily ** holding_days) - 1.0
        if growth_over_holding <= 0:
            st.warning("Under the current growth rate and holding period, expected return ≤ 0. Goal cannot be reached.")
            return

        required_capital = target_profit / growth_over_holding
        final_value = required_capital * (1.0 + growth_over_holding)
        roi_pct = growth_over_holding * 100.0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Required Investment", f"${required_capital:,.2f}")
        with c2:
            st.metric("Projected Final Value", f"${final_value:,.2f}")
        with c3:
            st.metric("ROI over period (est.)", f"{roi_pct:.2f}%")


        
# Run the app
if __name__ == "__main__":
    main()
