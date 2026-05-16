import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from config import HF_DATA_REPO, HF_DATA_FILE, HF_TOKEN

def load_master_data():
    path = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename=HF_DATA_FILE,
        repo_type="dataset",
        token=HF_TOKEN if HF_TOKEN else None
    )
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    return df

class DataManager:
    def __init__(self, tickers):
        self.df = load_master_data()
        self.tickers = [t for t in tickers if t in self.df.columns]
        if not self.tickers:
            raise ValueError(f"No tickers found: {tickers}")
        prices = self.df[self.tickers].ffill().bfill()
        self.returns = prices.pct_change().dropna()
    
    def get_returns(self, start_date=None, end_date=None):
        mask = pd.Series(True, index=self.returns.index)
        if start_date:
            mask &= (self.returns.index >= pd.to_datetime(start_date))
        if end_date:
            mask &= (self.returns.index <= pd.to_datetime(end_date))
        return self.returns.loc[mask]
