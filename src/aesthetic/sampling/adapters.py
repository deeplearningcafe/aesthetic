import pandas as pd
from abc import ABC, abstractmethod
import glob
import os


class DataAdapter(ABC):
    @abstractmethod
    def read_data(self) -> pd.DataFrame:
        pass


class ParquetAdapter(DataAdapter):
    def __init__(self, parquet_path: str, num_parquets: int = 1):
        self.parquet_path = parquet_path
        self.num_parquets = num_parquets

    def read_data(self) -> pd.DataFrame:
        files = glob.glob(os.path.join(self.parquet_path, "*.parquet"))
        files = sorted(files)[: self.num_parquets]
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


class PriorAdapter(DataAdapter):
    def __init__(self, prior_csv_path: str):
        self.prior_csv_path = prior_csv_path

    def read_data(self) -> pd.DataFrame:
        if not os.path.exists(self.prior_csv_path):
            return pd.DataFrame()
        return pd.read_csv(self.prior_csv_path, low_memory=False)
