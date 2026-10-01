import pandas as pd
from collections import defaultdict


class History:
    def __init__(self):
        self.data = defaultdict(list)

    def reset(self):
        self.data = defaultdict(list)

    def log(self, name, value):
        self.data[name].append(value)

    def get(self, name: str, default=None):
        try:
            return self.data[name]
        except KeyError as e:
            raise e

    def list(self):
        # return list of variables names logged
        return list(self.data.keys())

    def get_df(self):
        return pd.DataFrame(self.data)

    def filter(self, keywords=["cost"], as_df=True):
        d = self.data
        for word in keywords:
            d = {k: self.data[k] for k in d.keys() if word in k}

        if as_df:
            return pd.DataFrame(d)

        return d

    def get_costs_summary(self, warmup_steps=0):
        df = self.filter(["cost"]).iloc[warmup_steps:, :]
        unique_attr_names = {c for c in df.columns}
        attrs = {c: [] for c in unique_attr_names}
        for i, cost_name in enumerate(unique_attr_names):
            current_costs = [c for c in df.columns if cost_name in c]
            cost_df = df[current_costs]
            s = cost_df.sum(axis=1).sum()
            attrs[cost_name] = s
        # attrs = {k: v for k, v in sorted(attrs.items(), key=lambda item: item[1])}
        return attrs