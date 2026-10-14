"""Data/configuration helpers for the retained Glu workflow.

This module deliberately has no dependency on historical experiments or
generated outputs. The exploratory validation IDs are preserved in config/.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR/'config/glu_current_workflow.json'


def load_config():
    config = json.loads(CONFIG_PATH.read_text())
    if config['schema_version'] != 1 or config['sugar'] != 'Glu':
        raise ValueError('Unsupported Glu workflow configuration')
    return config


def validation_ids():
    return set(load_config()['validation_ids'])


def load_data():
    """Preserve the historical Glu row filters, sorting, and generation-0 rule."""
    columns = ['source','lineage','generation','time','alpha','f','length_birth']
    data = pd.read_csv(PROJECT_DIR/'data/processed/processed_input_data.csv', usecols=columns)
    for column in columns[1:]:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna(subset=columns)
    data = data.loc[data.source.eq('Glu') & data.time.gt(0) & data.alpha.gt(0)
                    & data.f.between(0,1,inclusive='neither') & data.length_birth.gt(0)]
    data = data.sort_values(['lineage','generation'])
    initial_ids = data.loc[data.generation.eq(0),'lineage'].unique()
    data = data.loc[data.lineage.isin(initial_ids)].copy()
    groups = [g.sort_values('generation').reset_index(drop=True)
              for _,g in data.groupby('lineage',sort=False)]
    return data.reset_index(drop=True), groups


def corr(x,y):
    x = x-x.mean(axis=-1,keepdims=True)
    y = y-y.mean(axis=-1,keepdims=True)
    return np.sum(x*y,axis=-1)/np.sqrt(np.sum(x*x,axis=-1)*np.sum(y*y,axis=-1))
