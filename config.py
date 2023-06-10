import torch

from dataclasses import dataclass
from torch import Tensor

@dataclass
class ColumnConfig:
    """
    Config for a column in dataset.

    Vars:
      -header: name of the data column header (e.g. 'revenue').
      -scaling_strategy: scaling technique to apply to column. Valid values 
       include: 'mean', 'standardization', 'log', 'log10' and sequential
       'sqrt->mean' and 'sqrt->mean->sqrt'.
      -lag_steps: number of lag steps to apply as input feature. Valid values: 
       0 - block_size. Generally improves model prediction accuracy.
      -include_yoy_prediction: whether or not to include a YoY prediction as 
       input feature. Generally improves model prediction accuracy.
      -is_output: whether or not to include metric as an output feature.
      -data_index: read only. Column's channel index in input data features.
      -mean: read only. Column's mean (used for scaling).
      -std: read only. Column's standard deviation (used for scaling).
      -scaling_adj: read only. Column's min value if negative and zero if 
       positive (used for scaling).
    """

    header: str
    scaling_strategy: str = "none"
    lag_steps: int = 0
    include_yoy_prediction: bool = False
    is_output: bool = False
    data_index: int = -1
    mean: Tensor = None
    std: Tensor = None
    scaling_adj: Tensor = None

@dataclass
class DataConfig:
    """
    Config used by `DataLoader` to build dataset. Determines: (1) the data 
    inputs to provide to the model and (2) the data outputs that should be 
    expected from the model's `forward` method.

    When running in 'forecast mode' (see forecast.py), every input column must 
    also be output so that the model's predictions can be stacked on top of one 
    another to generate data for multiple quarters.

    Vars:
      -financial_columns: the columns in data file to be included in dataset. 
       Values in list should only include columns that correspond to numerical/
       financial data for a specific company.
      -macro_columns: the macro columns to include in dataset. Valid header 
       values are: 'three_month_tbill', 'ten_year_tbill',
       'personal_consumption', and 'trucking'.
      -include_quarter_ended: whether or not the quarter ended month should 
       be included in dataset (e.g. if the month a given quarter ended is 
       "10", value would be 10 / 12).
      -include_sectors: whether or not the sector data should be included
       in dataset.
      -shuffle_data: whether or not to shuffle the data. Should always be set to 
       True unless order is required for testing. Seed is used.
      -split_company_data: whether or not to split each Company's data in half, 
       which roughly doubles the number of company partitions in dataset. This 
       increases the variety of the training split, allowing the model to see 
       more companies during training. Generally improves model prediction 
       accuracy.
      -percent_train: percent of dataset to allocate to the training split. 
       Value cannot be greater than 1. Percent of dataset allocated to val and 
       test splits is the result of subtracting 1 by `percent_train`. Result is 
       divided by two inorder to allocate data to both splits.
    """

    financial_columns: list[ColumnConfig]
    macro_columns: list[ColumnConfig]
    include_quarter_ended: bool
    include_sectors: bool
    shuffle_data: bool
    split_company_data: bool
    percent_train: float