import torch

from config import ColumnConfig, DataConfig
from data_loader import DataLoader

"""
'Bare-bones' forecasting tool. Loads a model checkpoint and forecasts a given 
number of quarters (`num_predictions`) of financial performance. The input and
output dataset config (i.e. DataConfig) must be identical in order to stack 
multiple quarters of predictions).

The below hyperparams were used to train model used for forecast demo.

batch_size = 8
block_size = 8
num_heads = 2
num_layers = 2
dropout = 0.1
"""

# Constants.
model_path = "./checkpoints/val_loss-0.000680.pt"
device = "cpu"
num_predictions = 4
block_size = 8

# Rebuild same dataset used for model training (not ideal). 
data_config = DataConfig(
    financial_columns=[
      ColumnConfig(
        header="revenue", scaling_strategy="log10", is_output=True),
      ColumnConfig(
        header="gross_profit", scaling_strategy="log10", is_output=True),
      ColumnConfig(
        header="op_income", scaling_strategy="log10", is_output=True),
      ColumnConfig(
        header="net_income", scaling_strategy="log10", is_output=True),
    ],
    macro_columns=[],
    include_quarter_ended=False,
    include_sectors=False,
    shuffle_data = False,
    split_company_data = False,
    percent_train = 0.7,
)

data_loader = DataLoader(data_config, block_size)

model = torch.load(model_path, map_location=torch.device(device))

model.eval()

inputs, targets = data_loader.get_forecast_data(num_predictions)
targets = torch.squeeze(targets)
outputs = model.forecast(inputs, num_predictions)

raw_outputs = data_loader.generate_raw_data(outputs)[-num_predictions:, :]
raw_targets = data_loader.generate_raw_data(targets)
print("Predictions: ", raw_outputs)
print("Actual: ", raw_targets)