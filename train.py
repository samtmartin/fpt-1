import torch

from config import ColumnConfig, DataConfig
from evaluate import evaluate_model, evaluate_trained_model
from data_loader import DataLoader
from fpt import FPT
from torch import Tensor
from torch.nn import functional as F

# MODEL HYPERPARAMETERS
batch_size = 16  # Independent sequences processed in parallel.
block_size = 16  # Context length used for predictions.
num_heads = 6
num_layers = 6
dropout = 0.1

# TRAINING PARAMETERS
train_epochs = 15000
eval_interval = train_epochs * 0.1
eval_epochs = 200
learning_rate = 0.001

# DATALOADER PARAMETERS
data_config = DataConfig(
    financial_columns=[
      ColumnConfig(header="revenue", scaling_strategy="log10", 
                   lag_steps=4, include_yoy_prediction=True, is_output=True),
      ColumnConfig(header="gross_profit", scaling_strategy="log10", 
                   lag_steps=4, include_yoy_prediction=True, is_output=True),
      ColumnConfig(header="op_income", scaling_strategy="log10", 
                   lag_steps=4, include_yoy_prediction=True, is_output=True),
      ColumnConfig(header="net_income", scaling_strategy="log10", 
                   lag_steps=4, include_yoy_prediction=True, is_output=True),
      ColumnConfig(header="sga_expense", scaling_strategy="log10"),
      ColumnConfig(header="rd_expense", scaling_strategy="log10"),
      ColumnConfig(header="dep_amort", scaling_strategy="log10"),
      ColumnConfig(header="other_opex", scaling_strategy="log10"),
      ColumnConfig(header="interest_expense", scaling_strategy="log10"),
      ColumnConfig(header="interest_income", scaling_strategy="log10"),
      ColumnConfig(header="fx_gain", scaling_strategy="log10"),
      ColumnConfig(header="other_non_op_income", scaling_strategy="log10"),
      ColumnConfig(header="unusual_expenses", scaling_strategy="log10"),
      ColumnConfig(header="tax_expense", scaling_strategy="log10"),
      ColumnConfig(header="disc_ops", scaling_strategy="log10"),
      ColumnConfig(header="extraordinary_items", scaling_strategy="log10"),
      ColumnConfig(header="minority_interest", scaling_strategy="log10"),
      ColumnConfig(header="total_assets", scaling_strategy="log10"),
      ColumnConfig(header="total_liabilities", scaling_strategy="log10"),
      ColumnConfig(header="total_equity", scaling_strategy="log10"),
      ColumnConfig(header="cash_short_term_inv", scaling_strategy="log10"),
      ColumnConfig(header="receivables", scaling_strategy="log10"),
      ColumnConfig(header="current_assets", scaling_strategy="log10"),
      ColumnConfig(header="net_ppe", scaling_strategy="log10"),
      ColumnConfig(header="current_liabilities", scaling_strategy="log10"),
      ColumnConfig(header="total_debt", scaling_strategy="log10"),
      ColumnConfig(header="cash_from_ops", scaling_strategy="log10"),
      ColumnConfig(header="cash_from_inv", scaling_strategy="log10"),
      ColumnConfig(header="cash_from_fin", scaling_strategy="log10"),
      ColumnConfig(header="capex", scaling_strategy="log10"),
      ColumnConfig(header="debt_issued", scaling_strategy="log10"),
      ColumnConfig(header="debt_repaid", scaling_strategy="log10"),
      ColumnConfig(header="tev_ltm_revenue", scaling_strategy="log10"),
    ],
    macro_columns=[
      ColumnConfig(header="three_month_tbill", scaling_strategy="log10", 
                   lag_steps=3),
      ColumnConfig(header="ten_year_tbill", scaling_strategy="log10", 
                   lag_steps=3),
      ColumnConfig(header="personal_consumption", scaling_strategy="log10", 
                   lag_steps=3),
      ColumnConfig(header="trucking", scaling_strategy="mean", 
                   lag_steps=3),
    ],
    include_quarter_ended=True,
    include_sectors=True,
    shuffle_data = True,
    split_company_data = True,
    percent_train = 0.7,
)

def log_training_metrics(
        data_loader: DataLoader, 
        model: FPT) -> None:
    # Single use vars required to keep print statements on one line.
    companies = len(data_loader.get_partition("train")) + len(
        data_loader.get_partition("val")) + len(
        data_loader.get_partition("test"))
    T,C = data_loader.get_data().shape
    params = sum(p.numel() for p in model.parameters())
    num_in = data_loader.get_num_input_features()
    num_out = data_loader.get_num_output_features()
    print(f"DataLoader has {companies} company partitions and {T} records")
    print(f"Model has {params} trainable parameters")
    print(f"Model has {num_in} input and {num_out} output features")

def calculate_loss(
        outputs: Tensor, 
        targets: Tensor) -> Tensor:
    B, T, C = outputs.shape

    outputs = outputs.view(B*T, C)
    targets = targets.view(B*T, C)

    # Consistent with https://arxiv.org/pdf/2001.08317.pdf.
    loss = F.huber_loss(outputs, targets)
    return loss

def train(
        model: FPT, 
        optimizer: torch.optim.Adam, 
        data_loader: DataLoader,
        lr_step_down_iter: int,
        best_loss: float,
        save_model: bool,
        evaluate_against_test_split: bool) -> None:
    log_training_metrics(data_loader, model)

    for iter in range(train_epochs):
        if iter == lr_step_down_iter:
            lr = learning_rate / 10
            print(f"Reducing learning rate to {lr}.")
            optimizer.param_groups[0]['lr'] = lr

        if iter % eval_interval == 0 or iter == train_epochs - 1:
            eval = evaluate_model(
                model, data_loader, calculate_loss, eval_epochs, batch_size)
            train_loss = eval["train"]
            val_loss = eval["val"]
            
            print(f"Epoch #{iter + 1}:")
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Valid loss: {val_loss:.6f}")

            if save_model and val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    model,
                    f"checkpoints/val_loss-{val_loss:.6f}.pt")

        inputs, targets = data_loader.get_batch("train", batch_size)
        outputs = model(inputs)
        loss = calculate_loss(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if evaluate_against_test_split:
        evaluate_trained_model(model, data_loader, eval_epochs)

data_loader = DataLoader(data_config, block_size)

model = FPT(
    data_loader.get_num_input_features(), 
    data_loader.get_num_output_features(),
    block_size, num_heads, num_layers, dropout)

# Consistent with https://arxiv.org/pdf/2001.08317.pdf.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(
    model, optimizer, data_loader, int(train_epochs/2), best_loss=float('inf'), 
    save_model=False, evaluate_against_test_split=True)