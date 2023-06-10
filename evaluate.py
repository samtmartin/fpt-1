from typing import Callable
import torch

from data_loader import DataLoader
from fpt import FPT
from torch import Tensor

@torch.no_grad()
def evaluate_model(
    model: FPT, 
    data_loader: DataLoader, 
    calculate_loss: Callable[[Tensor, Tensor], Tensor],
    eval_epochs: int, 
    batch_size: int,
    iter: int) -> Tensor:
    """
    Evaluates `model` during training against training and validation splits. 
    Returns the loss from validation split.
    """
    
    model.eval()

    train_loss = calculate_mean_loss(
        model, data_loader, calculate_loss, "train", eval_epochs, batch_size)
    val_loss = calculate_mean_loss(
        model, data_loader, calculate_loss, "val", eval_epochs, batch_size)

    model_median_errors, baseline_median_errors = (
        calculate_median_percent_error(
        model, data_loader, eval_epochs, run_standalone=False))

    print(f"Epoch #{iter + 1}:")
    print(f"  Train loss: {train_loss:.6f}")
    print(f"  Valid loss: {val_loss:.6f}")
    print(f"  % error (model): {model_median_errors}")
    print(f"  % error (y-o-y): {baseline_median_errors}")

    model.train()
    return val_loss

@torch.no_grad()
def calculate_median_percent_error(
    model: FPT, 
    data_loader: DataLoader,
    eval_epochs: int,
    run_standalone: bool) -> tuple[Tensor, Tensor]:
    """
    Evalutes `model` against test split to determine the percent error. First 
    output tensor is the model's median percent error and the second is the 
    baseline prediction strategy's median percent error.
    """

    if run_standalone:
        model.eval()

    model_percent_errors = torch.zeros(
        eval_epochs, data_loader.get_num_output_features())
    baseline_percent_errors = torch.zeros(
        eval_epochs, data_loader.get_num_output_features())

    for k in range(eval_epochs):
        inputs, targets = data_loader.get_batch("test", 1)  # (1,T,C)
        outputs = model(inputs)  # (1,T,C)
        targets = torch.squeeze(targets, dim=0) # (T,C)
        outputs = torch.squeeze(outputs, dim=0) # (T,C)

        T,C = targets.shape
        
        raw_targets = data_loader.generate_raw_data(targets)
        raw_outputs = data_loader.generate_raw_data(outputs)
        
        # Populate `model_percent_errors`.
        for i, estimate in enumerate(raw_outputs[T-1]):
            actual = raw_targets[T-1][i]
            model_percent_errors[k][i] = calculate_percent_error(
                estimate, actual)
            
        # Populate `baseline_percent_errors`.
        for i, actual in enumerate(raw_targets[T-1]):
            prediction = generate_yoy_prediction(raw_targets, T, i)
            baseline_percent_errors[k][i] = calculate_percent_error(
                prediction, actual)

    model_median_errors = (
        [round(e.item(), 2) for e in 
         torch.median(model_percent_errors, dim=0).values])
    baseline_median_errors = (
        [round(e.item(), 2) for e in 
         torch.median(baseline_percent_errors, dim=0).values])
    return model_median_errors, baseline_median_errors

def calculate_mean_loss(
        model: FPT, 
        data_loader: DataLoader,
        calculate_loss: Callable[[Tensor, Tensor], Tensor],
        split: str, 
        eval_epochs: int, 
        batch_size: int) -> Tensor:
    losses = torch.zeros(eval_epochs)
    for k in range(eval_epochs):
        inputs, targets = data_loader.get_batch(split, batch_size)
        outputs = model(inputs)
        loss = calculate_loss(outputs, targets)
        losses[k] = loss.item()
    return losses.mean()

def generate_yoy_prediction(
        raw_targets: Tensor, 
        block_size: int, 
        i: int) -> float:
    """
    Applies previous quarter's year-over-year growth rate to current 
    quarter's prior year value. This is a crude, but common, method for 
    forecasting business performance when there aren't fine-grained P&L 
    drivers available.
    """

    pq = raw_targets[block_size-2][i] # Prior quarter.
    pqpy = raw_targets[block_size-6][i] # Prior quarter's prior year.
    gr = pq / pqpy # Prior quarter's year-over-year growth rate.
    cqpy = raw_targets[block_size-5][i] # Current quarter's prior year.
    prediction = cqpy * gr
    return prediction

def calculate_percent_error(
        estimate, 
        actual):
    if actual == 0:
      return 0
    return abs(estimate - actual) / abs(actual) * 100.0
