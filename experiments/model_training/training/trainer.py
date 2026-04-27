import gc
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from relbench.base import TaskType
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from .models.base_model import BaseModel


class Trainer:

    def __init__(
        self,
        task: TaskType,
        model: BaseModel,
        optimizer: Optimizer,
        criterion: _Loss,
        device: torch.device
    ) -> None:
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader: NeighborLoader) -> float:
        self.model.train()
        total_loss = total_count = 0.0

        for batch in tqdm(loader, desc="Training"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()


            logits = self.model(batch, self.task.entity_table)

            if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                y = batch.y.to(torch.long)
            elif self.task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                y = batch.y.to(torch.float)
            else:
                logits = logits.squeeze(-1)
                y = batch.y.to(torch.float)

            logits = logits[:y.size(0)]

            loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)

        return total_loss / total_count

    @torch.no_grad()
    def evaluate(self, loader: NeighborLoader) -> dict:
        self.model.eval()
        total_loss = total_count = 0.0
        labels_list = []
        logits_list = []

        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(self.device)


            logits = self.model(batch, self.task.entity_table)

            if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                y = batch.y.to(torch.long)
            elif self.task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                y = batch.y.to(torch.float)
            else:
                logits = logits.squeeze(-1)
                y = batch.y.to(torch.float)
            logits = logits[:y.size(0)]

            loss = self.criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)

            labels_list.append(y.cpu())
            logits_list.append(logits.cpu())

        metrics = self.task.compute_metrics(torch.cat(logits_list, dim=0).numpy(),
                                            torch.cat(labels_list, dim=0).numpy())
        metrics["loss"] = total_loss / total_count

        return metrics

    def train(
        self,
        loaders: NeighborLoader,
        num_epochs: int,
        tune_metric: str="loss",
        higher_is_better: bool=False,
        patience: int=5,
        print_every: int=1
    ) -> tuple[dict, dict]:
        best_weights = None
        best_tune_metric = float("-inf") if higher_is_better else float("inf")
        epochs_without_improvement = 0

        history = {
            "train_loss": [], "val_loss": [],
            "train_tune_metric": [], "val_tune_metric": []
        }

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(loaders["train"])
            train_metrics = self.evaluate(loaders["train"])

            val_metrics = self.evaluate(loaders["val"])
            val_loss = val_metrics["loss"]

            cur_train_tune_metric = train_metrics.get(tune_metric)
            cur_val_tune_metric = val_metrics.get(tune_metric)

            if cur_val_tune_metric is None:
                raise ValueError(f"Metric '{tune_metric}' not found in evaluation metrics!")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_tune_metric"].append(cur_train_tune_metric)
            history["val_tune_metric"].append(cur_val_tune_metric)

            if epoch == 0 or ((epoch + 1) % print_every == 0):
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train {tune_metric}: {cur_train_tune_metric:.4f}")
                print(f"Epoch {epoch+1}/{num_epochs} |   Val Loss: {val_loss:.4f} |   Val {tune_metric}: {cur_val_tune_metric:.4f}")

            epochs_without_improvement += 1
            if (higher_is_better and cur_val_tune_metric > best_tune_metric) or (
                not higher_is_better and cur_val_tune_metric < best_tune_metric
            ):
                best_tune_metric = cur_val_tune_metric
                best_weights = deepcopy(self.model.state_dict())
                print(f"New best model found with Val {tune_metric}: {best_tune_metric:.4f}")
                epochs_without_improvement = 0

            if epochs_without_improvement >= patience:
                print(f"!!! No improvement in Val {tune_metric} for {patience} epochs (Early stopping at epoch {epoch+1}) !!!")
                break

        self.model.load_state_dict(best_weights)

        return best_weights, history

    def plot_results(self, histories: list[dict], tune_metric: str) -> None:
        max_epoch = max(len(h["train_loss"]) for h in histories)

        def get_stats(name):
            arrays = []
            for h in histories:
                arr = np.full(max_epoch, np.nan)
                arr[:len(h[name])] = h[name]
                arrays.append(arr)
            stacked = np.vstack(arrays)
            return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)

        epochs = np.arange(1, max_epoch + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        train_loss_mean, train_loss_std = get_stats("train_loss")
        val_loss_mean, val_loss_std = get_stats("val_loss")

        train_metric_mean, train_metric_std = get_stats("train_tune_metric")
        val_metric_mean, val_metric_std = get_stats("val_tune_metric")

        loss_name = self.criterion.__class__.__name__

        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.plot(epochs, train_loss_mean, label=f"Train Loss ({loss_name})", color="blue")
        ax1.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color="blue", alpha=0.2)
        ax1.plot(epochs, val_loss_mean, label=f"Val Loss ({loss_name})", color="orange")
        ax1.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, color="orange", alpha=0.2)
        ax1.set_title(f"Loss ({loss_name}) over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(f"Loss ({loss_name})")
        ax1.legend()
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        tune_metric_cap = tune_metric.capitalize()
        ax2.plot(epochs, train_metric_mean, label=f"Train {tune_metric_cap}", color="blue")
        ax2.fill_between(epochs, train_metric_mean - train_metric_std, train_metric_mean + train_metric_std, color="blue", alpha=0.2)
        ax2.plot(epochs, val_metric_mean, label=f"Val {tune_metric_cap}", color="orange")
        ax2.fill_between(epochs, val_metric_mean - val_metric_std, val_metric_mean + val_metric_std, color="orange", alpha=0.2)
        ax2.set_title(f"{tune_metric_cap} over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(f"{tune_metric_cap}")
        ax2.legend()
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    def display_final_metrics(self, all_run_metrics: list[dict], num_runs: int) -> None:
        print("\n" + "="*40)
        print(f"FINAL STATS OVER {num_runs} RUNS")
        print("="*40)

        for split in ["train", "val", "test"]:
            split_metrics = [run[split] for run in all_run_metrics]

            metric_names = split_metrics[0].keys()

            print(f"\n[{split.upper()}] Metrics:")
            for metric_name in metric_names:
                metric_list = [run_metrics[metric_name] for run_metrics in split_metrics]

                metric_mean = np.mean(metric_list)
                metric_std = np.std(metric_list)

                display_name = metric_name.capitalize()

                print(f"  {display_name}: {metric_mean:.4f} ± {metric_std:.4f}")

        print("="*40 + "\n")

    def run_experiment(
        self,
        loaders: NeighborLoader,
        num_epochs: int,
        tune_metric: str="loss",
        higher_is_better: bool=False,
        patience: int=5,
        print_every: int=1,
        num_runs: int=1
    ) -> None:
        print("\n" + "="*40)
        print("MODEL INFO:")
        print("="*40)
        print(f"Model name: {self.model.gnn_name}")
        print(f"Number of parameters: {self.model.count_parameters}")
        print("="*40 + "\n")

        all_run_metrics = []
        all_histories = []

        for run in range(num_runs):
            gc.collect()
            torch.cuda.empty_cache()

            self.model.reset_parameters()
            self.optimizer = type(self.optimizer)(self.model.parameters(), **self.optimizer.defaults)

            print("\n" + "="*40)
            print(f"STARTING RUN {run+1}/{num_runs}")
            print("="*40)

            _, history = self.train(
                loaders, num_epochs, tune_metric, higher_is_better, patience, print_every
            )
            all_histories.append(history)

            final_metrics_train = self.evaluate(loaders["train"])
            final_metrics_val = self.evaluate(loaders["val"])
            final_metrics_test = self.evaluate(loaders["test"])

            print(f"Run {run+1}/{num_runs} | Final Train Metrics: {final_metrics_train}")
            print(f"Run {run+1}/{num_runs} | Final Val Metrics: {final_metrics_val}")
            print(f"Run {run+1}/{num_runs} | Final Test Metrics: {final_metrics_test}")

            final_metrics = {
                "train": final_metrics_train,
                "val": final_metrics_val,
                "test": final_metrics_test
            }

            all_run_metrics.append(final_metrics)

            print("="*40)
            print(f"FINISHED RUN {run+1}/{num_runs}")
            print("="*40 + "\n")

        self.plot_results(all_histories, tune_metric)
        self.display_final_metrics(all_run_metrics, num_runs)
