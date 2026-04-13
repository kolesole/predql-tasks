from copy import deepcopy

from tqdm import tqdm
import torch
from relbench.base import TaskType


class Trainer:

    def __init__(self, task, model, optimizer, scheduler, criterion, device):
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = total_count = 0.0

        for batch in tqdm(loader, desc="Training"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(batch, self.task.entity_table)
            if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                y = batch.y.to(torch.long)
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
    def evaluate(self, loader):
        self.model.eval()
        total_loss = total_count = 0.0
        labels_list = []
        logits_list = []
        
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(self.device)

            logits = self.model(batch, self.task.entity_table)
            if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                y = batch.y.to(torch.long)
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
    
    def train(self, loaders, num_epochs, tune_metric="loss", higher_is_better=False, patience=5):
        best_weights = None
        best_tune_metric = float("-inf") if higher_is_better else float("inf")
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(loaders["train"])
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

            eval_metrics = self.evaluate(loaders["val"])
            print(f"Epoch {epoch+1}/{num_epochs}, Eval Metrics: {eval_metrics}")

            cur_tune_metric = eval_metrics.get(tune_metric)

            if cur_tune_metric is None:
                raise ValueError(f"Metric '{tune_metric}' not found in evaluation metrics!")

            epochs_without_improvement += 1
            if (higher_is_better and cur_tune_metric >= best_tune_metric) or (
                not higher_is_better and cur_tune_metric <= best_tune_metric
            ):
                best_tune_metric = cur_tune_metric
                best_weights = deepcopy(self.model.state_dict())
                print(f"New best model found with {tune_metric}: {best_tune_metric:.4f}")
                epochs_without_improvement = 0

            if epochs_without_improvement >= patience:
                print(f"!!! No improvement in {tune_metric} for {patience} epochs (Early stopping) !!!")
                break
            
            self.scheduler.step(cur_tune_metric)

        self.model.load_state_dict(best_weights)

        return best_weights
