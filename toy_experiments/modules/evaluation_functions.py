import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_ensemble(models, loader, DEVICE):
    """
    Function to evaluate a uniform ensemble of NNs independently
    of how models were trained. All models are assigned the same
    weight (1/M).
    """
    all_probs = []

    for model in models:
        model.eval()
        probs = []

        for batch in loader:
            images, *rest = batch
            images = images.to(DEVICE)
            logits = model(images)
            prob = F.softmax(logits, dim=1)
            probs.append(prob)

        probs = torch.cat(probs, dim=0)
        all_probs.append(probs)

    # Shape: [num_models, num_samples, num_classes]
    ensemble_probs = torch.stack(all_probs, dim=0)
    # Predictive mean
    mean_probs = ensemble_probs.mean(dim=0)
    # Predictive uncertainty (entropy of the mean)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)

    # Aleatoric uncertainty (expected entropy): E[ H[p] ]
    per_model_entropy = -torch.sum(
        ensemble_probs * torch.log(ensemble_probs + 1e-10), dim=2
    )  # [M, N]
    aleatoric = per_model_entropy.mean(dim=0)  # [N]

    # Compute lower bound
    log_probs = torch.log(ensemble_probs + 1e-10)  # [M, N, C]
    max_log_probs, _ = log_probs.max(dim=0)  # [N, C]
    lower_bound = -torch.sum(mean_probs * max_log_probs, dim=1)  # [N]
    upper_bound = lower_bound + torch.log(torch.tensor(len(models), dtype=torch.float32))

    predictions = mean_probs.cpu()
    entropy = entropy.cpu()
    lower_bound = lower_bound.cpu()
    upper_bound = upper_bound.cpu()
    aleatoric = aleatoric.cpu()
    return predictions, entropy, lower_bound, upper_bound, aleatoric


###### EVALUATE PERFORMANCE METRICS (ACCURACY) ######
@torch.no_grad()
def evaluate_ensemble_performance(models, test_loader, DEVICE):
    """
    Evaluates ensemble performance on:
    - full test set (ID + OOD)
    - ID-only subset

    Assumes test_loader yields (x, y, is_id),
    where is_id is a boolean tensor indicating ID samples.
    """

    for model in models:
        model.eval()

    num_models = len(models)

    correct_per_model = [0 for _ in range(num_models)]
    correct_per_model_id = [0 for _ in range(num_models)]

    correct_ensemble = 0
    correct_ensemble_id = 0

    total = 0
    total_id = 0

    for batch in test_loader:

        if len(batch) == 2:
            x, y = batch
            is_id = torch.ones_like(y, dtype=torch.bool)
        elif len(batch) == 3:
            x, y, is_ood = batch
            is_id = 1 - is_ood
        else:
            raise ValueError("test_loader must yield (x, y) or (x, y, is_id)")

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        is_id = is_id.to(DEVICE).bool()

        batch_size = y.size(0)
        total += batch_size
        total_id += is_id.sum().item()

        # Collect logits
        logits = torch.stack([model(x) for model in models], dim=0)
        # [M, B, C]

        probs = F.softmax(logits, dim=-1)

        # ----- Individual model accuracy -----
        for m in range(num_models):
            preds = probs[m].argmax(dim=-1)

            correct_per_model[m] += (preds == y).sum().item()
            correct_per_model_id[m] += ((preds == y) & is_id).sum().item()

        # ----- Ensemble accuracy (mean probability) -----
        ensemble_probs = probs.mean(dim=0)
        ensemble_preds = ensemble_probs.argmax(dim=-1)

        correct_ensemble += (ensemble_preds == y).sum().item()
        correct_ensemble_id += ((ensemble_preds == y) & is_id).sum().item()

    # Final accuracies
    acc_per_model = [c / total for c in correct_per_model]
    acc_per_model_id = [c / total_id if total_id > 0 else float("nan") for c in correct_per_model_id]

    ensemble_acc = correct_ensemble / total
    ensemble_acc_id = correct_ensemble_id / total_id if total_id > 0 else float("nan")

    return acc_per_model, ensemble_acc, acc_per_model_id, ensemble_acc_id
