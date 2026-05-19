# Load the pytorch models and evaluate ensemble uncertainties
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import argparse
import os
import sys
sys.path.append('/home/aaldoma/Projects/PositionUncert_Anonym')

from toy_experiments.modules.models import SimpleCNN
from toy_experiments.modules.evaluation_functions import evaluate_ensemble, evaluate_ensemble_performance
from toy_experiments.modules.dataset_utils import cifar10_loaders


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train ensemble on CIFAR-10 Dataset')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size for training')
    parser.add_argument('--num_models', type=int, default=5, required=False, help='Number of models in the ensemble')
    parser.add_argument('--seed', type=int, default=42, required=False, help="Seed for reproducibility")
    parser.add_argument('--order', type=str, default="lower", required=False, help="In which order arrays will be aranged.")
    parser.add_argument('--ensemble_type', type=str, default="deep", required=True, help="Type of ensemble among DE, JE, MoE...")
    parser.add_argument('--data_transformation', type=str, default="transf_data", help="Transformation to apply in the data loader.")
    parser.add_argument('--ood_fraction', type=float, default=0.0, help="Fraction of OOD samples in test set (0.0-1.0)")
    parser.add_argument('--ood_dataset', type=str, default="cifar10c", help="Type of OOD dataset: cifar10c or svhn")
    parser.add_argument('--cifar10c_corruption', type=str, default="gaussian_noise", help="CIFAR-10C corruption type")
    parser.add_argument('--cifar10c_severity', type=int, default=3, help="CIFAR-10C severity level (1-5)")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Parse command-line arguments
    args = parse_arguments()
    config = vars(args)

    # Load CIFAR-10 test dataset
    data_transformation = config["data_transformation"]  # or "transf_data" depending on training

    _, _, cifar10_test_loader = cifar10_loaders(config, data_transformation=data_transformation)
    cifar10_test_dataset = cifar10_test_loader.dataset

    if config["ood_fraction"] > 0.0:
        from toy_experiments.modules.dataset_utils import DatasetWithOODFlag
        num_id = int((1 - config["ood_fraction"]) * len(cifar10_test_dataset))
        num_ood = len(cifar10_test_dataset) - num_id

        id_subset, _ = random_split(
            cifar10_test_dataset,
            [num_id, num_ood],
            generator=torch.Generator().manual_seed(42)
        )

        transform = cifar10_test_dataset.transform

        if config["ood_dataset"] == "cifar10c":
            from toy_experiments.modules.dataset_utils import CIFAR10C
            ood_data = CIFAR10C(
                root="data/cifar10c",
                corruption=config["cifar10c_corruption"],
                severity=config["cifar10c_severity"],
                transform=transform
            )
        elif config["ood_dataset"] == "svhn":
            from toy_experiments.modules.dataset_utils import load_svhn_dataset
            ood_data = load_svhn_dataset(
                root="data/svhn",
                transform=transform
            )

        ood_subset, _ = random_split(
            ood_data,
            [num_ood, len(ood_data) - num_ood],
            generator=torch.Generator().manual_seed(42)
        )

        id_subset = DatasetWithOODFlag(id_subset, is_ood=False)
        ood_subset = DatasetWithOODFlag(ood_subset, is_ood=True)

        mixed_test_dataset = torch.utils.data.ConcatDataset([id_subset, ood_subset])

        test_loader = DataLoader(
            mixed_test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # is_ood = torch.cat([
        #     torch.zeros(len(id_subset)),
        #     torch.ones(len(ood_subset))
        # ])

        # Create is_ood vector
        is_ood = []
        for _, _, ood_flag in mixed_test_dataset:
            is_ood.append(ood_flag)
        is_ood = torch.tensor(is_ood)

    else:
        test_loader = cifar10_test_loader
        is_ood = torch.zeros(len(cifar10_test_dataset))

    entropies_list, lower_bound_list, upper_bound_list, aleatoric_list = [], [], [], []
    accuracy_per_sample_list, true_class_probability_list, max_class_probability_list = [], [], []
    acc_per_model_list, ensemble_acc_list, acc_per_model_id_list, ensemble_acc_id_list = [], [], [], []
    for seed in range(5):
        config["seed"] = seed
        # Ensemble folder
        ensemble_folder = f"toy_experiments/saved_models/cifar10/{config['ensemble_type']}_{data_transformation}/ensemble_{config['seed']}"
        # num_models = len([f for f in os.listdir(ensemble_folder) if f.startswith("ensemble_model_") and f.endswith(".pth")])
        # config["num_models"] = num_models
        # Load saved ensemble models
        ensemble_models = []
        for i in range(config["num_models"]):
            new_seed = config["seed"] + i
            set_seed(new_seed)
            model = SimpleCNN().to(DEVICE)
            model.load_state_dict(torch.load(f"{ensemble_folder}/ensemble_model_{i}.pth", map_location=DEVICE))
            ensemble_models.append(model)

        # Evaluate ensemble on test set
        if config["ensemble_type"] == "deep" or config["ensemble_type"] == "joint":
            predictions, entropies, lower_bound, upper_bound, aleatoric = evaluate_ensemble(ensemble_models, test_loader, DEVICE)
            acc_per_model, ensemble_acc, acc_per_model_id, ensemble_acc_id = evaluate_ensemble_performance(ensemble_models, test_loader, DEVICE)
            print(f"Test set predictions shape: {predictions.shape}")
            print(f"Test set entropies shape: {entropies.shape}")
        else:
             raise ValueError(f"Ensemble type: {config['ensemble_type']} not implemented yet.")

        # Get accuracies
        y_test = []

        for batch in test_loader:
            labels = batch[1]
            y_test.append(labels)
        y_test = torch.cat(y_test, dim=0).to(DEVICE)  # shape (N,)

        # Move vectors to same devide
        y_test = y_test.cpu()
        predictions = predictions.cpu()
        pred_labels = predictions.argmax(dim=-1)
        accuracy_per_sample = (pred_labels == y_test).float()
        true_class_probability = predictions[torch.arange(len(y_test)), y_test]
        max_class_probability = predictions[torch.arange(len(pred_labels)), pred_labels]

        print(f"Accuracy vector: {accuracy_per_sample}")
        print(f"TCP: {true_class_probability}")
        print(f"MCP: {max_class_probability}")

        # Plot the uncertainty boundaries #
        entropies = entropies.numpy()
        lower_bound = lower_bound.numpy()
        upper_bound = upper_bound.numpy()
        aleatoric = aleatoric.numpy()

        entropies_list.append(entropies)
        lower_bound_list.append(lower_bound)
        upper_bound_list.append(upper_bound)
        aleatoric_list.append(aleatoric)
        accuracy_per_sample_list.append(accuracy_per_sample.numpy())
        true_class_probability_list.append(true_class_probability.numpy())
        max_class_probability_list.append(max_class_probability.numpy())
        acc_per_model_list.append(acc_per_model)
        ensemble_acc_list.append(ensemble_acc)
        if "acc_per_model_id" in locals() and acc_per_model_id is not None:
            acc_per_model_id_list.append(acc_per_model_id)
            ensemble_acc_id_list.append(ensemble_acc_id)

        # Save the arrays for plotting later
        if config["ood_fraction"] > 0.0:
            if config["ood_dataset"] == "cifar10c":
                results_dir = f"toy_experiments/results/cifar10/{config['ensemble_type']}_{data_transformation}/{config['ood_dataset']}/num_models_{config['num_models']}/frac_{config['ood_fraction']}/sev_{config['cifar10c_severity']}/ensemble_{config['seed']}"
            elif config["ood_dataset"] == "svhn":
                results_dir = f"toy_experiments/results/cifar10/{config['ensemble_type']}_{data_transformation}/{config['ood_dataset']}/num_models_{config['num_models']}/frac_{config['ood_fraction']}/ensemble_{config['seed']}"
        else:
            results_dir = f"toy_experiments/results/cifar10/{config['ensemble_type']}_{data_transformation}/normal/num_models_{config['num_models']}/ensemble_{config['seed']}"
        os.makedirs(results_dir, exist_ok=True)
        np.savez(f"{results_dir}/uncertainty_accuracy_vectors_no_order.npz", entropies=entropies, lower_bound=lower_bound, upper_bound=upper_bound, aleatoric=aleatoric,
        accuracy_per_sample=accuracy_per_sample, true_class_probability=true_class_probability, max_class_probability=max_class_probability, is_ood=is_ood.numpy())
        np.savez(f"{results_dir}/accuracy_per_model.npz", acc_per_model=acc_per_model, ensemble_acc=ensemble_acc, acc_per_model_id=acc_per_model_id, ensemble_acc_id=ensemble_acc_id)

        np.save(os.path.join(results_dir, "config"), config)
        # Also save as a human-readable text file
        with open(os.path.join(results_dir, "config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

    # Average results fot the different runs
    entropies = np.mean(np.array(entropies_list), axis=0)
    lower_bound = np.mean(np.array(lower_bound_list), axis=0)
    upper_bound = np.mean(np.array(upper_bound_list), axis=0)
    aleatoric = np.mean(np.array(aleatoric_list), axis=0)
    accuracy_per_sample = np.mean(np.array(accuracy_per_sample_list), axis=0)
    true_class_probability = np.mean(np.array(true_class_probability_list), axis=0)
    max_class_probability = np.mean(np.array(max_class_probability_list), axis=0)
    acc_per_model = np.mean(np.array(acc_per_model_list), axis=0)
    ensemble_acc = np.mean(np.array(ensemble_acc_list), axis=0)
    if "acc_per_model_id_list" in locals() and len(acc_per_model_id_list) > 0:
        acc_per_model_id = np.mean(np.array(acc_per_model_id_list), axis=0)
        ensemble_acc_id = np.mean(np.array(ensemble_acc_id_list), axis=0)
        
    # Save the arrays for plotting later
    if config["ood_fraction"] > 0.0:
        if config["ood_dataset"] == "cifar10c":
            results_dir = f"toy_experiments/results/cifar10/{config['ensemble_type']}_{data_transformation}/{config['ood_dataset']}/num_models_{config['num_models']}/frac_{config['ood_fraction']}/sev_{config['cifar10c_severity']}"
        elif config["ood_dataset"] == "svhn":
            results_dir = f"toy_experiments/results/cifar10/{config['ensemble_type']}_{data_transformation}/{config['ood_dataset']}/num_models_{config['num_models']}/frac_{config['ood_fraction']}"
    else:
        results_dir = f"toy_experiments/results/cifar10/{config['ensemble_type']}_{data_transformation}/normal/num_models_{config['num_models']}"
    os.makedirs(results_dir, exist_ok=True)
    np.savez(f"{results_dir}/uncertainty_accuracy_vectors_no_order.npz", entropies=entropies, lower_bound=lower_bound, upper_bound=upper_bound, aleatoric=aleatoric,
    accuracy_per_sample=accuracy_per_sample, true_class_probability=true_class_probability, max_class_probability=max_class_probability, is_ood=is_ood.numpy())
    np.savez(f"{results_dir}/accuracy_per_model.npz", acc_per_model=acc_per_model, ensemble_acc=ensemble_acc, acc_per_model_id=acc_per_model_id, ensemble_acc_id=ensemble_acc_id)

    np.save(os.path.join(results_dir, "config"), config)
    # Also save as a human-readable text file
    with open(os.path.join(results_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")