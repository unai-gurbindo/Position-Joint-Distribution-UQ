#### SCRIPT FOR TRAINING AN ENSEMBLE OF 10 MODELS ON CIFAR-10 AND COMPUTE THEIR UNCERTAINTY BOUNDARIES ####
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import os
import sys
sys.path.append('/home/aaldoma/Projects/PositionUncert_Anonym')

from toy_experiments.modules.models import SimpleCNN
from toy_experiments.modules.training_loops import train_model
from toy_experiments.modules.dataset_utils import cifar10_loaders

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train ensemble on CIFAR-10 Dataset')
    parser.add_argument('--epochs', type=int, default=50, required=False, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size for training')
    parser.add_argument('--num_models', type=int, default=10, required=False, help='Number of models in the ensemble')
    parser.add_argument('--seed', type=int, default=42, required=False, help="Seed for reproducibility")
    parser.add_argument('--ensemble_type', type=str, default="deep", required=True, help="Type of ensemble among DE, JE, MoE...")
    parser.add_argument('--data_transformation', type=str, default="transf_data", help="Transformation to apply in the data loader.")
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

    # Set random seeds for reproducibility
    set_seed(config["seed"])

    data_transformation = config["data_transformation"]  # or "transf_data" depending on training

    train_loader, val_loader, test_loader = cifar10_loaders(config, data_transformation=data_transformation)

    # Initialize the ensemble
    # ensemble_models = [SimpleCNN().to(DEVICE) for _ in range(config["num_models"])]

    # Train ensemble of models
    ensemble_models = []
    if config["ensemble_type"] == "deep":
        for i in range(config["num_models"]):
            new_seed = config["seed"] + i
            print(f"New seed is: {new_seed}")
            print(f"Training model {i+1}/{config['num_models']}")
            set_seed(new_seed)
            model = SimpleCNN().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(config["epochs"]):
                train_model(model, train_loader, optimizer, DEVICE)
                if (epoch + 1) % 10 == 0:
                    print(f" Model {i+1}, Epoch {epoch+1}/{config['epochs']} completed.")
            ensemble_models.append(model)

    else:
        raise ValueError(f"Ensemble type {config['ensemble_type']} not implemented yet.")

    # Save the models weights for future use
    ensemble_folder = f"toy_experiments/saved_models/cifar10/{config['ensemble_type']}_{data_transformation}"
    os.makedirs(ensemble_folder, exist_ok=True)
    ensemble_number = len([f for f in os.listdir(ensemble_folder) if f.startswith("ensemble_") and f.endswith(".pth")])
    os.makedirs(f"{ensemble_folder}/ensemble_{config['seed']}", exist_ok=True)
    for i, model in enumerate(ensemble_models):
        torch.save(model.state_dict(), f"{ensemble_folder}/ensemble_{config['seed']}/ensemble_model_{i}.pth")
    
