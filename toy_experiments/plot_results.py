#### SCRIPT FOR TRAINING AN ENSEMBLE OF 10 MODELS ON CIFAR-10 AND COMPUTE THEIR UNCERTAINTY BOUNDARIES ####
import numpy as np
import torch

import argparse
import os
import sys
sys.path.append('/home/aaldoma/Projects/PositionUncert_Anonym')

from toy_experiments.modules.plotting_functions import plot_entropy_and_bounds_with_ood_bins_icml, plot_binned_averages_icml, plot_ensemble_histogram_icml, plot_risk_coverage

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train ensemble on CIFAR-10 Dataset')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size for training')
    parser.add_argument('--num_models', type=int, default=5, required=False, help='Number of models in the ensemble')
    parser.add_argument('--seed', type=int, default=42, required=False, help="Seed for reproducibility")
    parser.add_argument('--ensemble_type', type=str, default="deep", required=False, help="Type of ensemble among DE, JE, MoE...")
    parser.add_argument('--data_transformation', type=str, default="transf_data", help="Transformation to apply in the data loader.")
    parser.add_argument('--order', type=str, default="lower", required=False, help="In which order arrays will be aranged.")
    parser.add_argument('--ood_fractions', type=float, nargs='+', help="Fraction of OOD samples in test set (0.0-1.0)")
    parser.add_argument('--ood_dataset', type=str, default="svhn", help="Type of OOD dataset: cifar10c or svhn")
    parser.add_argument('--cifar10c_corruption', type=str, default="gaussian_noise", help="CIFAR-10C corruption type")
    parser.add_argument('--cifar10c_severities', type=int, nargs='+', help="CIFAR-10C severity level (1-5). It can be a list of severities. Pass as: --cifar10c_severity 3 4 5")
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

    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if config["ood_dataset"] == "svhn":
        config["cifar10c_severities"] = [0]  # Dummy severity for SVHN

    entropies, lower_bound, upper_bound, aleatoric, epistemic, accuracy_per_sample, true_class_probability, max_class_probability, is_ood = [], [], [], [], [], [], [], [], []
    acc_per_model, ensemble_acc, acc_per_model_id, ensemble_acc_id = [], [], [], []

    data_transformation = config["data_transformation"] # or "transf_data" depending on training

    base_results_path = f'/home/aaldoma/Projects/PositionUncert/toy_experiments/results/cifar10/{config["ensemble_type"]}_{data_transformation}/{config["ood_dataset"]}/num_models_{config["num_models"]}'

    for frac in config["ood_fractions"]:
        entropies_sev, lower_bound_sev, upper_bound_sev, aleatoric_sev, epistemic_sev, accuracy_per_sample_sev, true_class_probability_sev, max_class_probability_sev, is_ood_sev = [], [], [], [], [], [], [], [], []
        acc_per_model_sev, ensemble_acc_sev, acc_per_model_id_sev, ensemble_acc_id_sev = [], [], [], []
        for severity in config["cifar10c_severities"]:
            print(f"Severity: {severity}, Fraction: {frac}")
            
            if config["ood_dataset"] == "cifar10c":
                results_path = f'{base_results_path}/frac_{frac}/sev_{severity}/'
            elif config["ood_dataset"] == "svhn":
                results_path = f'{base_results_path}/frac_{frac}/'
            else:
                results_path = f'{base_results_path}/normal/'

            data = np.load(f"{results_path}/uncertainty_accuracy_vectors_no_order.npz")
            accuracies = np.load(f"{results_path}/accuracy_per_model.npz")

            entropies_sev.append(data['entropies'])
            lower_bound_sev.append(data['lower_bound'])
            upper_bound_sev.append(data['upper_bound'])
            aleatoric_sev.append(data['aleatoric'])
            epistemic_sev.append(data['entropies'] - data['aleatoric'])
            accuracy_per_sample_sev.append(data['accuracy_per_sample'])
            true_class_probability_sev.append(data['true_class_probability'])
            max_class_probability_sev.append(data['max_class_probability'])
            is_ood_sev.append(data['is_ood'])
            acc_per_model_sev.append(accuracies['acc_per_model'])
            ensemble_acc_sev.append(accuracies['ensemble_acc'])
            acc_per_model_id_sev.append(accuracies['acc_per_model_id'])
            ensemble_acc_id_sev.append(accuracies['ensemble_acc_id'])

            figures_path = f"{results_path}/figures/bin_100/"
            os.makedirs(figures_path, exist_ok=True)
            plot_entropy_and_bounds_with_ood_bins_icml(entropies_sev[-1], lower_bound_sev[-1], upper_bound_sev[-1], is_ood=is_ood_sev[-1], sort_by="lower_bound", aleatoric=aleatoric_sev[-1], title=f"Fraction of OOD {int(frac*100)}% Severity {severity}", num_bins=100, window=10, save_path=figures_path)
            plot_ensemble_histogram_icml(acc_per_model_sev[-1], ensemble_acc_sev[-1], acc_per_model_id_sev[-1], ensemble_acc_id_sev[-1], f"Fraction of 00D {int(frac*100)}% Severity {severity}", save_path=figures_path)

            true_class_probability_sev[-1] = true_class_probability_sev[-1] * (~np.asarray(is_ood_sev[-1]).astype(bool))

            uncertainties_ROC = {r"$1-p(y|x)$": 1-true_class_probability_sev[-1], 
                                # r"$1-p(x)p(y|x)$": 1-(true_class_probability_sev[-1]*(1-is_ood_sev[-1])), 
                                r"$1-p(\hat{y}|x)$": 1-max_class_probability_sev[-1], 
                                r"$1-p(x)p(\hat{y}|x)$": 1-(max_class_probability_sev[-1]*(1-is_ood_sev[-1])),
                                "Aleatoric": aleatoric_sev[-1],
                                "Epistemic": epistemic_sev[-1]}
            print(f"Plotting Risk-Coverage for Fraction {frac} Severity {severity}")
            plot_risk_coverage(uncertainties_ROC, accuracy_per_sample_sev[-1], ax=None, label=None, save_path=figures_path, is_ood=is_ood_sev[-1])
            

        entropies.append(entropies_sev)
        lower_bound.append(lower_bound_sev)
        upper_bound.append(upper_bound_sev)
        aleatoric.append(aleatoric_sev)
        epistemic.append(epistemic_sev)
        accuracy_per_sample.append(accuracy_per_sample_sev)
        true_class_probability.append(true_class_probability_sev)
        max_class_probability.append(max_class_probability_sev)
        is_ood.append(is_ood_sev)
        acc_per_model.append(acc_per_model_sev)
        ensemble_acc.append(ensemble_acc_sev)
        acc_per_model_id.append(acc_per_model_id_sev)
        ensemble_acc_id.append(ensemble_acc_id_sev)
    
    figures_path = f"{base_results_path}/figures/"
    os.makedirs(figures_path, exist_ok=True)
    # plot_binned_averages_icml([(aleatoric[j][1], epistemic[j][1], f"Fraction of OOD {int(frac*100)}%") for j, frac in enumerate(config["ood_fractions"])], num_bins=20, x_label="Aleatoric uncertainty (binned)", y_label="Epistemic uncertainty", save_path=figures_path, title="CIFAR10C Severity 2")
    for f, frac in enumerate(config["ood_fractions"]):
        figures_path_frac = f"{base_results_path}/frac_{frac}/figures"
        os.makedirs(figures_path_frac, exist_ok=True)
        plot_binned_averages_icml([(aleatoric[f][j], epistemic[f][j], f"Severity {int(severity)}") for j, severity in enumerate(config["cifar10c_severities"])], num_bins=20, x_label="Aleatoric uncertainty (binned)", y_label="Epistemic uncertainty", save_path=figures_path_frac, title=f"Fraction {int(frac*100)}%")
    for s, sev in enumerate(config["cifar10c_severities"]):
        figures_path_sev = f"{base_results_path}/figures"
        os.makedirs(figures_path_sev, exist_ok=True)
        plot_binned_averages_icml([(aleatoric[f][s], epistemic[f][s], f"Fraction of OOD {int(frac*100)}%") for f, frac in enumerate(config["ood_fractions"])], num_bins=20, x_label="Aleatoric uncertainty (binned)", y_label="Epistemic uncertainty", save_path=figures_path_sev, title=f"Severity {int(sev)}")