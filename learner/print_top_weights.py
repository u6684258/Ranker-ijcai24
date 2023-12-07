import os
from matplotlib import pyplot as plt
import numpy as np
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS
from kernels.wrapper import KernelModelWrapper
from util.save_load import load_kernel_model

def model_path(domain):
    return f"icaps24_wl_models/{domain}_ilg_1wl_4_0_gp_none_H.pkl"

_TOP_K = 5

PLOT_DIR = "plots_weights"
os.makedirs(PLOT_DIR, exist_ok=True)

for domain in IPC2023_LEARNING_DOMAINS:
    model : KernelModelWrapper = load_kernel_model(model_path(domain))

    weights = model.get_weights()
    abs_weights = abs(weights)
    wl_hash = model.get_hash()
    reverse_hash = model.get_reverse_hash()

    sorted_indices = np.argsort(abs_weights)[::-1]
    top_indices = sorted_indices[:_TOP_K]
    top_weights = weights[top_indices]

    # plt.hist(weights, bins=30, density=True)
    # plt.title(domain)
    # plt.yscale("log")
    # plt.xlabel('Weight Value')
    # plt.ylabel('Density')
    # plt.savefig(f"{PLOT_DIR}/{domain}.png", dpi=360)
    # plt.clf()

    print(domain)
    print("num weights:", len(weights))
    print("top indices:", top_indices)
    print("top weights:", top_weights)
    for index in top_indices:
        w = f"{weights[index]:.2f}"
        print(f"{index:>5} {w:>5}  {reverse_hash[index]}")

    model.debug_colour_information()

    breakpoint()
