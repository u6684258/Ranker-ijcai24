from argparse import Namespace
import os
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS as DOMAINS
from util.save_load import save_kernel_model, load_kernel_model_and_setup
from kernels.wrapper import KernelModelWrapper

# SVR
for domain in DOMAINS:
    df = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
    pf = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training/easy/p01.pddl"

    m1 = f"icaps24_wl_models/{domain}_ilg2_1wl_4_0_linear-svr_none_H.pkl"
    m2 = f"icaps24_wl_models/{domain}_ilg2_1wl_4_0_mip_schema_H.pkl"
    target = f"icaps24_wl_models/{domain}_combined_svr.pkl"

    m1 : KernelModelWrapper = load_kernel_model_and_setup(m1, df, pf)

    m1.combine_with_other_models([m2])

    args = Namespace()
    args.model_save_file = target
    save_kernel_model(m1, args)

# GP
for domain in DOMAINS:
    df = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
    pf = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training/easy/p01.pddl"

    m1 = f"icaps24_wl_models/{domain}_ilg2_1wl_4_0_gp_none_H.pkl"
    m2 = f"icaps24_wl_models/{domain}_ilg2_1wl_4_0_mip_schema_H.pkl"
    target = f"icaps24_wl_models/{domain}_combined_gp.pkl"

    m1 : KernelModelWrapper = load_kernel_model_and_setup(m1, df, pf)

    m1.combine_with_other_models([m2])

    args = Namespace()
    args.model_save_file = target
    save_kernel_model(m1, args)

print("All models combined successfully!")

print("learned mip schemata")
for domain in DOMAINS:
    train_log_file = f"icaps24_train_logs/{domain}_ilg2_1wl_4_0_mip_schema_H.log"
    assert os.path.exists(train_log_file), train_log_file
    with open(train_log_file, "r") as file:
        for line in file.readlines():
            if "schemata to keep" in line:
                to_keep = int(line.split()[0])
                print(domain, to_keep)
