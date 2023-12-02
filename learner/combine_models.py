from util.save_load import load_kernel_model, load_kernel_model_and_setup
from kernels.wrapper import KernelModelWrapper

domain = "spanner"
df = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
pf = f"../benchmarks/ipc2023-learning-benchmarks/{domain}/training/easy/p01.pddl"

m1 = "icaps24_wl_models/spanner_ilg_1wl_4_0_linear-svr_none_H.pkl"
m2 = "icaps24_wl_models/spanner_ilg_1wl_4_0_mip_schema_H.pkl"
target = "spanner_combined.pkl"

m1 : KernelModelWrapper = load_kernel_model_and_setup(m1, df, pf)

m1.combine_with_other_models(target, [m2])

m1.write_model_data()
print(m1.get_model_data_path())
