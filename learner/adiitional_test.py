import torch

from util.pyperplan_api import STRIPSProblem
from dataset.graphs_ranker import get_state_data
import matplotlib.pyplot as plt
import numpy as np
from representation import REPRESENTATIONS
from util.save_load import load_gnn_model_and_setup, load_gnn_model

base_dir = "../benchmarks/goose"
domain="blocks"
test_file = "train"
model_file = f"trained_models_gnn/rank-{domain}-L4-coord.dt"
df = f"{base_dir}/{domain}/domain.pddl"
pf = f"{base_dir}/{domain}/{test_file}/blocks10-task01.pddl"
sf = f"{base_dir}/{domain}/{test_file}_solution/blocks10-task01_1800.out"
# pf = f"{base_dir}/{domain}/{test_file}/p-l10-c10-s2.pddl"

last_x = None
hs = None
data_list = None
hs1_all = None
hs2_all = None
hs_all = None
weight = None
# for i in range(5):

problem = STRIPSProblem(df, pf, sf)
# problem.state_to_heuristic

# model, _ = load_gnn_model(model_file, ignore_subdir=True)
model = load_gnn_model_and_setup(model_file, df, pf)
# model.eval()

# datalist = generate_graph_from_domain_problem_pddl("goose-blocks", df, pf, None, "llg")
states = get_state_data(f"goose-{domain}", df, pf, None, 5)

rep = model.rep
states_only = [rep.str_to_state(state[0]) for state in states]
plan_states_only = [(rep.str_to_state(state[0]), state[0]) for state in states if state[1][1]==0]
states_real_heu = [problem.get_state_heuristic(state[0]) for state in states if state[1][1] == 0]
    # if last_x is None:
    #     last_x = model.rep.state_to_tensor(states_only[0])[0].numpy()
    #     hs, data_list, hs1_all, hs2_all, hs_all = model.h_batch(states_only)
    #     weight = model.model.ranker.weight
    # else:
    #     cur_x = model.rep.state_to_tensor(states_only[0])[0].numpy()
    #     assert (last_x == cur_x).all()
    #     last_x = cur_x
    #     cur_weight = model.model.ranker.weight
    #     assert (weight == cur_weight).all(), weight - cur_weight
    #     weight = model.model.ranker.weight
    #     hsc, data_listc, hs1_allc, hs2_allc, hs_allc = model.h_batch(states_only)
    #     # assert (np.array(hsc) == np.array(hs)).all()
    #     # assert (np.array(data_listc) == np.array(data_list)).all()
    #     assert ((np.array(hs1_allc) - np.array(hs1_all))<1e-3).all()
    #     assert ((np.array(hs2_allc) - np.array(hs2_all))<1e-3).all(), (np.array(hs2_allc) == np.array(hs2_all))<1e-3
    #     assert ((np.array(hs_allc) - np.array(hs_all))<1e-3).all()
    #     hs = hsc
    #     data_list = data_listc
    #     hs1_all = hs1_allc
    #     hs2_all = hs2_allc
    #     hs_all = hs_allc
    # print(list(model.rep.state_to_tensor(states_only[0])[0].numpy()))

predictions = model.h_batch(states_only)


coord_to_heu = np.array([(state[1][0], state[1][1],predictions[i]) for i, state in enumerate(states)])

accuracy = {}
accuracy_on_plan = {}

for i in range(max(coord_to_heu[:, 0])+1):
    slice = coord_to_heu[coord_to_heu[:, 0] == i]

    accuracy[states_real_heu[i]] = np.sum((slice[0, 2] - slice[:, 2])[1:] < 0)/(slice.shape[0]-1)

    accuracy_on_plan[states_real_heu[i]] = np.sum((slice[0, 2] - slice[1, 2]) < 0)

print(sum(accuracy.values())/len(accuracy))
print(accuracy_on_plan)


plt.bar(accuracy.keys(), accuracy.values())
plt.show()
print(sum(accuracy_on_plan.values())/len(accuracy_on_plan))
plt.bar(accuracy_on_plan.keys(), accuracy_on_plan.values())
plt.show()