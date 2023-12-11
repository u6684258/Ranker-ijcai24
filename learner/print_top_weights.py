from collections import OrderedDict
import os
import shutil
import networkx as nx
from typing import List
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from dataset.ipc2023_learning_domain_info import IPC2023_LEARNING_DOMAINS
from kernels.wrapper import KernelModelWrapper
from util.save_load import load_kernel_model

_TOP_K = 20
_TOP_L = 3
_ACC = f".6f"
_DOMAIN = "blocksworld"


@dataclass
class Node:
    name: str
    defn: str


def model_path(domain):
    return f"icaps24_wl_models/{domain}_ilg2_1wl_4_0_gp_none_H.pkl"


def main():
    PLOT_DIR = "plots_weights"
    shutil.rmtree(PLOT_DIR)
    os.makedirs(PLOT_DIR, exist_ok=True)

    for domain in IPC2023_LEARNING_DOMAINS:
        model: KernelModelWrapper = load_kernel_model(model_path(domain))

        weights = model.get_weights()
        abs_weights = abs(weights)
        wl_hash = model.get_hash()
        reverse_hash = model.get_reverse_hash()
        debug_map = model.debug_colour_information()

        nodes_set = {}
        children_graph = OrderedDict()
        parents_graph = OrderedDict()

        for i in range(len(weights)):
            if i in nodes_set:
                continue
            defn = reverse_hash[i]
            if "," not in defn and wl_hash[defn] in debug_map:
                defn = debug_map[wl_hash[defn]]
                parents = []
            else:
                defn = defn.split(",")
                parents = [defn[0]] + [defn[j] for j in range(1, len(defn), 2)]

                ret = defn[0] + " "
                for j in range(1, len(defn)):
                    ret += defn[j]
                    if j % 2 == 1:
                        ret += "_"
                    else:
                        ret += " "
                defn = ret
            node = f"{i:>5} = {defn}"
            # print(node)
            nodes_set[i] = node

            for parent in parents:
                parent = int(parent)
                if parent not in children_graph:
                    children_graph[parent] = set()
                children_graph[parent].add(i)
                parents_graph[i] = parents

        # G = nx.DiGraph()

        # for parent, children_set in children_graph.items():
        #     if parent >= 10:
        #         continue
        #     for child in children_set:
        #         G.add_edge(nodes_set[parent], nodes_set[child])
        #     # print(parent, children_set)

        # # print(G.nodes)
        # # print(G.edges)

        # # nx.draw(G, with_labels=True)
        # # plt.show()
        # nx.drawing.nx_pydot.write_dot(G, f"{domain}.dot")

        # breakpoint()

        sorted_indices = np.argsort(abs_weights)[::-1]
        top_indices = sorted_indices[:_TOP_K]
        top_weights = weights[top_indices]

        print(domain)
        print("num weights:", len(weights))
        print("top indices:", top_indices)
        print("top weights:", top_weights)
        sames = {}
        expls = {}
        expl_to_index = {}
        for rank, index in enumerate(sorted_indices):
            w = f"{weights[index]:{_ACC}}"
            k = reverse_hash[index]
            if "," not in k:
                k = wl_hash[k]
            k = str(k)
            # k = k.replace(",", " ")
            # print(f"{w:>5}  {index:>5} {k}")

            if w not in sames:
                sames[w] = 0
                expls[w] = set()
            sames[w] += 1
            expls[w].add(k)
            expl_to_index[k] = index

            if rank >= _TOP_K:
                continue

            print(f"{w:>5}  {index:>5} {k}")

        for k, v in debug_map.items():
            print(f"{k} -> {v}")

        summs = []
        summs_signed = []
        mapp = {}
        for w, cnt in sames.items():
            combined = float(w) * cnt
            summs_signed.append(combined)
            combined = abs(combined)
            summs.append(combined)
            summs.append(combined)
            mapp[combined] = w

        summs = sorted(summs, reverse=True)
        print("top summed")
        for i in range(_TOP_L):
            w = mapp[summs[i]]
            print(f"{summs[i]:{_ACC}} & {w} & ")
            # print(f"  {len(expls[w])}")
            # print(f"  {sorted(expls[w])}")
            print(f"  {sorted([expl_to_index[expl] for expl in expls[w]])}")
            for expl in expls[w]:
                G = nx.DiGraph()
                init_index = expl_to_index[expl]
                toks = expl.split(",")
                tuples = [f"({toks[i]},{toks[i+1]})" for i in range(1, len(toks), 2)]
                to_tex = r"(" + toks[0] + r",\mset{" + ",".join(tuples) + r"})"
                print(f" & \\multicolumn{{2}}{{l}}${init_index} = \\hash{to_tex}$ \\\\")
                q = [init_index]
                while len(q) > 0:
                    index = int(q.pop(0))
                    node = nodes_set[index]
                    if node not in G.nodes:
                        G.add_node(node)
                    if index not in parents_graph:
                        continue
                    for parent in parents_graph[index]:
                        parent_node = nodes_set[int(parent)]
                        # if parent_node in G.nodes:
                        #     continue
                        G.add_edge(parent_node, node)
                        q.append(parent)

                graph_save_dir = f"{PLOT_DIR}/{domain}/{w}"
                os.makedirs(graph_save_dir, exist_ok=True)
                nx.drawing.nx_pydot.write_dot(G, f"{graph_save_dir}/{init_index}  {expl}.dot")

        bin_edges = np.linspace(-max(summs), max(summs), 10)
        plt.hist(summs_signed, density=False, bins=bin_edges, edgecolor="black")
        plt.title(domain + " " + str(len(summs_signed)))
        plt.yscale("log")
        plt.xlabel("Weight Value")
        plt.ylabel("Occurences")
        plt.savefig(f"{PLOT_DIR}/freq_{domain}.png", dpi=360)
        plt.clf()
        plt.hist(summs_signed, density=True, bins=bin_edges, edgecolor="black")
        plt.title(domain + " " + str(len(summs_signed)))
        # plt.yscale("log")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.savefig(f"{PLOT_DIR}/dens_{domain}.png", dpi=360)
        plt.clf()
        print()

        # breakpoint()
        # break


if __name__ == "__main__":
    main()
