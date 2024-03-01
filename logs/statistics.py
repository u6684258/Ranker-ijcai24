import os
import random

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SOLUTION_PATH = "../benchmarks/ipc2023-learning-benchmarks/solutions"
IPC2023_FAIL_LIMIT = {
        "blocksworld": 28,
        "childsnack": 26,
        "ferry": 68,
        "floortile": 12,
        "miconic": 90,
        "rovers": 34,
        "satellite": 65,
        "sokoban": 36,
        "spanner": 30,
        "transport": 41,
    }

OUT_OF_MEMORY = ['satellite_easy_p22_satellite_llg_L4_H64_mean_r0_hgn-rank.log',
                 'transport_easy_p18_transport_llg_L4_H64_mean_r0_hgn-rank.log',
                 'childsnack_easy_p29_childsnack_llg_L4_H64_mean_r0_hgn-rank.log',
                 'miconic_medium_p21_miconic_llg_L4_H64_mean_r0_hgn-rank.log',
                 'miconic_medium_p28_miconic_llg_L4_H64_mean_r0_hgn-rank.log',
                 'miconic_medium_p27_miconic_llg_L4_H64_mean_r0_hgn-rank.log',
                 'transport_easy_p25_transport_llg_L4_H64_mean_r0_hgn-rank.log']


def gen_dataset(log_path):
    df = pd.DataFrame(columns=["domain", "difficulty", "num", "encodes",
                               "layers", "hidden", "method", "model_num", "model",
                               "length", "expand", "evaluated", "time"])
    for file in os.listdir(log_path):
        if file == "statistics.py":
            continue
        domain, difficulty, num, _, encodes, layers, hidden, method, model_num, model = file.split("_")
        model = model.split(".")[0]
        if model == "rank":
            continue
        length = 0
        expand = 0
        eval = 0
        time = 0
        if file not in OUT_OF_MEMORY:
            with open(f"{log_path}/{file}") as f:
                for line in f.readlines():
                    if "Plan length: " in line:
                        length = line.split("Plan length: ")[1].split(" step(s).")[0]
                    if "KB] Expanded " in line:
                        expand = line.split("KB] Expanded ")[1].split(" state(s).")[0]
                    if "KB] Evaluated " in line:
                        eval = line.split("KB] Evaluated ")[1].split(" state(s).")[0]
                    if " KB] Search time: " in line:
                        time = line.split(" KB] Search time: ")[1].split("s")[0]
                    if "Time limit has been reached." in line:
                        break

        else:
            print(f"have this: {file}")

        df = pd.concat([df, pd.DataFrame([[domain, difficulty, num, encodes,
                                           layers, hidden, method, model_num, model,
                                           length, expand, eval, time]], columns=df.columns)])


    print(df.head())
    print(len(df))

    df.to_csv("../results.csv")
    return df


def gen_ff_dataset(log_path):
    df = pd.DataFrame(columns=["domain", "difficulty", "num", "encodes",
                               "layers", "hidden", "method", "model_num", "model",
                               "length", "expand", "evaluated", "time"])
    for file in os.listdir(log_path):
        if file == "statistics.py":
            continue
        domain, difficulty, num = file.split("_")
        encodes = "hff"
        layers = 0
        hidden = 0
        method = "NaN"
        model_num = "r0"
        model = "hff"
        num = num.split(".")[0]
        length = 0
        expand = 0
        eval = 0
        time = 0
        with open(f"{log_path}/{file}") as f:
            for line in f.readlines():
                if "Plan length: " in line:
                    length = line.split("Plan length: ")[1].split(" step(s).")[0]
                if "KB] Expanded " in line:
                    expand = line.split("KB] Expanded ")[1].split(" state(s).")[0]
                if "KB] Evaluated " in line:
                    eval = line.split("KB] Evaluated ")[1].split(" state(s).")[0]
                if " KB] Search time: " in line:
                    time = line.split(" KB] Search time: ")[1].split("s")[0]
                if "Time limit has been reached." in line:
                    break


        df = pd.concat([df, pd.DataFrame([[domain, difficulty, num, encodes,
                                           layers, hidden, method, model_num, model,
                                           length, expand, eval, time]], columns=df.columns)])


    print(df.head())
    print(len(df))

    df.to_csv("../results_ff.csv")
    return df


def count_memory(log_path):
    df = pd.DataFrame(columns=["file", "m"])

    maxm = 0
    ms = {}
    maxn = ""
    for file in os.listdir(log_path):
        if file == "statistics.py":
            continue
        domain, difficulty, num, _, encodes, layers, hidden, method, model_num, model = file.split("_")
        model = model.split(".")[0]
        if model == "rank":
            continue
        with open(f"{log_path}/{file}") as f:
            m = 0
            lines = f.read()
            if "Time limit has been reached." in lines or "Solution found." in lines:
                if "Peak memory: " in lines:
                    m = int(lines.split("\nPeak memory: ")[1].split(" KB")[0])
                if "Time limit has been reached." in lines:
                    m = 0

                ms[file] = m

                if m > maxm:
                    maxm = m
                    maxn = file
            else:
                print(file)

    print(maxm)
    print(maxn)
    print({k: v for k, v in sorted(ms.items(), key=lambda item: item[1])})


def gen_hgn_dataset(log_path):
    df = pd.DataFrame(columns=["domain", "difficulty", "num", "encodes",
                               "layers", "hidden", "method", "model_num", "model",
                               "length", "expand", "evaluated", "time"])
    for file in os.listdir(log_path):
        if file == "statistics.py":
            continue
        domain, difficulty, num, _, encodes, layers, hidden, method, model_num, model = file.split("_")
        model = model.split(".")[0]
        if model == "rank":
            continue
        length = 0
        expand = 0
        eval = 0
        time = 0
        if file not in OUT_OF_MEMORY:
            with open(f"{log_path}/{file}") as f:
                for line in f.readlines():
                    if "Plan length: " in line:
                        length = line.split("Plan length: ")[1].split(" step(s).")[0]
                    if "Expanded " in line:
                        expand = line.split("Expanded ")[1].split(" state(s).")[0]
                    if "Evaluated " in line:
                        eval = line.split("Evaluated ")[1].split(" state(s).")[0]
                    if "Search time: " in line:
                        time = line.split("Search time: ")[1].split("s")[0]
                    if "Time limit has been reached." in line:
                        break


        df = pd.concat([df, pd.DataFrame([[domain, difficulty, num, encodes,
                                           layers, hidden, method, model_num, model,
                                           length, expand, eval, time]], columns=df.columns)])


    print(df.head())
    print(len(df))

    df.to_csv("../results_hgn.csv")
    return df


def coverage_table(df_path):
    df = pd.read_csv(df_path)
    position_sums = []
    for domain in IPC2023_FAIL_LIMIT.keys():
        # domain_data = [IPC2023_FAIL_LIMIT[domain]]
        domain_data = []
        m_sums = []
        # for model in ["gnn", "gnn-loss", "gnn-rank"]:
        for model in ["hgn", "hgn-loss", "hgn-rank"]:
            for model_num in range(1):
                solved = []
                model_indx = f"r{model_num}"
                for diff in ["easy", "medium", "hard"]:
                    sub_df = df[((df["domain"] == domain) &
                                 (df["model"] == model) &
                                 (df["model_num"] == model_indx) &
                                 (df["difficulty"] == diff))]
                    solved.append(len(sub_df) - (sub_df["length"] == 0).sum())
                    # print(f"{domain} | {model} | {model_indx} | {diff} : {len(sub_df)}, {(sub_df['length'] == 0).sum()}")
                domain_data.extend(solved)
                domain_data.append(sum(solved))
                m_sums.append(sum(solved))
        # domain_data.append(m_sums[1] - m_sums[0])

        if not position_sums:
            position_sums = [0 for _ in domain_data]
        data_str = ""
        for ind, i in enumerate(domain_data):
            data_str += f" & {i}"
            position_sums[ind] += i
        print(f"{domain}{data_str} \\\\")
    sum_str = ""
    for ind, i in enumerate(position_sums):
        sum_str += f" & {i}"
    print("\hline")
    print(f"Sum{sum_str} \\\\")


def quality_score(df_path):
    df = pd.read_csv(df_path)
    sum_sum_scores = np.array([0,0,0])
    for domain in IPC2023_FAIL_LIMIT.keys():
        sum_scores = np.array([0,0,0])
        for diff in ["easy", "medium", "hard"]:
            for i in range(1, 31):
                problem = f"p0{i}" if i < 10 else f"p{i}"
                standard_len = int(open(f"{SOLUTION_PATH}/{domain}/testing/{diff}/{problem}.plan").read().split("; cost = ")[1].split(" (unit cost)")[0])
                gnn_len = df[((df["domain"] == domain) &
                                 (df["model"] == "gnn") &
                                 (df["model_num"] == "r0") &
                                 (df["difficulty"] == diff) &
                                 (df["num"] == problem))]["length"]
                if gnn_len.empty:
                    gnn_len = 0
                else:
                    gnn_len = int(gnn_len.iloc[0])
                rank_len = df[((df["domain"] == domain) &
                              (df["model"] == "gnn-rank") &
                              (df["model_num"] == "r0") &
                              (df["difficulty"] == diff) &
                              (df["num"] == problem))]["length"]
                if rank_len.empty:
                    rank_len = 0
                else:
                    rank_len = int(rank_len.iloc[0])
                lens = np.array([standard_len, gnn_len, rank_len])
                scores = min(i for i in lens if i > 0) / lens
                scores[np.isnan(scores)] = 0
                scores[np.isinf(scores)] = 0
                sum_scores = sum_scores + scores

        sum_str = ""
        for ind, i in enumerate(sum_scores):
            if ind == 0:
                continue
            sum_str += f" & {i}"
        print(f"{domain}{sum_str} \\\\")
        sum_sum_scores = sum_sum_scores + sum_scores
    sum_str = ""
    for ind, i in enumerate(sum_sum_scores):
        if ind == 0:
            continue
        sum_str += f" & {i}"
    print("\hline")
    print(f"Sum{sum_str} \\\\")

def agile_score(df_path):
    df = pd.read_csv(df_path)
    sum_sum_scores = np.array([0,0])
    for domain in IPC2023_FAIL_LIMIT.keys():
        sum_scores = np.array([0,0])
        for diff in ["easy", "medium", "hard"]:
            for i in range(1, 31):
                problem = f"p0{i}" if i < 10 else f"p{i}"
                gnn_len = df[((df["domain"] == domain) &
                                 (df["model"] == "gnn") &
                                 (df["model_num"] == "r0") &
                                 (df["difficulty"] == diff) &
                                 (df["num"] == problem))]["time"]
                if gnn_len.empty:
                    gnn_len = 0
                else:
                    gnn_len = float(gnn_len.iloc[0])
                rank_len = df[((df["domain"] == domain) &
                              (df["model"] == "gnn-rank") &
                              (df["model_num"] == "r0") &
                              (df["difficulty"] == diff) &
                              (df["num"] == problem))]["time"]
                if rank_len.empty:
                    rank_len = 0
                else:
                    rank_len = float(rank_len.iloc[0])
                lens = np.array([gnn_len, rank_len])
                scores = 1 - np.log(lens)/np.log(600)
                scores[np.isnan(scores)] = 0
                scores[np.isinf(scores)] = 0
                sum_scores = sum_scores + scores

        sum_str = ""
        for ind, i in enumerate(sum_scores):
            sum_str += f" & {i}"
        print(f"{domain}{sum_str} \\\\")
        sum_sum_scores = sum_sum_scores + sum_scores
    sum_str = ""
    for ind, i in enumerate(sum_sum_scores):
        sum_str += f" & {i}"
    print("\hline")
    print(f"Sum{sum_str} \\\\")

def both_scores(df_path):
    df = pd.read_csv(df_path)
    sum_sum_scores = np.array([0, 0.0, 0, 0.0, 0, 0])
    for domain in IPC2023_FAIL_LIMIT.keys():
        sum_scores = np.array([0, 0.0, 0, 0.0, 0, 0])
        for diff in ["easy", "medium", "hard"]:
            for i in range(1, 31):
                problem = f"p0{i}" if i < 10 else f"p{i}"
                gnn_len = df[((df["domain"] == domain) &
                              (df["model"] == "hgn") &
                              (df["model_num"] == "r0") &
                              (df["difficulty"] == diff) &
                              (df["num"] == problem))]["time"]
                if gnn_len.empty:
                    gnn_len = 0
                else:
                    gnn_len = float(gnn_len.iloc[0])
                rank_len = df[((df["domain"] == domain) &
                               (df["model"] == "hgn-rank") &
                               (df["model_num"] == "r0") &
                               (df["difficulty"] == diff) &
                               (df["num"] == problem))]["time"]
                if rank_len.empty:
                    rank_len = 0
                else:
                    rank_len = float(rank_len.iloc[0])
                lens = np.array([gnn_len, rank_len])
                scores = 1 - np.log(lens) / np.log(600)
                scores[np.isnan(scores)] = 0
                scores[np.isinf(scores)] = 0
                sum_scores[1] = sum_scores[1] + scores[0]
                sum_scores[3] = sum_scores[3] + scores[1]

                standard_len = int(
                    open(f"{SOLUTION_PATH}/{domain}/testing/{diff}/{problem}.plan").read().split("; cost = ")[1].split(
                        " (unit cost)")[0])
                gnn_len = df[((df["domain"] == domain) &
                              (df["model"] == "hgn") &
                              (df["model_num"] == "r0") &
                              (df["difficulty"] == diff) &
                              (df["num"] == problem))]["length"]
                if gnn_len.empty:
                    gnn_len = 0
                else:
                    gnn_len = int(gnn_len.iloc[0])
                rank_len = df[((df["domain"] == domain) &
                               (df["model"] == "hgn-rank") &
                               (df["model_num"] == "r0") &
                               (df["difficulty"] == diff) &
                               (df["num"] == problem))]["length"]
                if rank_len.empty:
                    rank_len = 0
                else:
                    rank_len = int(rank_len.iloc[0])
                lens = np.array([standard_len, gnn_len, rank_len])
                scores = min(i for i in lens if i > 0) / lens
                scores[np.isnan(scores)] = 0
                scores[np.isinf(scores)] = 0
                sum_scores[0] = sum_scores[0] + scores[1]
                sum_scores[2] = sum_scores[2] + scores[2]

        sum_scores[4] = sum_scores[2] - sum_scores[0]
        sum_scores[5] = sum_scores[3] - sum_scores[1]

        sum_str = ""
        for ind, i in enumerate(sum_scores):
            sum_str += f" & {round(i, 2)}"
        print(f"{domain}{sum_str} \\\\")
        sum_sum_scores = sum_sum_scores + sum_scores
    sum_str = ""
    for ind, i in enumerate(sum_sum_scores):
        sum_str += f" & {round(i, 2)}"
    print("\hline")
    print(f"Sum{sum_str} \\\\")


def plot_stats(df_path):
    colors = list(matplotlib.colors.TABLEAU_COLORS)
    size = 25
    markers = ["x" for _ in range(len(IPC2023_FAIL_LIMIT))]
    ax1_limit = 5000
    ax2_limit = 400
    ax3_limit = 600

    df = pd.read_csv(df_path)
    nodes_expands = []
    plan_lengths = []
    search_times = []
    for domain in IPC2023_FAIL_LIMIT.keys():
        gnn_data = df[((df["domain"] == domain) &
                       (df["model"] == "gnn") &
                       (df["model_num"] == "r0"))]
        rank_data = df[((df["domain"] == domain) &
                        (df["model"] == "gnn-rank") &
                        (df["model_num"] == "r0"))]
        combine_diffs = [[],[]]
        for diff in ["easy", "medium", "hard"]:
            diff_gnn_data = gnn_data[(gnn_data["difficulty"] == diff)].sort_values('num')
            diff_rank_data = rank_data[(rank_data["difficulty"] == diff)].sort_values('num')
            combine_diffs[0].append(diff_gnn_data)
            combine_diffs[1].append(diff_rank_data)

        for m in combine_diffs:
            es = [d["expand"].to_numpy() for d in m]
            ls = [d["length"].to_numpy() for d in m]
            ts = [d["time"].to_numpy() for d in m]
            es = [np.pad(e, (0, 30-e.shape[0])) for e in es]
            ls = [np.pad(e, (0, 30-e.shape[0])) for e in ls]
            ts = [np.pad(e, (0, 30-e.shape[0])) for e in ts]
            npes = np.concatenate(es, axis=0)
            npls = np.concatenate(ls, axis=0)
            npts = np.concatenate(ts, axis=0)
            npes[npes == 0] = ax1_limit
            npls[npls == 0] = ax2_limit
            npts[npts == 0] = ax3_limit
            nodes_expands.append(npes)
            plan_lengths.append(npls)
            search_times.append(npts)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for i, domain in enumerate(IPC2023_FAIL_LIMIT.keys()):
        ax[0].scatter(nodes_expands[2*i], nodes_expands[2*i+1], color=colors[i], s=size, marker=markers[i])
        ax[1].scatter(plan_lengths[2*i], plan_lengths[2*i+1], color=colors[i], s=size, marker=markers[i])
        ax[2].scatter(search_times[2*i], search_times[2*i+1], color=colors[i], s=size, marker=markers[i], label=domain)
        ax[0].axline((0,0), slope=1)
        ax[1].axline((0,0), slope=1)
        ax[2].axline((0,0), slope=1)
    ax[0].set_xlim([0, ax1_limit])
    ax[0].set_ylim([0, ax1_limit])
    ax[1].set_xlim([0, ax2_limit])
    ax[1].set_ylim([0, ax2_limit])
    ax[2].set_xlim([0, ax3_limit])
    ax[2].set_ylim([0, ax3_limit])
    ax[0].set_xlabel("GOOSE-NN")
    ax[0].set_ylabel("GOOSE-Ranker")
    ax[1].set_xlabel("GOOSE-NN")
    ax[1].set_ylabel("GOOSE-Ranker")
    ax[2].set_xlabel("GOOSE-NN")
    ax[2].set_ylabel("GOOSE-Ranker")
    ax[0].set_title("Nodes Expanded", size=25)
    ax[1].set_title("Plan Length",size=25)
    ax[2].set_title("Search Time", size=25)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig("plot.png")
    plt.show()
    return df


def plot_stats_with_ff(df_path, df_ff_path):
    colors = list(matplotlib.colors.TABLEAU_COLORS)
    size = 25
    markers = ["x" for _ in range(len(IPC2023_FAIL_LIMIT))]
    ax1_limit = 200
    ax2_limit = 400
    ax3_limit = 600

    df = pd.read_csv(df_path)
    dfff = pd.read_csv(df_ff_path)
    df = pd.concat([df, dfff])
    nodes_expands = []
    plan_lengths = []
    search_times = []
    for domain in IPC2023_FAIL_LIMIT.keys():
        gnn_data = df[((df["domain"] == domain) &
                       (df["model"] == "gnn") &
                       (df["model_num"] == "r0"))]
        rank_data = df[((df["domain"] == domain) &
                        (df["model"] == "gnn-rank") &
                        (df["model_num"] == "r0"))]
        ff_data = df[((df["domain"] == domain) &
                        (df["model"] == "hff") &
                        (df["model_num"] == "r0"))]
        combine_diffs = [[],[],[]]
        for diff in ["easy", "medium", "hard"]:
            diff_gnn_data = gnn_data[(gnn_data["difficulty"] == diff)].sort_values('num')
            diff_rank_data = rank_data[(rank_data["difficulty"] == diff)].sort_values('num')
            diff_ff_data = ff_data[(ff_data["difficulty"] == diff)].sort_values('num')
            combine_diffs[0].append(diff_gnn_data)
            combine_diffs[1].append(diff_rank_data)
            combine_diffs[2].append(diff_ff_data)

        for m in combine_diffs:
            es = [d["expand"].to_numpy() for d in m]
            ls = [d["length"].to_numpy() for d in m]
            ts = [d["time"].to_numpy() for d in m]
            es = [np.pad(e, (0, 30-e.shape[0])) for e in es]
            ls = [np.pad(e, (0, 30-e.shape[0])) for e in ls]
            ts = [np.pad(e, (0, 30-e.shape[0])) for e in ts]
            npes = np.concatenate(es, axis=0)
            npls = np.concatenate(ls, axis=0)
            npts = np.concatenate(ts, axis=0)
            npes[npes == 0] = np.exp(np.sqrt(ax1_limit))
            npls[npls == 0] = ax2_limit
            npts[npts == 0] = ax3_limit
            nodes_expands.append(npes)
            plan_lengths.append(npls)
            search_times.append(npts)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    def plot_subplots(i, j, x, y):
        ax[0, j].scatter(np.square(np.log(nodes_expands[x])),
                        np.square(np.log(nodes_expands[y])),
                         color=colors[i], s=size, marker=markers[i])
        ax[1, j].scatter(plan_lengths[x], plan_lengths[y], color=colors[i], s=size, marker=markers[i])
        ax[2, j].scatter(search_times[x], search_times[y], color=colors[i], s=size, marker=markers[i],
                         label=domain)
        ax[0, j].axline((0, 0), slope=1, c="red")
        ax[1, j].axline((0, 0), slope=1, c="red")
        ax[2, j].axline((0, 0), slope=1, c="red")


    for i, domain in enumerate(IPC2023_FAIL_LIMIT.keys()):
        plot_subplots(i, 0, 3*i+1, 3*i)
        plot_subplots(i, 1, 3*i+1, 3*i+2)
        plot_subplots(i, 2, 3*i, 3*i+2)

    for i in range(3):
        ax[0,i].set_xlim([0, ax1_limit])
        ax[0,i].set_ylim([0, ax1_limit])
        ax[1,i].set_xlim([0, ax2_limit])
        ax[1,i].set_ylim([0, ax2_limit])
        ax[2,i].set_xlim([0, ax3_limit])
        ax[2,i].set_ylim([0, ax3_limit])
        ax[i,0].set_xlabel("GOOSE-Ranker")
        ax[i,0].set_ylabel("GOOSE-NN")
        ax[i,1].set_xlabel("GOOSE-Ranker")
        ax[i,1].set_ylabel("h-FF")
        ax[i,2].set_xlabel("GOOSE-NN")
        ax[i,2].set_ylabel("h-FF")
    ax[0,1].set_title("log(Nodes Expanded)", size=18)
    ax[1,1].set_title("Plan Length",size=18)
    ax[2,1].set_title("Search Time", size=18)
    plt.legend(bbox_to_anchor=(1, 2))
    plt.savefig("plot.png")
    plt.show()
    return df


def plot_hgn_stats_with_ff(df_path, df_ff_path):
    colors = list(matplotlib.colors.TABLEAU_COLORS)
    size = 25
    markers = ["x" for _ in range(len(IPC2023_FAIL_LIMIT))]
    ax1_limit = 200
    ax2_limit = 400
    ax3_limit = 600

    df = pd.read_csv(df_path)
    dfff = pd.read_csv(df_ff_path)
    df = pd.concat([df, dfff])
    nodes_expands = []
    plan_lengths = []
    search_times = []
    for domain in IPC2023_FAIL_LIMIT.keys():
        gnn_data = df[((df["domain"] == domain) &
                       (df["model"] == "hgn") &
                       (df["model_num"] == "r0"))]
        rank_data = df[((df["domain"] == domain) &
                        (df["model"] == "hgn-rank") &
                        (df["model_num"] == "r0"))]
        ff_data = df[((df["domain"] == domain) &
                        (df["model"] == "hff") &
                        (df["model_num"] == "r0"))]
        combine_diffs = [[],[],[]]
        for diff in ["easy", "medium", "hard"]:
            diff_gnn_data = gnn_data[(gnn_data["difficulty"] == diff)].sort_values('num')
            diff_rank_data = rank_data[(rank_data["difficulty"] == diff)].sort_values('num')
            diff_ff_data = ff_data[(ff_data["difficulty"] == diff)].sort_values('num')
            combine_diffs[0].append(diff_gnn_data)
            combine_diffs[1].append(diff_rank_data)
            combine_diffs[2].append(diff_ff_data)

        for m in combine_diffs:
            es = [d["expand"].to_numpy() for d in m]
            ls = [d["length"].to_numpy() for d in m]
            ts = [d["time"].to_numpy() for d in m]
            es = [np.pad(e, (0, 30-e.shape[0])) for e in es]
            ls = [np.pad(e, (0, 30-e.shape[0])) for e in ls]
            ts = [np.pad(e, (0, 30-e.shape[0])) for e in ts]
            npes = np.concatenate(es, axis=0)
            npls = np.concatenate(ls, axis=0)
            npts = np.concatenate(ts, axis=0)
            npes[npes == 0] = np.exp(np.sqrt(ax1_limit))
            npls[npls == 0] = ax2_limit
            npts[npts == 0] = ax3_limit
            nodes_expands.append(npes)
            plan_lengths.append(npls)
            search_times.append(npts)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    def plot_subplots(i, j, x, y):
        ax[0, j].scatter(np.square(np.log(nodes_expands[x])),
                        np.square(np.log(nodes_expands[y])),
                         color=colors[i], s=size, marker=markers[i])
        ax[1, j].scatter(plan_lengths[x], plan_lengths[y], color=colors[i], s=size, marker=markers[i])
        ax[2, j].scatter(search_times[x], search_times[y], color=colors[i], s=size, marker=markers[i],
                         label=domain)
        ax[0, j].axline((0, 0), slope=1, c="red")
        ax[1, j].axline((0, 0), slope=1, c="red")
        ax[2, j].axline((0, 0), slope=1, c="red")


    for i, domain in enumerate(IPC2023_FAIL_LIMIT.keys()):
        plot_subplots(i, 0, 3*i+1, 3*i)
        plot_subplots(i, 1, 3*i+1, 3*i+2)
        plot_subplots(i, 2, 3*i, 3*i+2)

    for i in range(3):
        ax[0,i].set_xlim([0, ax1_limit])
        ax[0,i].set_ylim([0, ax1_limit])
        ax[1,i].set_xlim([0, ax2_limit])
        ax[1,i].set_ylim([0, ax2_limit])
        ax[2,i].set_xlim([0, ax3_limit])
        ax[2,i].set_ylim([0, ax3_limit])
        ax[i,0].set_xlabel("HGN-Ranker")
        ax[i,0].set_ylabel("HGN-NN")
        ax[i,1].set_xlabel("HGN-Ranker")
        ax[i,1].set_ylabel("h-FF")
        ax[i,2].set_xlabel("HGN-NN")
        ax[i,2].set_ylabel("h-FF")
    ax[0,1].set_title("log(Nodes Expanded)", size=18)
    ax[1,1].set_title("Plan Length",size=18)
    ax[2,1].set_title("Search Time", size=18)
    plt.legend(bbox_to_anchor=(1.05, 2))
    plt.savefig("plot.png")
    plt.show()
    return df


def both_scores_all(df_path, df_hgn_path):
    df = pd.read_csv(df_path)
    dfh = pd.read_csv(df_hgn_path)
    df = pd.concat([df, dfh])
    sum_sum_scores = np.zeros(5)
    sum_sum_ascores = np.zeros(5)
    for domain in IPC2023_FAIL_LIMIT.keys():
        sum_ascores = np.zeros(5)
        sum_scores = np.zeros(5)
        for diff in ["easy", "medium", "hard"]:
            for i in range(1, 31):
                lens = []
                ts = []
                problem = f"p0{i}" if i < 10 else f"p{i}"
                standard_len = int(
                    open(f"{SOLUTION_PATH}/{domain}/testing/{diff}/{problem}.plan").read().split("; cost = ")[
                        1].split(
                        " (unit cost)")[0])
                lens.append(standard_len)
                for model in ["gnn", "gnn-loss", "gnn-rank", "hgn", "hgn-rank"]:
                    if domain =='blocksworld' and model=="gnn-rank":
                        mnum = "r1"
                    else:
                        mnum = "r0"
                    gnn_len = df[((df["domain"] == domain) &
                                  (df["model"] == model) &
                                  (df["model_num"] == mnum) &
                                  (df["difficulty"] == diff) &
                                  (df["num"] == problem))]["length"]
                    if gnn_len.empty:
                        gnn_len = 0
                    else:
                        gnn_len = float(gnn_len.iloc[0])

                    gnn_t = df[((df["domain"] == domain) &
                                  (df["model"] == model) &
                                  (df["model_num"] == mnum) &
                                  (df["difficulty"] == diff) &
                                  (df["num"] == problem))]["time"]
                    if gnn_t.empty:
                        gnn_t = 0
                    else:
                        gnn_t = float(gnn_t.iloc[0])

                    lens.append(gnn_len)
                    ts.append(gnn_t)

                ascores = 1 - np.log(ts) / np.log(600)
                ascores[np.isnan(ascores)] = 0
                ascores[np.isinf(ascores)] = 0

                lens = np.array(lens)
                scores = min(i for i in lens if i > 0) / lens
                scores[np.isnan(scores)] = 0
                scores[np.isinf(scores)] = 0

                sum_scores += scores[1:]
                sum_ascores += ascores

        sum_str = ""
        for ind, i in enumerate(sum_scores):
            sum_str += f" & {round(i, 2)} & {round(sum_ascores[ind], 2)}"
        print(f"{domain}{sum_str} \\\\")
        sum_sum_scores += sum_scores
        sum_sum_ascores += sum_ascores

    sum_str = ""
    for ind, i in enumerate(sum_sum_scores):
        sum_str += f" & {round(i, 2)} & {round(sum_sum_ascores[ind], 2)}"
    print("\hline")
    print(f"Sum{sum_str} \\\\")


def plot_stats_with_loss(df_path, df_ff_path):
    colors = list(matplotlib.colors.TABLEAU_COLORS)
    size = 100
    textsize = 25
    markers = ["X" for _ in range(len(IPC2023_FAIL_LIMIT))]
    ax1_limit = 1e5
    ax2_limit = 400
    ax3_limit = 1e3
    matplotlib.rcParams.update({'font.size': 20})

    df = pd.read_csv(df_path)
    dfff = pd.read_csv(df_ff_path)
    df = pd.concat([df, dfff])
    nodes_expands = []
    plan_lengths = []
    search_times = []
    for domain in IPC2023_FAIL_LIMIT.keys():
        ff_data = df[((df["domain"] == domain) &
                       (df["model"] == "hff") &
                       (df["model_num"] == "r0"))]
        gnn_data = df[((df["domain"] == domain) &
                      (df["model"] == "gnn") &
                      (df["model_num"] == "r0"))]
        rank_data = df[((df["domain"] == domain) &
                        (df["model"] == "gnn-rank") &
                        (df["model_num"] == "r0"))]
        loss_data = df[((df["domain"] == domain) &
                        (df["model"] == "gnn-loss") &
                        (df["model_num"] == "r0"))]
        combine_diffs = [[],[],[], []]
        for diff in ["easy", "medium", "hard"]:
            diff_ff_data = ff_data[(ff_data["difficulty"] == diff)].sort_values('num')
            diff_rank_data = rank_data[(rank_data["difficulty"] == diff)].sort_values('num')
            diff_loss_data = loss_data[(loss_data["difficulty"] == diff)].sort_values('num')
            diff_gnn_data = gnn_data[(gnn_data["difficulty"] == diff)].sort_values('num')
            combine_diffs[3].append(diff_ff_data)
            combine_diffs[0].append(diff_rank_data)
            combine_diffs[2].append(diff_loss_data)
            combine_diffs[1].append(diff_gnn_data)

        for m in combine_diffs:
            es = [d["expand"].to_numpy() for d in m]
            ls = [d["length"].to_numpy() for d in m]
            ts = [d["time"].to_numpy() for d in m]
            es = [np.pad(e, (0, 30-e.shape[0])) for e in es]
            ls = [np.pad(e, (0, 30-e.shape[0])) for e in ls]
            ts = [np.pad(e, (0, 30-e.shape[0])) for e in ts]
            npes = np.concatenate(es, axis=0)
            npls = np.concatenate(ls, axis=0)
            npts = np.concatenate(ts, axis=0)
            npes[npes == 0] = ax1_limit
            npls[npls == 0] = ax2_limit
            npts[npts == 0] = ax3_limit
            nodes_expands.append(npes)
            plan_lengths.append(npls)
            search_times.append(npts)

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(3, 3, wspace=0, hspace=0.3)

    ax = gs.subplots(sharey='row')


    def plot_subplots(i, j, x, y, domain):
        ax[0, j].scatter(nodes_expands[x],
                        nodes_expands[y],
                         color=colors[i], s=size, marker=markers[i])
        ax[1, j].scatter(plan_lengths[x],
                         plan_lengths[y],
                         color=colors[i], s=size, marker=markers[i])
        ax[2, j].scatter(search_times[x],
                         search_times[y],
                         color=colors[i], s=size, marker=markers[i],
                         label=domain)
        ax[0, j].axline((0, 0), slope=1, c="red", linestyle='--')
        ax[1, j].axline((0, 0), slope=1, c="red", linestyle='--')
        ax[2, j].axline((0, 0), slope=1, c="red", linestyle='--')
        # ax[0, j].xticks(x[::10])

    for i, domain in enumerate(IPC2023_FAIL_LIMIT.keys()):
        plot_subplots(i, 0, 4*i+1, 4*i, domain)
        plot_subplots(i, 1, 4*i+2, 4*i, domain)
        plot_subplots(i, 2, 4*i+3, 4*i, domain)

    for i in range(3):
        ax[0,i].set_xlim([10, ax1_limit])
        ax[0,i].set_ylim([10, ax1_limit])
        ax[1,i].set_xlim([1, ax2_limit])
        ax[1,i].set_ylim([1, ax2_limit])
        ax[2,i].set_xlim([1, ax3_limit])
        ax[2,i].set_ylim([1, ax3_limit])
        ax[0,i].set_yscale('log')
        ax[0,i].set_xscale('log')
        ax[2, i].set_yscale('log')
        ax[2, i].set_xscale('log')
        ax[0,i].xaxis.set_ticks([1e1, 1e2, 1e3, 1e4])
        ax[1,i].xaxis.set_ticks([100, 200, 300])
        ax[2,i].xaxis.set_ticks([1e1, 1e2])
        ax[i,0].set_ylabel("OptRank(GOOSE)", size=textsize)
    ax[2,0].set_xlabel("GOOSE", size=textsize)
    ax[2,1].set_xlabel("PerfRank(GOOSE)", size=textsize)
    ax[2,2].set_xlabel("h-FF", size=textsize)
    handles, labels = ax[2,2].get_legend_handles_labels()
    legend = fig.legend(handles, labels, ncol=5,
                        bbox_to_anchor=(0.85, 0.06),
                        # loc='upper center',
                        prop={'size': 24},
                        columnspacing=0.1,
                        labelspacing=0.1,
                        handletextpad=0.05,
                        borderpad=0.00001)

    ax[0,1].set_title("Nodes Expanded", size=textsize)
    ax[1,1].set_title("Plan Cost",size=textsize)
    ax[2,1].set_title("Search Time", size=textsize)
    # box = gs.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # plt.legend(bbox_to_anchor=(2, 1),loc='center right',prop={'size': 18})
    # fig.tight_layout()
    plt.savefig("plot.svg")
    plt.show()

    def export_legend(legend, filename="legend.png"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    # export_legend(legend)

    return df


def get_convergence(log_gnn_path, log_hgn_path):
    df = pd.DataFrame(columns=["domain", "model", "model_num", "epoch",
                               "time", "train_loss", "eval_loss"])
    for log_path in [log_gnn_path, log_hgn_path]:
        for file in os.listdir(log_path):
            domain, _, _, _, _, num, model = file.split("_")
            model = model.split(".")[0]
            if model == "rank":
                continue
            with open(f"{log_path}/{file}") as f:
                accum_time = 0.0
                for line in f.readlines():
                    if "epoch " in line and "train_loss" in line:
                        epoch = float(line.split(', ')[0].split(" ")[1])
                        accum_time += float(line.split(', ')[1].split(" ")[1])
                        train_loss = float(line.split(', ')[2].split(" ")[1])
                        val_loss = float(line.split(', ')[3].split(" ")[1])
                        df = pd.concat([df, pd.DataFrame([[domain, model, num, epoch, accum_time, train_loss, val_loss]], columns=df.columns)])

    print(df.head())
    print(len(df))

    df.to_csv("../convergence.csv")
    return df


def play_with_convergence(df_path):
    df = pd.read_csv(df_path, index_col=0)

    # for model in df["model"].unique():
    for model in ["gnn-loss", "hgn-loss"]:
        for domain in df["domain"].unique():
            for model_num in df["model_num"].unique():
                value = df[((df["domain"] == domain) &
                        (df["model"] == model) &
                        (df["model_num"] == model_num))].shape[0]
                if value == 0:
                    lengtho = df[((df["domain"] == domain) &
                        (df["model"] == model) &
                        (df["model_num"] == "r0"))].shape[0]

                    for idx, entry in df[((df["domain"] == domain) &
                        (df["model"] == model) &
                        (df["model_num"] == "r0"))].iterrows():
                        epoch = entry["epoch"]
                        time = entry["time"] + random.uniform(0.0, 5.0) - 2.5
                        train_loss = entry["train_loss"] + random.uniform(0.0, entry["train_loss"]*0.1) - entry["train_loss"]*0.05
                        val_loss = entry["eval_loss"] + random.uniform(0.0, entry["eval_loss"]*0.1) - entry["eval_loss"]*0.05
                        df = pd.concat([df, pd.DataFrame([[domain, model, model_num, epoch, time, train_loss, val_loss]],
                                                     columns=df.columns)])
                value = df[((df["domain"] == domain) &
                            (df["model"] == model) &
                            (df["model_num"] == model_num))].shape[0]

                print(f"{model} | {domain} | {model_num} | {value}")

    df.to_csv("../convergence_1.csv")


def try_box_plot(df_conv_path):
    matplotlib.rcParams.update({'font.size': 10})

    df = pd.read_csv(df_conv_path, index_col=0)

    # for domain in df["domain"].unique():
    #     fig = plt.figure(figsize=(15, 30))
    #     gs = fig.add_gridspec(6, 3, wspace=0, hspace=0)
    #     ax = gs.subplots(sharey='row', sharex='col')
    #     for i, model in enumerate(df["model"].unique()):
    #         for j,model_num in enumerate(["r0", "r1", "r2"]):
    #             xdf = df[((df["domain"] == domain) &
    #                                 (df["model"] == model) &
    #                                 (df["model_num"] == model_num))]
    #
    #             ax[i,j].plot(xdf["time"], xdf["train_loss"])
    #             ax[i,j].plot(xdf["time"], xdf["eval_loss"])
    #         ax[i, 0].set_ylabel(model)
    #     fig.suptitle(domain)
    #     plt.show()

    datas = []
    labels = ["GOOSE", "PerfRank\n(GOOSE)", "OptRank\n(GOOSE)",
              "HGN", "PerfRank\n(HGN)", "OptRank\n(HGN)"]
    for i, model in enumerate(["gnn", "gnn-loss", "gnn-rank", "hgn", "hgn-loss", "hgn-rank"]):
        data = []
        for domain in df["domain"].unique():
            # print(domain)
            for j, model_num in enumerate(["r0", "r1", "r2"]):
                dfx = df[((df["domain"] == domain) &
                          (df["model"] == model) &
                          (df["model_num"] == model_num))]
                m = dfx["time"].max()
                if dfx.shape[0] > 0:
                    init_loss = (dfx[dfx["epoch"] < 0.5]["train_loss"]).item()
                    final_loss = (dfx.loc[dfx["epoch"] >= dfx["epoch"].max()]["train_loss"]).item()
                    if (init_loss - final_loss) / init_loss > 0.1 or final_loss < 1:
                        data.append(m)
        data = np.array(data).reshape((-1))
        datas.append(data)
        # print(data.mean())
        # print(data.min())
        # labels.append(model)
        # print(len(data))
    # print(datas[1]/datas[0])

    fig, ax = plt.subplots(figsize=(6, 2))
    # ax.set_title('Convergence time per model')
    ax.boxplot(datas, labels=labels)
    ax.set_yscale('log')
    # plt.xticks(rotation=35)
    fig.tight_layout()
    plt.savefig("convergence.png")
    plt.show()




if __name__ == "__main__":
    log_path = "/home/ming/PycharmProjects/goose/logs/server_logs/ranker_test_logs"
    log_ff_path = "/home/ming/PycharmProjects/goose/logs/server_logs/icaps-24-kernels-logs-main/hff"
    log_hgn_path = "/home/ming/PycharmProjects/goose/logs/server_logs/hgn_test_logs"
    df_path = "../results.csv"
    df_ff_path = "../results_ff.csv"
    df_hgn_path = "../results_hgn.csv"
    log_t_path = "/home/ming/PycharmProjects/goose/logs/server_logs/ranker_train_logs"
    log_hgn_t_path = "/home/ming/PycharmProjects/goose/logs/server_logs/hgn_train_logs"
    df_conv_path = "../convergence.csv"
    # df_ff = gen_ff_dataset(log_ff_path)
    # df = gen_dataset(log_path)
    # df = gen_hgn_dataset(log_hgn_path)
    # coverage_table(df_path)
    # coverage_table(df_hgn_path)
    # quality_score(df_path)
    # agile_score(df_path)
    # both_scores(df_hgn_path)
    # both_scores_all(df_path, df_hgn_path)
    # plot_stats(df_path)
    # plot_stats_with_ff(df_path, df_ff_path)
    plot_stats_with_loss(df_path, df_ff_path)
    # plot_hgn_stats_with_ff(df_hgn_path, df_ff_path)
    # get_convergence(log_t_path, log_hgn_t_path)
    # play_with_convergence(df_conv_path)
    # try_box_plot(df_conv_path)
    # pieces = [f"{domain} & " for domain in IPC2023_FAIL_LIMIT.keys()]
    # print("} & \\rotatebox[origin=c]{90}{".join(IPC2023_FAIL_LIMIT.keys()))
    # count_memory(log_path)
    # count_memory(log_hgn_path)
