import os

import pandas as pd

df = pd.DataFrame(columns=["domain", "difficulty", "num", "encodes",
                           "layers", "hidden", "method", "model_num", "model",
                           "length", "expand", "evaluated", "time"])
for file in os.listdir("./"):
    if file == "statistics.py":
        continue
    domain, difficulty, num, _, encodes, layers, hidden, method, model_num, model = file.split("_")
    model = model.split(".")[0]
    length = 0
    expand = 0
    eval = 0
    time = 0
    with open(file) as f:
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

df.to_csv("../results.csv")
