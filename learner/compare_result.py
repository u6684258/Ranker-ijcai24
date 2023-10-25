import os.path as osp
import os
import json

GOOSE_EXP_PATH = "logs/2023-10-24T12:05:47.855728/result"
RANKER_EXP_PATH = "logs/2023-10-23T16:12:44.301576/result"

ranker_dirs = os.listdir(RANKER_EXP_PATH)
goose_dirs = os.listdir(GOOSE_EXP_PATH)

result = {}

for dir in ranker_dirs:
    succ_count = 0
    p = osp.join(RANKER_EXP_PATH, dir)
    domain = os.listdir(p)[0]
    result_file = f"{p}/{domain}/{domain}.json"
    with open(result_file, "r") as f:
        result_list = json.load(f)

        for item in result_list:
            # count success
            if item["search_state"] == "success":
                succ_count += 1

    result[dir] = succ_count

print(result)
