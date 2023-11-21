import os.path as osp
import os
import json

RANKER_EXP_PATH = "logs/logs_from_server/latest/logs/final"

FF_EXP_PATH = "result/hff-2023-11-15T18:14:36.713994"

ranker_dirs = os.listdir(RANKER_EXP_PATH)
ff_dirs = os.listdir(FF_EXP_PATH)


param = "rank_coverage"
result = {}
resultf = {}

if param == "rank_coverage":
    for dir in ranker_dirs:
        succ_count = 0
        fail_count = 0
        p = osp.join(RANKER_EXP_PATH, dir)
        domain = os.listdir(p)[0]
        result_file = f"{p}/{domain}/{domain}.json"
        try:
            with open(result_file, "r") as f:
                result_list = json.load(f)

                for item in result_list:
                    # count success
                    if item["search_state"] == "success":
                        succ_count += 1

                    elif item["search_state"] == "failed":
                        fail_count += 1

            result[dir] = succ_count
            resultf[dir] = fail_count
        except Exception:
            print(f"Log Not found or something wrong! {result_file}")

elif param == "ff":
    for dir in ranker_dirs:
        succ_count = 0
        p = osp.join(RANKER_EXP_PATH, dir)
        result_file = f"{p}/{dir}.json"
        try:
            with open(result_file, "r") as f:
                result_list = json.load(f)

                for item in result_list:
                    # count success
                    if item["search_state"] == "success":
                        succ_count += 1

            result[dir] = succ_count
        except Exception:
            print(f"Log Not found or something wrong! {result_file}")



for key, value in sorted(list((k, v) for k, v in result.items())):
    print(f"{key} : {value}")

print("fail count:")
for key, value in sorted(list((k, v) for k, v in resultf.items())):
    if value > 0:
        print(f"{key} : {value}")