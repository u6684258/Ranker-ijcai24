import os
import re
from representation import REPRESENTATIONS

""" Module containing useful methods and configurations for 24-AAAI search experiments. """

REPEATS = 1
VAL_REPEATS = 5
TIMEOUT = 600000  # 10 minute timeout + time to load model etc.
FAIL_LIMIT = {
    "gripper": 1,
    "spanner": 10,
    "visitall": 10,
    "visitsome": 10,
    "blocks": 10,
    "ferry": 10,
    "sokoban": 20,
    "n-puzzle": 10,
}

PROFILE_CMD_ = "valgrind --tool=callgrind --callgrind-out-file=callgrind.out --dump-instr=yes --collect-jumps=yes"


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def search_cmd(df, pf, m, model_type, planner, search, seed, profile, timeout=TIMEOUT, aux_file=None, plan_file=None):
    search_engine = {
        "pwl": pwl_cmd,
        "fd": fd_cmd,
        "hgn": hgn_cmd,
    }[planner]
    cmd, aux_file = search_engine(df, pf, model_type, m, search, seed, profile, timeout, aux_file, plan_file)
    os.environ['GOOSE'] = f'{os.getcwd()}'
    os.environ['STRIPS_HGN_NEW'] = f'{os.getcwd()}'
    os.environ['FD_HGN'] = f'{os.getcwd()}/../FD-Hypernet-master'
    # cmd = f"export GOOSE={os.getcwd()} && {cmd}"
    return cmd, aux_file


def pwl_cmd(df, pf, model_type, m, search, seed, profile, timeout=TIMEOUT, aux_file=None, plan_file=None):
    description = f"pwl_{pf.replace('.pddl', '').replace('/', '-')}_{search}_{os.path.basename(m).replace('.dt', '')}".replace(
        '.', '')

    if aux_file is None:
        os.makedirs("lifted", exist_ok=True)
        aux_file = f"lifted/{description}.lifted"

    if plan_file is None:
        os.makedirs("plans", exist_ok=True)
        plan_file = f"plans/{description}.plan"

    cmd = f"./../powerlifted/powerlifted.py --gpu " \
          f"-d {df} " \
          f"-i {pf} " \
          f"-m {m} " \
          f"-e {model_type} " \
          f"-s {search} " \
          f"--time-limit {timeout} " \
          f"--seed {seed} " \
          f"--translator-output-file {aux_file} " \
          f"--plan-file {plan_file}"
    return cmd, aux_file


def fd_cmd(df, pf, model_type, m, search, seed, profile, timeout=TIMEOUT, aux_file=None, plan_file=None):
    if search == "gbbfs":
        search = "batch_eager_greedy"
    elif search == "gbfs":
        search = "eager_greedy"
    else:
        raise NotImplementedError

    description = f"fd_{pf.replace('.pddl', '').replace('/', '-')}_{search}_{os.path.basename(m).replace('.dt', '')}".replace(
        '.', '')

    if aux_file is None:
        os.makedirs("sas_files", exist_ok=True)
        aux_file = f"sas_files/{description}.sas_file"

    if plan_file is None:
        os.makedirs("plans", exist_ok=True)
        plan_file = f"plans/{description}.plan"

    cmd = f"./../downward/fast-downward.py --search-time-limit {timeout} --sas-file {aux_file} --plan-file {plan_file} " + \
          f"{df} {pf} --search {search}([goose(model_path=\"{m}\"," + \
          f"model_type=\"{model_type}\"," + \
          f"domain_file=\"{df}\"," + \
          f"instance_file=\"{pf}\"" + \
          f")])"
    # print(cmd)
    return cmd, aux_file


def fd_general_cmd(domain_file, problem_file, result_file, search="astar+lmcut"):
    THIS_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../downward-full"))
    cmd = ["python3", os.path.join(THIS_DIR, "fast-downward.py"),
           "--search-time-limit", "60",
           "--plan-file", result_file,
           domain_file, problem_file]

    if search == "astar+lmcut":
        cmd += ["--search", "astar(lmcut())"]

    if search == "hff":
        cmd += ["--evaluator",
                "hff=ff(transform=adapt_costs(one))", "--search",
                ("eager_greedy([hff])")]

    elif search == "lamafirst":
        cmd += ["--evaluator",
                ("hlm=landmark_sum(lm_factory=lm_reasonable_orders_hps(lm_rhw()),"
                 "transform=adapt_costs(one),pref=false)"), "--evaluator",
                "hff=ff(transform=adapt_costs(one))", "--search",
                ("lazy_greedy([hff,hlm],preferred=[hff,hlm],cost_type=one,"
                 "reopen_closed=false)")]
    else:
        cmd += ["--search", "astar(lmcut())"]

    return cmd


def hgn_cmd(df, pf, model_type, m, search, seed, profile, timeout=TIMEOUT, aux_file=None, plan_file=None):
    if search == "gbbfs":
        search = "batch_eager_greedy"
    elif search == "gbfs":
        search = "eager_greedy"
    else:
        raise NotImplementedError

    description = f"fd_{pf.replace('.pddl', '').replace('/', '-')}_{search}_{os.path.basename(m).replace('.dt', '')}".replace(
        '.', '')

    if aux_file is None:
        os.makedirs("sas_files", exist_ok=True)
        aux_file = f"sas_files/{description}.sas_file"

    if plan_file is None:
        os.makedirs("plans", exist_ok=True)
        plan_file = f"plans/{description}.plan"

    cmd = f"./../FD-Hypernet-master/fast-downward.py --search-time-limit {timeout} --sas-file {aux_file} --plan-file {plan_file} " + \
          f"{df} {pf} --translate-options --full-encoding --search-options " \
          f"--search eager(single(hgn2(network_file={m}," \
                                    f"domain_file={df}," \
                                    f"instance_file={pf}," \
                                    f"type={model_type})))"
    # print(cmd)
    return cmd, aux_file