""" Module for dealing with model saving and loading. """
import os
import pickle
import traceback

from models.rank_model import RankModel


def arg_to_params(args, in_feat=4, out_feat=1):
    # this is an artifact of legacy code, it could
    nlayers = args.nlayers
    nhid = args.nhid
    in_feat = args.in_feat
    # out_feat = args.out_feat
    n_edge_labels = args.n_edge_labels
    pool = args.pool
    aggr = args.aggr
    rep = args.rep
    model_params = {
        "in_feat": in_feat,
        "out_feat": out_feat,
        "nlayers": nlayers,
        "n_edge_labels": n_edge_labels,
        "nhid": nhid,
        "aggr": aggr,
        "pool": pool,
        "rep": rep,
        "model": args.model,
    }
    return model_params


def print_arguments(args, ignore_params=set()):
    print("Parsed arguments:")
    for k, v in vars(args).items():
        if k in ignore_params.union(
            {"device", "optimal", "save_model", "save_file", "no_tqdm", "tqdm", "fast_train"}
        ):
            continue
        print("{0:20}  {1}".format(k, v))
    print("___")


def save_gnn_model_from_dict(model_dict, args):
    import torch
    if not hasattr(args, "save_file") or args.save_file is None:
        return
    save_file = args.save_file
    save_dir = os.path.dirname(save_file)
    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)
    print(f"Saving model at {save_file}...")
    torch.save((model_dict, args), save_file)
    print("Model saved!")
    print("Model parameter file:", save_file)
    return


def save_gnn_model(model, args):
    save_gnn_model_from_dict(model.model.state_dict(), args)
    return


def save_kernel_model(model, args):
    if not hasattr(args, "model_save_file") or args.model_save_file is None:
        return
    print("Saving model...")
    model_save_file = args.model_save_file
    base_dir = os.path.dirname(model_save_file)
    if len(base_dir) > 0:
        os.makedirs(base_dir, exist_ok=True)
    model.setup_for_saving()
    with open(model_save_file, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Model saved!")
    print("Model parameter file:", model_save_file)
    return


def load_gnn_model(path, print_args=False):
    # returns (GNN, Args)
    import torch
    from models.gnn import Model

    print(f"Loading model from {path}...")
    if not os.path.exists(path):
        print(f"Model not found at {path}")
        exit(-1)
    if torch.cuda.is_available():
        model_state_dict, args = torch.load(path)
    else:
        model_state_dict, args = torch.load(path, map_location=torch.device("cpu"))
    model_params = arg_to_params(args)
    if args.model == "gnn-rank":
        model_params["out_feat"] = args.nEmb
        model = RankModel(params=model_params)
    else:
        model = Model(params=model_params)
    model.load_state_dict_into_gnn(model_state_dict)
    print("Model loaded!")
    if print_args:
        print_arguments(args)
    model.set_eval()
    return model, args


def load_kernel_model(path):
    with open(path, 'rb') as handle:
        try:
            model = pickle.load(handle)
        except Exception:
            print(traceback.format_exc(), flush=True)
    return model


def load_gnn_model_and_setup(path, domain_file, problem_file):
    import torch
    model, args = load_gnn_model(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.batch_search(True)
    model.update_representation(
        domain_pddl=domain_file, problem_pddl=problem_file, args=args, device=device
    )
    model.set_zero_grad()
    model.eval()
    return model


def load_kernel_model_and_setup(path, domain_file, problem_file):
    try:
        print(f"Entered Python code for loading model", flush=True)
        model = load_kernel_model(path)
        print("Updating representation", flush=True)
        model.update_representation(domain_pddl=domain_file, problem_pddl=problem_file)
        print("Representation updated!", flush=True)
        model.eval()
        print("Set to eval mode.", flush=True)
    except Exception:
        print(traceback.format_exc(), flush=True)
    return model


def save_hgn_model(model, args):
    import torch
    if not hasattr(args, "save_file") or args.save_file is None:
        return
    save_file = args.save_file
    save_dir = os.path.dirname(save_file)
    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)
    print(f"Saving model at {save_file}...")
    torch.save(model, save_file)
    print("Model saved!")
    print("Model parameter file:", save_file)
    return


def load_hgn_model(path, print_args=False, jit=False, ignore_subdir=False):
    print("Loading model...")
    import torch
    if torch.cuda.is_available():
        model = torch.load(path)
    else:
        model = torch.load(path, map_location=torch.device("cpu"))

    # update legacy naming
    print("Model loaded!")
    model.eval()
    return model