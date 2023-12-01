import argparse
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.save_load import load_kernel_model

parser = argparse.ArgumentParser()
parser.add_argument("m1")
parser.add_argument("m2")
args = parser.parse_args()

assert os.path.exists(args.m1), args.m1
assert os.path.exists(args.m2), args.m2
m1 = load_kernel_model(args.m1)
m2 = load_kernel_model(args.m2)
w1 = m1.get_weights()
w2 = m2.get_weights()

l1_diff = np.linalg.norm(w1-w2, ord=1)
l2_diff = np.linalg.norm(w1-w2, ord=2)
linf_diff = np.linalg.norm(w1-w2, ord=np.inf)
print("l1", l1_diff)
print("l2", l2_diff)
print("linf", linf_diff)
breakpoint()
