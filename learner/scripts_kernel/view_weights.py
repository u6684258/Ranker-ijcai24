import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.save_load import load_kernel_model

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

assert os.path.exists(args.model), args.model
model = load_kernel_model(args.model)
weights = model.get_weights()

print(weights)
breakpoint()
