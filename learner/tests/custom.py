import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="linear-svr")
    parser.add_argument("-r", "--representation", default="ilg")
    parser.add_argument("-k", "--wl", required=True)
    parser.add_argument("-l", "--iterations", default="1", type=str)
    parser.add_argument("-d", "--domain", default="ferry")

    parser.add_argument("--train", dest="run", action="store_false")
    parser.add_argument("--run", dest="train", action="store_false")
    
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("-p", "--problem", default="p10")
    args = parser.parse_args()

    model=args.model
    representation=args.representation
    domain=args.domain
    wl=args.wl
    iterations=args.iterations
    difficulty = args.difficulty
    problem = args.problem

    save_file = f"tests/" + "_".join([wl, iterations, model, domain]) + ".joblib"

    if args.train:
        os.system(f"python3 train_kernel.py -m {model} -r {representation} -d {domain} -k {wl} -l {iterations} --model-save-file {save_file}")

    if args.run:
        os.system(f"cd ../planners/downward && python3 build.py")
        domain_file=f"../benchmarks/ipc2023-learning-benchmarks/{domain}/domain.pddl"
        problem_file=f"../benchmarks/ipc2023-learning-benchmarks/{domain}/testing/{difficulty}/{problem}.pddl"
        assert os.path.exists(problem_file), problem_file
        os.system(f"python3 run_kernel.py {domain_file} {problem_file} {save_file}")
