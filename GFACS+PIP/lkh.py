from datetime import timedelta
import argparse
import os
import pickle
import time

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from urllib.parse import urlparse
from subprocess import check_call
from tqdm import tqdm

import numpy as np
import torch

from utils import load_test_dataset, load_val_dataset, get_capacity


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def calc_vrp_cost(depot, loc, tour):
    assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
    # TODO validate capacity constraints
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.9.tgz"):
    cwd = os.path.abspath("lkh")
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)


def write_vrplib(filename, depot, loc, demand, capacity, grid_size, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x / grid_size * 100000 + 0.5), int(y / grid_size * 100000 + 0.5))  # VRPlib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_vrplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def solve_lkh_log(executable, directory, name, depot, loc, demand, capacity, grid_size=1, runs=10, disable_cache=False):
    problem_filename = os.path.join(directory, f"lkh_run{runs}", f"{name}.vrp")
    tour_filename = os.path.join(directory, f"lkh_run{runs}", f"{name}.tour")
    output_filename = os.path.join(directory, f"lkh_run{runs}", f"{name}.pkl")
    param_filename = os.path.join(directory, f"lkh_run{runs}", f"{name}.par")
    log_filename = os.path.join(directory, f"lkh_run{runs}", f"{name}.log")

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_vrplib(problem_filename, depot, loc, demand, capacity, grid_size, name=name)

            params = {
                "PROBLEM_FILE": problem_filename,
                "OUTPUT_TOUR_FILE": tour_filename,
                "RUNS": runs,
                "MAX_TRIALS": len(loc) * 100,
                "SEED": 1234
            }
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_vrplib(tour_filename, n=len(demand))

            save_dataset((tour, duration), output_filename)

        return calc_vrp_cost(depot, loc, tour), tour, duration

    except Exception as e:
        raise e
        print("Exception occured")
        print(e)
        return None


def run_all_in_pool(func, directory, dataset, n_cpus=None, use_multiprocessing=True):
    num_cpus = os.cpu_count() if n_cpus is None else n_cpus

    w = len(str(len(dataset) - 1))
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)  # type: ignore
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.map(
            func,
            [(directory, str(i).zfill(w), *problem) for i, problem in enumerate(dataset)],
        ), total=len(dataset)))

    failed = [str(i) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("--type", type=str, default="val", help="Dataset type")
    parser.add_argument("--result_dir", type=str, default="lkh/results", help="Result directory")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of cpus to use")
    parser.add_argument("--size", type=int, default=None, help="Number of instances to solve")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to perform")
    parser.add_argument("--tam", action="store_true", help="Use TAM dataset")

    opt = parser.parse_args()

    if opt.type == "val":
        dataset = load_val_dataset(opt.nodes, opt.nodes // 5, "cpu", tam=opt.tam)
    elif opt.type == "test":
        dataset = load_test_dataset(opt.nodes, opt.nodes // 5, "cpu", tam=opt.tam)
    else:
        raise ValueError("Invalid dataset type")
    capacity = get_capacity(opt.nodes, opt.tam)

    size = opt.size or len(dataset)
    dataset = dataset[:size]

    target_dir = os.path.join(opt.result_dir, f"{opt.type}Dataset{'-tam' if opt.tam else ''}-{opt.nodes}-{size}")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    if not os.path.isdir(os.path.join(target_dir, f"lkh_run{opt.runs}")):
        os.makedirs(os.path.join(target_dir, f"lkh_run{opt.runs}"))
    out_file = os.path.join(target_dir, f"{opt.type}Dataset{'-tam' if opt.tam else ''}-{opt.nodes}-lkh_run{opt.runs}.pkl")

    solver_input = []
    for data in dataset:
        solver_input.append(
            (
                data[3][0].tolist(),
                data[3][1:].tolist(),
                (data[1][1:] * capacity).type(torch.int16).tolist(),
                capacity,
            )
        )

    executable = get_lkh_executable()
    runs = opt.runs
    use_multiprocessing = False

    def run_func(args):
        directory, name, *args = args
        depot, loc, demand, capacity = args

        return solve_lkh_log(
            executable, directory, name, depot, loc, demand, capacity, runs=runs
        )

    # Note: only processing n items is handled by run_all_in_pool
    results, parallelism = run_all_in_pool(
        run_func, target_dir, solver_input, n_cpus=opt.n_cpus, use_multiprocessing=use_multiprocessing
    )

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative
    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))  # type: ignore
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))  # type: ignore
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    save_dataset((results, parallelism), out_file)
