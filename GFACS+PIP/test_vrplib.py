from typing import Tuple, List
import os
import random
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from net import Net
from aco import ACO, get_subroutes
from utils import load_vrplib_dataset


EPS = 1e-10


def validate_route(int_distances: np.ndarray, demands: torch.Tensor, routes: List[torch.Tensor]) -> Tuple[bool, int]:
    length = 0
    valid = True
    visited = {0}
    for r in routes:
        d = demands[r].sum().item()
        if d>1.000001:
            valid = False
        length += int_distances[r[:-1], r[1:]].sum()
        for i in r:
            i = i.item()
            if i<0 or i >= int_distances.shape[0]:
                valid = False
            else:
                visited.add(i)  # type: ignore
    if len(visited) != int_distances.shape[0]:
        valid = False
    return valid, length


@torch.no_grad()
def infer_instance(model, pyg_data, demands, distances, positions, n_ants, t_aco_diff, int_distances):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
        demand=demands.cpu(),
        distances=distances.cpu(),
        device='cpu',
        swapstar=True,
        positions=positions.cpu(),
    )

    results = torch.zeros(size=(len(t_aco_diff),), dtype=torch.int64)
    for i, t in enumerate(t_aco_diff):
        _, _ = aco.run(t)
        path = get_subroutes(aco.shortest_path)
        valid, length = validate_route(int_distances, demands, path)  # use int_distances here
        if valid is False:
           print("invalid solution.")
        results[i] = length
    return results


@torch.no_grad()
def test(dataset, model, n_ants, t_aco, int_dist_list):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    results_list = []
    start = time.time()
    for (pyg_data, demands, distances, positions), int_distances in tqdm(zip(dataset, int_dist_list)):
        results = infer_instance(model, pyg_data, demands, distances, positions, n_ants, t_aco_diff, int_distances)
        results_list.append(results)
    end = time.time()
    return results_list, end - start


def main(ckpt_path, n_nodes, k_sparse_factor, n_ants=100, n_iter=10, guided_exploration=False, seed=0, dataset_name="X"):
    test_list, int_dist_list, name_list = load_vrplib_dataset(n_nodes, k_sparse_factor, DEVICE, dataset_name)

    t_aco = list(range(1, n_iter + 1))
    print("problem scale:", n_nodes)
    print("checkpoint:", ckpt_path)
    print("number of instances:", len(test_list))
    print("device:", 'cpu' if DEVICE == 'cpu' else DEVICE+"+cpu" )
    print("n_ants:", n_ants)
    print("seed:", seed)

    if ckpt_path is not None:
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1).to(DEVICE)
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        net = None
    results_list, duration = test(test_list, net, n_ants, t_aco, int_dist_list)
    results_df = pd.DataFrame(index=name_list, columns=["Length"])
    for name, results in zip(name_list, results_list):
        results_df.loc[name, :] = results[-1].item()

    print('total duration: ', duration)

    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = os.path.dirname(ckpt_path) if ckpt_path is not None else f'../pretrained/cvrp_nls/{args.nodes}/no_model'
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-vrplib{dataset_name}{n_nodes}-nants{n_ants}-niter{n_iter}-seed{seed}"
    results_df.to_csv(os.path.join(dirname, f"{result_filename}.csv"), index=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse_factor", type=int, default=5, help="k_sparse factor")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--n_iter", type=int, default=10, help="Number of iterations of ACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    ### Dataset
    parser.add_argument("--dataset", type=str, default="X", help="Dataset name")
    args = parser.parse_args()

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.path is not None and not os.path.isfile(args.path):
        print(f"Checkpoint file '{args.path}' not found!")
        exit(1)

    main(
        args.path,
        args.nodes,
        args.k_sparse_factor,
        args.n_ants,
        args.n_iter,
        not args.disable_guided_exp,
        args.seed,
        args.dataset,
    )
