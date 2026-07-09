import os
import random
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from net import Net
from aco import ACO
from utils import load_test_dataset

from typing import Tuple, List


EPS = 1e-10
tw_normalize = True

def validate_route(distance: torch.Tensor, tw: torch.Tensor, routes: List[torch.Tensor]) -> Tuple[bool, float]:
    length = 0.0
    valid = True
    current_time = 0.0
    visited = {0}
    for i in range(1, len(routes)):
        current_node = routes[i]
        visited.add(current_node)
        tw_start, tw_end = tw[current_node]
        travel_time = distance[routes[i-1], current_node]
        current_time = torch.max(current_time+travel_time, tw_start)
        if current_time > tw_end + 0.000001:
            valid = False
        length += travel_time
    length += distance[current_node, routes[0]]
    if len(visited) != distance.size(0):
        valid = False
    return valid, length


@torch.no_grad()
def infer_instance(model, pyg_data, tw, distances, positions, n_ants, t_aco_diff, tw_normalize=True, generate_simulated_mask=True,normalize_phe=False):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data, tw_normalize)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    aco = ACO(
        distances=distances,
        tw=tw,
        n_ants=n_ants,
        heuristic=heu_mat,
        device=DEVICE,
        local_search='lkh',
        positions=positions,
        generate_simulated_mask = generate_simulated_mask
    )

    # aco_base = ACO(
    #     distances=distances,
    #     tw=tw,
    #     n_ants=n_ants,
    #     heuristic=heu_mat,
    #     device=DEVICE,
    #     local_search='lkh',
    #     positions=positions,
    #     generate_simulated_mask = generate_simulated_mask,
    #     normalize_phe=normalize_phe,
    # )

    results = torch.zeros(size=(len(t_aco_diff),))
    diversities = torch.zeros(size=(len(t_aco_diff),))
    sol_infsb_nums = torch.zeros(size=(len(t_aco_diff),))
    ins_infsb_nums = torch.zeros(size=(len(t_aco_diff),))
    base_fsb_cost = torch.zeros(size=(len(t_aco_diff),))
    base_diversities = torch.zeros(size=(len(t_aco_diff),))
    base_sol_infsb_nums = torch.zeros(size=(len(t_aco_diff),))
    base_ins_infsb_nums = torch.zeros(size=(len(t_aco_diff),))

    for i, t in enumerate(t_aco_diff):
        # with each run, pheromone updates

        # without local search
        # base_fsb_cost[i], base_diversities[i], total_timeout = aco_base.run(t, local_search=False)# for this instance, we see the n_ants solutions
        # base_sol_infsb_nums[i] = (total_timeout!=0).sum()
        # base_ins_infsb_nums[i] = 0 if (total_timeout==0).any() else 1
        # path = aco_base.shortest_path
        # if (total_timeout == 0).any():
        #     valid, length = validate_route(distances, tw, path)
        #     assert (length - base_fsb_cost[i].item()) < 1e-4  # double_check
        #     if valid is False:
        #        print("invalid solution.")
        base_costs, base_timeout, _, _, base_path, out = aco.sample()
        if (base_timeout == 0).any():
            base_ins_infsb_nums[i] = 0
            base_fsb_cost[i], base_best_idx = base_costs[base_timeout == 0].min(dim=0)
            base_best_path = base_path.T[base_timeout == 0][base_best_idx]
            base_valid, base_length = validate_route(distances, tw, base_best_path)
            assert (base_length - base_fsb_cost[i].item()) < 1e-4  # double_check
            if base_valid is False:
                print("invalid baseline solution.")
        else:
            base_ins_infsb_nums[i] = 1
            base_fsb_cost[i] = float('inf')
        base_sol_infsb_nums[i] = (base_timeout != 0).sum()

        # with local search
        results[i], diversities[i], total_timeout = aco.run(t, local_search=True)
        # for this instance, we see the n_ants solutions
        sol_infsb_nums[i] = (total_timeout!=0).sum()
        ins_infsb_nums[i] = 0 if (total_timeout==0).any() else 1
        path = aco.shortest_path
        if (total_timeout == 0).any():
            valid, length = validate_route(distances, tw, path)
            assert (length - results[i].item()) < 1e-4  # double_check
            if valid is False:
               print("invalid solution.")

    return path, results, diversities, sol_infsb_nums, ins_infsb_nums, base_fsb_cost, base_sol_infsb_nums, base_ins_infsb_nums


@torch.no_grad()
def test(dataset, model, n_ants, t_aco, generate_simulated_mask=True, normalize_phe=False, result_path =None):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    sum_results = torch.zeros(size=(len(t_aco_diff),))
    sum_gaps = torch.zeros(size=(len(t_aco_diff),))
    ls_out = torch.zeros(size=(0, len(t_aco_diff),))
    sum_diversities = torch.zeros(size=(len(t_aco_diff),))
    sum_sol_infsb_num = torch.zeros(size=(len(t_aco_diff),))
    sum_ins_infsb_num = torch.zeros(size=(len(t_aco_diff),))
    sum_base_fsb_cost = torch.zeros(size=(len(t_aco_diff),))
    base_out = torch.zeros(size=(0, len(t_aco_diff)))
    # sum_base_diversities= torch.zeros(size=(len(t_aco_diff),))
    sum_base_sol_infsb_nums = torch.zeros(size=(len(t_aco_diff),))
    sum_base_ins_infsb_nums = torch.zeros(size=(len(t_aco_diff),))

    import pickle
    lkh = "/home/jieyi/gfacs-0403/gfacs-main/data/tsptw/lkh_tsptw500_zhang_uniform_1020.pkl"
    with open(lkh, 'rb') as file:
        lkh = pickle.load(file)
    opt = [x[0] / 100 for x in lkh]
    opt = torch.tensor(opt)
    # idx = 0
    idx = 5
    print("idx", idx)
    start = time.time()
    for pyg_data, tw, distances, positions in tqdm(dataset):
        path, results, diversities, sol_infsb_nums, ins_infsb_nums, base_fsb_cost, base_sol_infsb_nums, base_ins_infsb_nums = infer_instance(
            model, pyg_data, tw, distances, positions, n_ants, t_aco_diff, tw_normalize=tw_normalize, generate_simulated_mask=generate_simulated_mask,
            normalize_phe = normalize_phe,
        )
        try:
            paths = torch.cat([paths, path.unsqueeze(0)], dim=0)
        except:
            paths = path.unsqueeze(0)
        sum_results = torch.where(ins_infsb_nums == 0., sum_results + results, sum_results)
        gap = (results - opt[idx]) / opt[idx] * 100
        sum_gaps = torch.where(ins_infsb_nums == 0., sum_gaps + gap, sum_gaps)
        idx += 1
        ls_out = torch.cat([ls_out, results.unsqueeze(0)], dim = 0)
        sum_diversities += diversities
        sum_sol_infsb_num += sol_infsb_nums
        sum_ins_infsb_num += ins_infsb_nums
        sum_base_fsb_cost = torch.where(base_ins_infsb_nums == 0., sum_base_fsb_cost + base_fsb_cost, sum_base_fsb_cost)
        base_out = torch.cat([base_out, base_fsb_cost.unsqueeze(0)], dim=0) #128*10
        # sum_base_diversities += base_diversities
        sum_base_sol_infsb_nums += base_sol_infsb_nums
        sum_base_ins_infsb_nums += base_ins_infsb_nums
    end = time.time()

    if result_path is not None:
        out = torch.cat([base_out.unsqueeze(0), ls_out.unsqueeze(0)], dim=0) # 2*n_instances*10
        torch.save(out, result_path)
        torch.save(paths, result_path+"ls_best_paths.pt")



    return (sum_gaps / (len(dataset)-sum_ins_infsb_num),
        sum_results / (len(dataset)-sum_ins_infsb_num), sum_diversities / len(dataset),
            sum_sol_infsb_num / (len(dataset)*n_ants) * 100, sum_ins_infsb_num / len(dataset) *100,
            sum_base_fsb_cost/ (len(dataset)-sum_base_ins_infsb_nums),
            sum_base_sol_infsb_nums/(len(dataset)*n_ants)*100, sum_base_ins_infsb_nums / len(dataset) *100,
            end - start)


def main(val_dataset, ckpt_path, n_nodes, k_sparse, size=None, n_ants=100, n_iter=10, guided_exploration=False, generate_simulated_mask=False, seed=0, normalize_phe=False):
    test_list = load_test_dataset(val_dataset, n_nodes, k_sparse, DEVICE)
    test_list = test_list[5:(size or len(test_list))]
    print(">> start from", 5)
    # test_list = test_list[:(size or len(test_list))]

    t_aco = list(range(1, n_iter + 1))
    print("problem scale:", n_nodes)
    print("checkpoint:", ckpt_path)
    print("number of instances:", size)
    print("device:", 'cpu' if DEVICE == 'cpu' else DEVICE+"+cpu" )
    print("n_ants:", n_ants)
    print("seed:", seed)
    print("generate_simulated_mask:", generate_simulated_mask)
    print("normalize_phe:", normalize_phe)

    # Save result in directory that contains model_file
    filename = os.path.splitext(os.path.basename(ckpt_path))[0] if ckpt_path is not None else 'none'
    dirname = os.path.dirname(ckpt_path) if ckpt_path is not None else f'../pretrained/tsptw_nls/{args.nodes}/no_model'
    os.makedirs(dirname, exist_ok=True)

    result_filename = f"test_result_ckpt{filename}-tsptw{n_nodes}-ninst{size}-nants{n_ants}-niter{n_iter}-seed{seed}"
    result_file = os.path.join(dirname, result_filename + ".txt")
    result_path = os.path.join(dirname, result_filename + ".pt")

    if ckpt_path is not None:
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1).to(DEVICE)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        try:
            net.load_state_dict(checkpoint)
        except:
            net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net = None
    avg_gap, avg_cost, avg_diversity, sol_infsb_rate, ins_infsb_rate, avg_base_fsb_cost, base_sol_infsb_rate, base_ins_infsb_rate, duration = test(test_list, net, n_ants, t_aco, generate_simulated_mask, normalize_phe, result_path)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print(f"T={t}, Base: avg. fsb cost {avg_base_fsb_cost[i]}, solution-level infsb% {base_sol_infsb_rate[i]}, instance-level infsb% {base_ins_infsb_rate[i]}")
    for i, t in enumerate(t_aco):
        print(f"T={t}, avg_gap{avg_gap[i]}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}, "
              f"solution-level infsb% {sol_infsb_rate[i]}, instance-level infsb% {ins_infsb_rate[i]}")


    with open(result_file, "w") as f:
        f.write(f"problem scale: {n_nodes}\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"number of instances: {len(test_list)}\n")
        f.write(f"device: {'cpu' if DEVICE == 'cpu' else DEVICE+'+cpu'}\n")
        f.write(f"n_ants: {n_ants}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"total duration: {duration}\n")
        for i, t in enumerate(t_aco):
            f.write(f"T={t}, Base: avg. fsb cost {avg_base_fsb_cost[i]}, solution-level infsb% {base_sol_infsb_rate[i]},instance-level infsb% {base_ins_infsb_rate[i]}")
            f.write(f"T={t}, avg. cost {avg_cost[i]}, avg. diversity {avg_diversity[i]}, solution-level infsb% {sol_infsb_rate[i]}, instance-level infsb% {ins_infsb_rate[i]}\n")

    results = pd.DataFrame(columns=['T', 'avg_gap', 'avg_cost', 'avg_diversity', 'sol_infsb_rate', 'ins_infsb_rate', 'base_avg_fsb_cost', 'base_sol_infsb_rate', 'base_ins_infsb_rate'])
    results['T'] = t_aco
    results['avg_gap'] = avg_gap.cpu()
    results['avg_cost'] = avg_cost.cpu()
    results['avg_diversity'] = avg_diversity.cpu()
    results['sol_infsb_rate'] = sol_infsb_rate.cpu()
    results['ins_infsb_rate'] = ins_infsb_rate.cpu()
    results['base_avg_fsb_cost'] = avg_base_fsb_cost.cpu()
    results['base_sol_infsb_rate'] = base_sol_infsb_rate.cpu()
    results['base_ins_infsb_rate'] = base_ins_infsb_rate.cpu()
    results.to_csv(os.path.join(dirname, result_filename + ".csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, default=500, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    # parser.add_argument("-p", "--path", type=str, default="")
    # parser.add_argument("-p", "--path", type=str, default="")
    # parser.add_argument("-p", "--path", type=str, default="")
    # parser.add_argument("-p", "--path", type=str, default="")
    # parser.add_argument("-p", "--path", type=str, default="")
    parser.add_argument("-p", "--path", type=str, default="/home/jieyi/gfacs-0403/gfacs-main/pretrained/tsptw_nls/500/[tw500_1020_sl_10_10_2_5_188]tsptw500_sd2023/50.pt")
    # parser.add_argument("-p", "--path", type=str, default="/home/jieyi/gfacs-0403/gfacs-main/pretrained/tsptw_nls/500/[tw500_1020_sl_factor_norm_188]tsptw500_sd2023/50.pt")
    # parser.add_argument("-p", "--path", type=str, default="/home/jieyi/gfacs-0403/gfacs-main/pretrained/tsptw_nls/100/[tw100_1020_188]tsptw100_sd2023/50.pt")
    # parser.add_argument("-p", "--path", type=str, default="/home/jieyi/gfacs-0403/gfacs-main/pretrained/tsptw_nls/100/[tw100_1020_simulatedMask_188]tsptw100_sd2023/best.pt")
    # parser.add_argument("-p", "--path", type=str, default="/home/jieyi/gfacs-0403/gfacs-main/pretrained/tsptw_nls/500/[tw1020_500_188]tsptw500_sd2023/50.pt")
    # parser.add_argument("-p", "--path", type=str, default="/home/jieyi/gfacs-0403/gfacs-main/pretrained/tsptw_nls/500/[tw500_1020_simulatedMask_188]tsptw500_sd2023/50.pt", help="Path to checkpoint file")
    # parser.add_argument("--val_dataset", type=str, default="/home/jieyi/gfacs-0403/gfacs-main/data/tsptw/valDataset-100.pt")
    parser.add_argument("--val_dataset", type=str,default="/home/jieyi/gfacs-0403/gfacs-main/data/tsptw/valDataset-500-zhang-1020.pt")
    parser.add_argument("-s", "--size", type=int, default=10, help="Number of instances to test")
    parser.add_argument("-i", "--n_iter", type=int, default=10, help="Number of iterations of ACO to run")
    parser.add_argument("-n", "--n_ants", type=int, default=100, help="Number of ants")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:2" if torch.cuda.is_available() else "cpu"),
                        help="The device to train NNs")

    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='True for model w/o guided exploration.')
    ### Simulation
    parser.add_argument("--generate_simulated_mask",type=bool, default=True)
    parser.add_argument("--normalize_phe", type=bool, default=True)
    ### Seed
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 5

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.path is not None and not os.path.isfile(args.path):
        print(f"Checkpoint file '{args.path}' not found!")
        exit(1)

    main(
        args.val_dataset,
        args.path,
        args.nodes,
        args.k_sparse,
        args.size,
        args.n_ants,
        args.n_iter,
        not args.disable_guided_exp,
        args.generate_simulated_mask,
        args.seed,
        args.normalize_phe
    )
