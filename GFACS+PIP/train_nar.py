import os
import math
import random
import time

from tqdm import tqdm
import scipy
import numpy as np
import torch

import sys
from net import Net, ParNet, AR_ParNet
from aco import ACO
from utils import gen_pyg_data, load_val_dataset, gen_instance
from typing import Tuple, List

import wandb
wandb.login(key="d7c2a4d107302d1b34184fb17ca47aa6f84055ac")


EPS = 1e-10
T = 5
START_NODE = 0
tw_normalize = True
use_penalty = True
fsb_dist_only = True

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

def calculate_log_pb_uniform(paths: torch.Tensor):
    # paths.shape: (batch, max_tour_length)
    # paths are start with 0 and end with 0

    _pi1 = paths.detach().cpu().numpy()
    # shape: (batch, max_tour_length)

    n_nodes = np.count_nonzero(_pi1, axis=1)
    _pi2 = _pi1[:, 1:] - _pi1[:, :-1]
    n_routes = np.count_nonzero(_pi2, axis=1) - n_nodes
    _pi3 = _pi1[:, 2:] - _pi1[:, :-2]
    n_multinode_routes = np.count_nonzero(_pi3, axis=1) - n_nodes
    log_b_p = - scipy.special.gammaln(n_routes + 1) - n_multinode_routes * math.log(2)

    return torch.from_numpy(log_b_p).to(paths.device)


def train_instance(
        model,
        dual_decoder,
        optimizer,
        data,
        n_ants,
        generate_simulated_mask,
        normalize_phe,
        # safety_layer,
        cost_w=1.0,
        invtemp=1.0,
        guided_exploration=False,
        shared_energy_norm=False,
        beta=100.0,
        it=0,
        is_train_dual_decoder = False
    ):
    model.train()
    if dual_decoder is not None:
        dual_decoder.train()

    if is_train_dual_decoder:
        generate_simulated_mask = True
    else:
        generate_simulated_mask = False

    ##################################################
    # wandb
    _train_mean_cost = 0.0
    _train_min_cost = 0.0
    _train_sol_infsb_rate = 0.0
    _train_ins_infsb_rate = 0.0
    _train_mean_fsb_cost = 0.0
    _train_min_fsb_cost = 0.0
    _train_mean_cost_total = 0.0
    _train_min_cost_total = 0.0
    _train_mean_timeout = 0.0
    _train_min_timeout = 0.0
    _train_mean_timeout_nodes = 0.0
    _train_min_timeout_nodes = 0.0

    _train_mean_cost_nls = 0.0
    _train_min_cost_nls = 0.0
    _train_sol_infsb_rate_nls = 0.0
    _train_ins_infsb_rate_nls = 0.0
    _train_mean_fsb_cost_nls = 0.0
    _train_min_fsb_cost_nls = 0.0
    _train_mean_cost_nls_total = 0.0
    _train_min_cost_nls_total = 0.0
    _train_mean_timeout_nls = 0.0
    _train_min_timeout_nls = 0.0
    _train_mean_timeout_nodes_nls = 0.0
    _train_min_timeout_nodes_nls = 0.0
    _train_entropy = 0.0
    _logZ_mean = torch.tensor(0.0, device=DEVICE)
    _logZ_nls_mean = torch.tensor(0.0, device=DEVICE)
    ##################################################
    sum_loss = torch.tensor(0.0, device=DEVICE)
    sum_loss_sl = torch.tensor(0.0, device=DEVICE)
    sum_loss_nls = torch.tensor(0.0, device=DEVICE)
    count = 0
    fsb_count = 0
    fsb_nls_count = 0
    true_infsb, infsb_num, true_fsb, fsb_num = 0, 0, 0, 0

    for pyg_data, tw, distances, positions in data:
        if dual_decoder is not None:
            heu_vec, logZ, embedding = model(pyg_data, tw_normalize, return_logZ=True, return_embedding=True)
        else:
            heu_vec, logZ = model(pyg_data, tw_normalize, return_logZ=True)
        if isinstance(heu_vec, list):
            heu_vec, sl_mask = heu_vec
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS

        if guided_exploration:
            logZ, logZ_nls = logZ
        else:
            logZ = logZ[0]

        aco = ACO(
            distances=distances.to(DEVICE),
            tw=tw.to(DEVICE),
            n_ants=n_ants,
            heuristic=heu_mat.to(DEVICE),
            device=DEVICE,
            local_search='lkh',
            positions=positions,
            generate_simulated_mask = generate_simulated_mask,
            normalize_phe =normalize_phe,
            sl_mask = sl_mask,
            # dual_decoder = dual_decoder if dual_decoder else None,
            # embedding = embedding if dual_decoder else None,
            # pyg_data=pyg_data if dual_decoder else None,
        )
        
        if guided_exploration:
            costs_nls, log_probs, paths_nls, costs_raw, paths, timeout_nls, timeout_nodes_nls, timeout_raw, timeout_nodes_raw, out = aco.sample_local_search(invtemp, return_loss = is_train_dual_decoder, return_auc=generate_simulated_mask)
            if use_penalty:
                costs_raw_total = costs_raw + timeout_raw + timeout_nodes_raw
                costs_nls_total = costs_nls + timeout_nls + timeout_nodes_nls
                advantage_raw = (costs_raw_total - (costs_raw_total.mean() if shared_energy_norm else 0.0))
                advantage_nls = (costs_nls_total - (costs_nls_total.mean() if shared_energy_norm else 0.0))
            else:
                advantage_raw = (costs_raw - (costs_raw.mean() if shared_energy_norm else 0.0))
                advantage_nls = (costs_nls - (costs_nls.mean() if shared_energy_norm else 0.0))
            weighted_advantage = cost_w * advantage_nls + (1 - cost_w) * advantage_raw
        else:
            costs_raw, timeout_raw, timeout_nodes_raw, log_probs, paths = aco.sample(invtemp)
            if use_penalty:
                costs_raw_total = costs_raw + timeout_raw + timeout_nodes_raw
                advantage_raw = (costs_raw_total - (costs_raw_total.mean() if shared_energy_norm else 0.0))
            else:
                advantage_raw = (costs_raw - (costs_raw.mean() if shared_energy_norm else 0.0))
            weighted_advantage = advantage_raw

        ##################################################
        # Loss from paths before local search
        forward_flow = log_probs.sum(0) + logZ.expand(n_ants)
        backward_flow = calculate_log_pb_uniform(paths.T) - weighted_advantage.detach() * beta
        tb_loss = torch.pow(forward_flow - backward_flow, 2).mean()
        sum_loss += tb_loss
        if len(out) != 0:
            true_infsb0, infsb_num0, true_fsb0, fsb_num0, sl_loss = out
            sum_loss_sl += sl_loss
            true_infsb += true_infsb0
            infsb_num += infsb_num0
            true_fsb += true_fsb0
            fsb_num += fsb_num0
        ##################################################
        # Loss from paths after local search
        if guided_exploration:
            n_ants_nls = n_ants
            _, log_probs_nls, feasible_idx, _ = aco.gen_path(  # type: ignore
                require_prob=True,
                invtemp=1.0,  # invtemp is 1.0 here, otherwise gradients from offpolicy data will be overestimated
                paths=paths_nls,  # type: ignore
            )
            if len(feasible_idx) < n_ants:  # type: ignore
                paths_nls = paths_nls[:, feasible_idx]  # type: ignore
                costs_nls = costs_nls[feasible_idx]  # type: ignore
                timeout_nls = timeout_nls[feasible_idx]
                timeout_nodes_nls = timeout_nodes_nls[feasible_idx]
                costs_nls_total = costs_nls_total[feasible_idx]
                advantage_nls = advantage_nls[feasible_idx]  # type: ignore
                n_ants_nls = len(feasible_idx)  # type: ignore

            forward_flow_nls = log_probs_nls.sum(0) + logZ_nls.expand(n_ants_nls)  # type: ignore
            backward_flow_nls = calculate_log_pb_uniform(paths_nls.T) - advantage_nls.detach() * beta  # type: ignore
            tb_loss_nls = torch.pow(forward_flow_nls - backward_flow_nls, 2).mean()
            sum_loss_nls += tb_loss_nls

        count += 1
        if (timeout_raw == 0).any():
            fsb_count += 1
        if (timeout_nls == 0).any():
            fsb_nls_count += 1

        ##################################################
        # wandb
        if USE_WANDB:
            _train_mean_cost += costs_raw.mean().item()
            _train_min_cost += costs_raw.min().item()
            _train_sol_infsb_rate += ((timeout_raw != 0).sum() / n_ants).item()
            _train_ins_infsb_rate += (timeout_raw != 0).all().int().item()
            _train_mean_fsb_cost += costs_raw[timeout_raw==0].mean().item() if (timeout_raw==0).any() else 0.0
            _train_min_fsb_cost += costs_raw[timeout_raw==0].min().item() if (timeout_raw==0).any() else 0.0
            _train_mean_cost_total += costs_raw_total.mean().item()
            _train_min_cost_total += costs_raw_total.min().item()
            _train_mean_timeout += timeout_raw.mean().item()
            _train_min_timeout += timeout_raw.min().item()
            _train_mean_timeout_nodes += timeout_nodes_raw.float().mean().item()
            _train_min_timeout_nodes += timeout_nodes_raw.float().min().item()

            normed_heumat = heu_mat / heu_mat.sum(dim=1, keepdim=True)
            entropy = -(normed_heumat * torch.log(normed_heumat)).sum(dim=1).mean()
            _train_entropy += entropy.item()

            _logZ_mean += logZ
            if guided_exploration:
                _train_mean_cost_nls += costs_nls.mean().item()
                _train_min_cost_nls += costs_nls.min().item()
                _train_sol_infsb_rate_nls += ((timeout_nls != 0).sum() / n_ants).item()
                _train_ins_infsb_rate_nls += (timeout_nls != 0).all().int().item()
                _train_mean_fsb_cost_nls += costs_nls[timeout_nls == 0].mean().item() if (timeout_nls == 0).any() else 0.0
                _train_min_fsb_cost_nls += costs_nls[timeout_nls == 0].min().item() if (timeout_nls == 0).any() else 0.0
                _train_mean_cost_nls_total += costs_nls_total.mean().item()
                _train_min_cost_nls_total += costs_nls_total.min().item()
                _train_mean_timeout_nls += timeout_nls.mean().item()
                _train_min_timeout_nls += timeout_nls.min().item()
                _train_mean_timeout_nodes_nls += timeout_nodes_nls.float().mean().item()
                _train_min_timeout_nodes_nls += timeout_nodes_nls.float().min().item()
                _logZ_nls_mean += logZ_nls
        ##################################################

    sum_loss = sum_loss / count
    sum_loss_nls = sum_loss_nls / count if guided_exploration else torch.tensor(0.0, device=DEVICE)
    sum_loss_sl = sum_loss_sl /count
    loss = sum_loss + sum_loss_nls + sum_loss_sl * 10**(torch.floor(torch.log10(sum_loss.detach())))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer.step()

    ##################################################
    # wandb
    if USE_WANDB:
        wandb.log(
            {
                "train_mean_cost": _train_mean_cost / count,
                "train_min_cost": _train_min_cost / count,
                "train_sol_infsb_rate": _train_sol_infsb_rate/count * 100,
                "train_ins_infsb_rate": _train_ins_infsb_rate/count * 100,
                "train_mean_fsb_cost": (_train_mean_fsb_cost / fsb_count) if fsb_count > 0 else 0.0,
                "train_min_fsb_cost": (_train_min_fsb_cost / fsb_count) if fsb_count > 0 else 0.0,
                "train_mean_cost_total": _train_mean_cost_total / count,
                "train_min_cost_total": _train_min_cost_total / count,
                "train_mean_timeout": _train_mean_timeout / count,
                "train_min_timeout": _train_min_timeout / count,
                "train_mean_timeout_nodes": _train_mean_timeout_nodes / count,
                "train_min_timeout_nodes": _train_min_timeout_nodes / count,

                "train_mean_cost_nls": _train_mean_cost_nls / count,
                "train_min_cost_nls": _train_min_cost_nls / count,
                "train_sol_infsb_rate_nls": _train_sol_infsb_rate_nls / count * 100,
                "train_ins_infsb_rate_nls": _train_ins_infsb_rate_nls / count * 100,
                "train_mean_fsb_cost_nls": (_train_mean_fsb_cost_nls / fsb_nls_count) if fsb_nls_count > 0 else 0.0,
                "train_min_fsb_cost_nls": (_train_min_fsb_cost_nls / fsb_nls_count) if fsb_nls_count > 0 else 0.0,
                "train_mean_cost_nls_total": _train_mean_cost_nls_total / count,
                "train_min_cost_nls_total": _train_min_cost_nls_total / count,
                "train_mean_timeout_nls": _train_mean_timeout_nls / count,
                "train_min_timeout_nls": _train_min_timeout_nls / count,
                "train_mean_timeout_nodes_nls": _train_mean_timeout_nodes_nls / count,
                "train_min_timeout_nodes_nls": _train_min_timeout_nodes_nls / count,

                "train_entropy": _train_entropy / count,
                "train_loss": sum_loss.item(),
                "train_loss_nls": sum_loss_nls.item(),
                "cost_w": cost_w,
                "invtemp": invtemp,
                "logZ": _logZ_mean.item() / count,
                "logZ_nls": _logZ_nls_mean.item() / count,
                "beta": beta,

                "train_loss_sl": sum_loss_sl.item(),
                "infsb_auc_sl": true_infsb / infsb_num * 100,
                "infsb_num_sl": infsb_num,
                "fsb_auc_sl": true_fsb / fsb_num * 100,
                "fsb_num_sl": fsb_num,

            },
            step=it,
        )

    ##################################################
    infsb_auc_sl = true_infsb / infsb_num * 100
    fsb_auc_sl = true_fsb / fsb_num * 100
    print(sum_loss_sl, sum_loss, _train_min_fsb_cost,  sum_loss_nls,  _train_min_fsb_cost_nls, infsb_auc_sl, infsb_num , fsb_auc_sl, fsb_num)

def infer_instance(model, dual_decoder, pyg_data, tw, distances, positions, n_ants, tw_normalize, generate_simulated_mask, normalize_phe, gen_sl_mask):
    model.eval()
    heu_vec = model(pyg_data, tw_normalize, return_embedding=gen_sl_mask)
    if gen_sl_mask:
        heu_vec, embedding = heu_vec
    if isinstance(heu_vec, list):
        heu_vec, sl_mask= heu_vec
        sl_mask = model.reshape(pyg_data, sl_mask) + EPS
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS

    # if dual_decoder is not None:
    #     dual_decoder.eval()

    aco = ACO(
        distances=distances,
        tw=tw,
        n_ants=n_ants,
        heuristic=heu_mat,
        device=DEVICE,
        local_search='lkh',
        positions=positions,
        generate_simulated_mask = generate_simulated_mask,
        normalize_phe =normalize_phe,
        sl_mask = sl_mask
        # dual_decoder = dual_decoder if gen_sl_mask else None,
        # embedding = embedding if gen_sl_mask else None,
        # pyg_data=pyg_data if gen_sl_mask else None,
    )

    costs, timeout_penalty, timeout_nodes_penalty, _, path, out = aco.sample(return_auc=generate_simulated_mask)
    fsb_cost = costs[timeout_penalty==0]
    if fsb_dist_only:
        if (timeout_penalty==0).any():
            baseline_fsb_cost, best_sample_fsb_cost = fsb_cost.mean().item(), fsb_cost.min().item()
            _, best_idx = fsb_cost.min(dim=0)
            path = path.T[timeout_penalty==0][best_idx]
            valid, length = validate_route(distances, tw, path)
            assert (length - best_sample_fsb_cost) < 1e-4  # double_check
            if valid is False:
                print("invalid solution on baseline.")
        else:
            baseline_fsb_cost, best_sample_fsb_cost = float('inf'), float('inf')

    baseline_cost, best_sample_cost = costs.mean().item(), costs.min().item()
    baseline_timeout, best_sample_timeout = timeout_penalty.mean().item(), timeout_penalty.min().item()
    baseline_timeout_nodes, best_sample_timeout_nodes = timeout_nodes_penalty.float().mean().item(), timeout_nodes_penalty.min().item()
    baseline_sol_infsb_rate = ((timeout_penalty != 0).sum() / n_ants).item()
    baseline_ins_infsb_rate = (timeout_penalty != 0).all().int().item()


    best_aco_1, diversity_1, total_timeout_1 = aco.run(n_iterations=1)
    sol_infsb_rate_aco_1 = ((total_timeout_1 != 0).sum() / n_ants).item()
    ins_infsb_rate_aco_1 = (total_timeout_1 != 0).all().int().item()
    if aco.shortest_path is not None:
        path = aco.shortest_path
        valid, length = validate_route(distances, tw, path)
        assert (length - best_aco_1) < 1e-4  # double_check
        if valid is False:
            print("invalid solution on aco 1.")

    best_aco_T, diversity_T, total_timeout_T = aco.run(n_iterations=T - 1)
    sol_infsb_rate_aco_T = ((total_timeout_T != 0).sum() / n_ants).item()
    ins_infsb_rate_aco_T = (total_timeout_T != 0).all().int().item()
    if aco.shortest_path is not None:
        path = aco.shortest_path
        valid, length = validate_route(distances, tw, path)
        assert (length - best_aco_T) < 1e-4  # double_check
        if valid is False:
            print("invalid solutionon aco T.")

    return np.array([baseline_cost, best_sample_cost, baseline_fsb_cost, best_sample_fsb_cost,
                     best_aco_1, best_aco_T, diversity_1, diversity_T,
                     baseline_timeout, best_sample_timeout, baseline_timeout_nodes, best_sample_timeout_nodes,
                     baseline_sol_infsb_rate, baseline_ins_infsb_rate,
                     sol_infsb_rate_aco_1, ins_infsb_rate_aco_1, sol_infsb_rate_aco_T, ins_infsb_rate_aco_T,
                     ]) ,out


def generate_traindata(count, n_node, k_sparse, tw_type, tw_duration):
    for _ in range(count):
        instance = tw, dist, position = gen_instance(1, n_node, tw_type, tw_duration, normalize=True, device=DEVICE)
        yield gen_pyg_data(tw, position, dist, k_sparse, device=DEVICE, start_node=0), *instance


def train_epoch(
    n_node,
    tw_type,
    tw_duration,
    k_sparse,
    n_ants,
    epoch,
    steps_per_epoch,
    net,
    dual_decoder,
    optimizer,
    generate_simulated_mask,
    normalize_phe,
    batch_size=1,
    cost_w=0.98,
    invtemp=1.0,
    guided_exploration=False,
    shared_energy_norm=False,
    beta=100.0,
    is_train_dual_decoder=False
):
    for i in tqdm(range(steps_per_epoch), desc="Train"):
        it = (epoch - 1) * steps_per_epoch + i
        data = generate_traindata(batch_size, n_node, k_sparse, tw_type, tw_duration)
        train_instance(net,  dual_decoder, optimizer, data, n_ants, generate_simulated_mask,normalize_phe, cost_w,
                       invtemp, guided_exploration, shared_energy_norm, beta, it, is_train_dual_decoder)

@torch.no_grad()
def validation(val_list, n_ants, net, dual_decoder, epoch, steps_per_epoch, generate_simulated_mask, normalize_phe, gen_sl_mask):
    stats = []
    outs = []
    print(len(val_list))
    for data, tw, distances, positions in tqdm(val_list, desc="Val"): # one by one
        stat, out = infer_instance(net, dual_decoder, data, tw, distances, positions, n_ants, tw_normalize=tw_normalize,
                       generate_simulated_mask=generate_simulated_mask, normalize_phe=normalize_phe,
                       gen_sl_mask=gen_sl_mask)
        stats.append(stat)
        outs.append(out)
    avg_stats = []
    for i in range(len(stats[0])):
        cur_col = np.stack(stats)[:, i]
        avg_stats.append(cur_col[~np.isinf(cur_col)].mean(0).item())

    if len(val_list) == 10000:
        baseline = np.stack(stats)[:, 3]
        aco_1 = np.stack(stats)[:, 4]
        aco_T = np.stack(stats)[:, 5]
        lkh = "~/Routing-Anything-main/data/TSPTW/lkh_tsptw100_zhang_uniform_1020.pkl"
        with open(lkh, 'rb') as file:
            lkh = pickle.load(file)
        gaps_bs, gaps_1, gaps_T = [], [], []
        for i in range(10000):
            if not np.isinf(baseline[i]):
                gap = (baseline[i] - lkh[i][0]) / lkh[i][0]
                gaps_bs.append(gap)
            if not np.isinf(aco_1[i]):
                gap = (aco_1[i] - lkh[i][0]) / lkh[i][0]
                gaps_1.append(gap)
            if not np.isinf(aco_T[i]):
                gap = (aco_T[i] - lkh[i][0]) / lkh[i][0]
                gaps_T.append(gap)
        val_gap_baseline = np.mean(gaps_bs).item()
        val_gao_aco_1 = np.mean(gaps_1).item()
        val_gap_aco_T = np.mean(gaps_T).item()
        print(f"Gap: baseline {val_gap_baseline}; aco1: {val_gao_aco_1}, acoT: {val_gap_aco_T}")
        if USE_WANDB:
            wandb.log(
                {
                    "val_gap_baseline": val_gap_baseline,
                    "val_gao_aco_1": val_gao_aco_1,
                    "val_gap_aco_T": val_gap_aco_T,
                },
                step=epoch * steps_per_epoch,
            )

    infsb_num = 0
    fsb_num = 0
    true_infsb = 0
    true_fsb = 0
    if len(outs[0]) != 0 :
        for out in outs:
            true_infsb0, infsb_num0, true_fsb0, fsb_num0 = out
            infsb_num += infsb_num0
            fsb_num += fsb_num0
            true_infsb += true_infsb0
            true_fsb += true_fsb0

    ##################################################
    print(f"epoch {epoch}:", avg_stats)
    # Wandb
    # baseline_cost, best_sample_cost, baseline_fsb_cost, best_sample_fsb_cost,
    # best_aco_1, best_aco_T, diversity_1, diversity_T,
    # baseline_timeout, best_sample_timeout, baseline_timeout_nodes, best_sample_timeout_nodes,
    # baseline_sol_infsb_rate, baseline_ins_infsb_rate,
    # sol_infsb_rate_aco_1, ins_infsb_rate_aco_1, sol_infsb_rate_aco_T, ins_infsb_rate_aco_T
    if USE_WANDB:
        wandb.log(
            {
                "val_baseline": avg_stats[0],
                "val_best_sample_cost": avg_stats[1],
                "val_baseline_fsb_cost": avg_stats[2],
                "val_best_sample_fsb_cost": avg_stats[3],
                "val_best_aco_1": avg_stats[4],
                "val_best_aco_T": avg_stats[5],
                "val_diversity_1": avg_stats[6],
                "val_diversity_T": avg_stats[7],
                "val_baseline_timeout": avg_stats[8],
                "val_best_sample_timeout": avg_stats[9], # minimum timeout
                "val_baseline_timeout_nodes": avg_stats[10],
                "val_best_sample_timeout_nodes": avg_stats[11], # minimum timeout nodes
                "val_baseline_sol_infsb_rate": avg_stats[12]* 100,
                "val_baseline_ins_infsb_rate": avg_stats[13]* 100,
                "val_sol_infsb_rate_aco_1": avg_stats[14] * 100,
                "val_ins_infsb_rate_aco_1": avg_stats[15]* 100,
                "val_sol_infsb_rate_aco_T": avg_stats[16]* 100,
                "val_ins_infsb_rate_aco_T": avg_stats[17]* 100,
                "val_epoch": epoch,
                "val_infsb_auc": true_infsb / infsb_num * 100,
                "val_infsb_num": infsb_num,
                "val_fsb_auc": true_fsb / fsb_num * 100,
                "val_fsb_num": fsb_num,
            },
            step=epoch * steps_per_epoch,
        )
    ##################################################

    return avg_stats[5]


def train(
        n_nodes,
        tw_type,
        tw_duration,
        k_sparse,
        n_ants,
        n_val_ants,
        steps_per_epoch,
        epochs,
        generate_simulated_mask,
        normalize_phe,
        lr=1e-4,
        batch_size=3,
        val_size=None,
        val_interval=5,
        pretrained=None,
        savepath="../pretrained/tsptw_nls",
        run_name="",
        cost_w_schedule_params=(0.5, 1.0, 5),  # (cost_w_min, cost_w_max, cost_w_flat_epochs)
        invtemp_schedule_params=(0.8, 1.0, 5),  # (invtemp_min, invtemp_max, invtemp_flat_epochs)
        guided_exploration=False,
        shared_energy_norm=False,
        beta_schedule_params=(50, 500, 5),  # (beta_min, beta_max, beta_flat_epochs)
        args = None
    ):
    print("generate_simulated_mask:", generate_simulated_mask)
    print("normalize_phe:", normalize_phe)
    print("dual_decoder:", args.dual_decoder)

    savepath = os.path.join(savepath, str(n_nodes), run_name)
    os.makedirs(savepath, exist_ok=True)

    net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1, dual_decoder=args.dual_decoder).to(DEVICE)
    # dual_decoder = None
    # if args.dual_decoder:
    #     lazy_model = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1).to(DEVICE)
    #     if args.dual_decoder_type == "NAR":
    #         dual_decoder = ParNet().to(DEVICE)
    #     elif args.dual_decoder_type == "AR":
    #         dual_decoder = AR_ParNet().to(DEVICE)
    if pretrained:
        net.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)

    val_list_total = load_val_dataset(n_nodes, tw_type, tw_duration, k_sparse, DEVICE, start_node=START_NODE)
    val_list = val_list_total[:(val_size or len(val_list_total))]
    dual_decoder = None
    best_result = validation(val_list, n_val_ants, net, dual_decoder, 0, steps_per_epoch, generate_simulated_mask, normalize_phe, gen_sl_mask=args.dual_decoder)

    sum_time = 0
    for epoch in range(1, epochs + 1):
        # Cost Weight Schedule
        cost_w_min, cost_w_max, cost_w_flat_epochs = cost_w_schedule_params
        cost_w = cost_w_min + (cost_w_max - cost_w_min) * min((epoch - 1) / (epochs - cost_w_flat_epochs), 1.0)

        # Heatmap Inverse Temperature Schedule
        invtemp_min, invtemp_max, invtemp_flat_epochs = invtemp_schedule_params
        invtemp = invtemp_min + (invtemp_max - invtemp_min) * min((epoch - 1) / (epochs - invtemp_flat_epochs), 1.0)

        # Beta Schedule
        beta_min, beta_max, beta_flat_epochs = beta_schedule_params
        beta = beta_min + (beta_max - beta_min) * min(math.log(epoch) / math.log(epochs - beta_flat_epochs), 1.0)

        if epoch in args.load_sl_epoch_list:
            lazy_checkpoint = {"last_epoch": "{}.pt".format(epoch - 1),
                               "train_fsb_bsf": "fsb_accuracy_bsf.pt",
                               "train_infsb_bsf": "infsb_accuracy_bsf.pt",
                               "train_accuracy_bsf": "accuracy_bsf.pt"}
            checkpoint_fullname = os.path.join(savepath, lazy_checkpoint[args.load_which_piggy])
            lazy_model.load_state_dict(torch.load(checkpoint_fullname, map_location=DEVICE))
        start = time.time()
        train_epoch(
            n_nodes,
            tw_type,
            tw_duration,
            k_sparse,
            n_ants,
            epoch,
            steps_per_epoch,
            net,
            dual_decoder,
            optimizer,
            generate_simulated_mask,
            normalize_phe,
            batch_size,
            cost_w,
            invtemp,
            guided_exploration,
            shared_energy_norm,
            beta,
            is_train_dual_decoder = epoch in args.train_sl_epoch_list
        )
        sum_time += time.time() - start

        if epoch % val_interval == 0:
            torch.save(net.state_dict(), os.path.join(savepath, f"{epoch}.pt"))
            if args.dual_decoder:
                generate_simulated_mask = (epoch in args.train_sl_epoch_list)
            else:
                generate_simulated_mask = generate_simulated_mask
            curr_result = validation(val_list, n_val_ants, net, epoch, steps_per_epoch, generate_simulated_mask, normalize_phe, gen_sl_mask=args.dual_decoder)
            # curr_result = validation(val_list if epoch!=epochs else val_list_total, n_val_ants, net, epoch, steps_per_epoch, generate_simulated_mask, normalize_phe)
            if curr_result < best_result:
                torch.save(net.state_dict(), os.path.join(savepath, f'best.pt'))
                best_result = curr_result

        scheduler.step()

    print('\ntotal training duration:', sum_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", metavar='N', type=int, default=100, help="Problem scale")
    parser.add_argument('--tw_type', type=str, default="zhang", choices=["da_silva", "cappart", "zhang", "random"])
    parser.add_argument('--tw_duration', type=str, default="1020", choices=["5075", "75100", "2550", "5075", "random", "curriculum"])
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=5e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str, 
                        default=("cuda:2" if torch.cuda.is_available() else "cpu"),
                        help="The device to train NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-a", "--ants", type=int, default=20, help="Number of ants (in ACO algorithm)")
    parser.add_argument("-va", "--val_ants", type=int, default=100, help="Number of ants for validation")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Epochs to run")
    parser.add_argument("-v", "--val_size", type=int, default=1, help="Number of instances for validation")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/tsptw_nls",
                        help="The directory to store checkpoints")
    parser.add_argument("--val_interval", type=int, default=5, help="The interval to validate model")
    ### Logging
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="debug", help="Run name")
    ### invtemp
    parser.add_argument("--invtemp_min", type=float, default=0.8, help='Inverse temperature min for GFACS')
    parser.add_argument("--invtemp_max", type=float, default=1.0, help='Inverse temperature max for GFACS')
    parser.add_argument("--invtemp_flat_epochs", type=int, default=5, help='Inverse temperature flat epochs for GFACS')
    ### GFACS
    parser.add_argument("--disable_guided_exp", action='store_true', help='Disable guided exploration for GFACS')
    parser.add_argument("--disable_shared_energy_norm", action='store_true', help='Disable shared energy normalization for GFACS')
    parser.add_argument("--beta_min", type=float, default=None, help='Beta min for GFACS')
    parser.add_argument("--beta_max", type=float, default=None, help='Beta max for GFACS')
    parser.add_argument("--beta_flat_epochs", type=int, default=5, help='Beta flat epochs for GFACS')
    ### Energy Reshaping
    parser.add_argument("--cost_w_min", type=float, default=None, help='Cost weight min for GFACS')
    parser.add_argument("--cost_w_max", type=float, default=1.0, help='Cost weight max for GFACS')
    parser.add_argument("--cost_w_flat_epochs", type=int, default=5, help='Cost weight flat epochs for GFACS')
    ### Simulation
    parser.add_argument("--generate_simulated_mask", action="store_true", default=True)
    parser.add_argument("--normalize_phe", action="store_true")
    # feasible learning
    parser.add_argument('--dual_decoder', type=bool, default=True)
    parser.add_argument('--dual_decoder_type', type=str, default="NAR", choices=['NAR', 'AR'])

    parser.add_argument('--simulation_stop_epoch', type=int, default=10)
    parser.add_argument('--piggy_update_interval', type=int, default=10)
    parser.add_argument('--piggy_update_epoch', type=int, default=2)
    parser.add_argument('--piggy_last_growup', type=int, default=5)
    parser.add_argument('--piggy_save', type=str, default="epoch")
    parser.add_argument('--load_which_piggy', type=str, default="train_fsb_bsf", choices=["last_epoch", "train_fsb_bsf", "train_infsb_bsf", "train_accuracy_bsf"])
    parser.add_argument('--lazy_checkpoint', type=str, default=None)
    parser.add_argument('--sl_loss', type=str, default="BCEWithLogitsLoss", choices=["BCEWithLogitsLoss", "BCELoss", "FL", "CE"], help="FL: focal loss; CE: cross entropy loss")
    parser.add_argument('--label_balance_sampling', type=bool, default=True)
    parser.add_argument('--fast_label_balance', type=bool, default=True)
    parser.add_argument('--fast_weight', type=bool, default=True)
    parser.add_argument('--decision_boundary', type=float, default=0.5)
    parser.add_argument('--dislocation_start', type=int, default=-1, help="-1 means deactivating")
    # parser.add_argument()
    ### Seed
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")

    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 5

    if args.beta_min is None:
        beta_min_map = {100: 200, 200: 500, 400: 500, 500: 500, 1000: 2000 if args.pretrained is None else 500}
        args.beta_min = beta_min_map[args.nodes]
    if args.beta_max is None:
        beta_max_map = {100: 1000, 200: 2000, 400: 2000, 500: 2000, 1000: 2000}
        args.beta_max = beta_max_map[args.nodes]

    if args.cost_w_min is None:
        args.cost_w_min = 0.5 if args.pretrained is None else 0.8

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    USE_WANDB = not args.disable_wandb
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

    ##################################################
    # wandb
    run_name = f"[{args.run_name}]" if args.run_name else ""
    run_name += f"tsptw{args.nodes}_sd{args.seed}"
    pretrained_name = (
        args.pretrained.replace("../pretrained/tsptw_nls/", "").replace("/", "_").replace(".pt", "")
        if args.pretrained is not None else None
    )
    run_name += f"{'' if pretrained_name is None else '_fromckpt-'+pretrained_name}"
    if USE_WANDB:
        wandb.init(project=f"gfacs-tsptw_nls", name=run_name)
        wandb.config.update(args)
        wandb.config.update({"T": T, "model": "GFACS"})
    ##################################################
    
    if args.dual_decoder:
        args.train_sl_epoch_list = list(range(args.simulation_stop_epoch))
        for start in range(args.piggy_update_interval, args.epochs+1, args.piggy_update_interval):
            args.train_sl_epoch_list.extend(range(start - args.piggy_update_epoch , start))

        if args.piggy_last_growup > args.piggy_update_epoch:
            args.train_sl_epoch_list.extend(range(args.epochs - args.piggy_last_growup, args.epochs))

        args.load_sl_epoch_list = [args.simulation_stop_epoch] + list(range(0, args.epochs - args.piggy_last_growup, args.piggy_update_interval))[1:]

        args.train_sl_epoch_list = [ i+ 1 for i in args.train_sl_epoch_list]
        args.load_sl_epoch_list = [i + 1 for i in args.load_sl_epoch_list]

        args.accuracy_bsf, args.infsb_accuracy_bsf, args.fsb_accuracy_bsf = 0., 0., 0.
        args.is_train_double_head_decoder = True
    else:
        args.is_train_double_head_decoder = False

    train(
        args.nodes,
        args.tw_type,
        args.tw_duration,
        args.k_sparse,
        args.ants,
        args.val_ants,
        args.steps,
        args.epochs,
        generate_simulated_mask=args.generate_simulated_mask,
        normalize_phe = args.normalize_phe,
        lr=args.lr,
        batch_size=args.batch_size,
        val_size=args.val_size,
        val_interval=args.val_interval,
        pretrained=args.pretrained,
        savepath=args.output,
        run_name=run_name,
        cost_w_schedule_params=(args.cost_w_min, args.cost_w_max, args.cost_w_flat_epochs),
        invtemp_schedule_params=(args.invtemp_min, args.invtemp_max, args.invtemp_flat_epochs),
        guided_exploration=(not args.disable_guided_exp),
        shared_energy_norm=(not args.disable_shared_energy_norm),
        beta_schedule_params=(args.beta_min, args.beta_max, args.beta_flat_epochs),
        args = args
    )
