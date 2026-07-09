import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from collections import namedtuple
from data_generator import generate_tsptw_data

max_tw_gap=10
max_tw_size=100


def gen_instance(batch_size, problem_size, tw_type, tw_duration, normalize, device, coord_factor=100):
    """
    Implements data-generation method as described by Cappart et al. (2021), Da Silva et al. (2010), Zhang et al. (2020),
                                                        Kool et al. (2022), Chen et al. (2024)
    * Quentin Cappart, Thierry Moisan and Louis-Martin Rousseau et al., Combining reinforcement learning and constraint programming for combinatorial optimization[C]. AAAI. 2021.
    * Rodrigo Ferreira Da Silva and Sebasti´an Urrutia. A general vns heuristic for the traveling salesman problem with time windows. Discrete Optimization, 7(4):203–211, 2010.
    * Rongkai Zhang, Anatolii Prokhorchuk, and Justin Dauwels. Deep reinforcement learning for traveling salesman problem with time windows and rejections. IJCNN, 2020.
    * Wouter Kool, et al. Deep policy dynamic programming for vehicle routing problems. International conference on integration of constraint programming, artificial intelligence, and operations research, 2022.
    * Jingxiao Chen, et al. Looking Ahead to Avoid Being Late: Solving Hard-Constrained Traveling Salesman Problem. arXiv preprint arXiv:2403.05318 (2024).
    """
    # locations = torch.rand(size=(n, 2), device=device, dtype=torch.double)
    # distances = gen_distance_matrix(locations)


    if tw_type in ["cappart", "da_silva"]:
        # Taken from DPDP (Kool et. al)
        # Taken from https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/tsptw/environment/tsptw.py
        """
        :param problem_size: number of cities
        :param grid_size (=1): x-pos/y-pos of cities will be in the range [0, grid_size]
        :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
        :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
        :return: a feasible TSPTW instance randomly generated using the parameters
        """
        node_xy = torch.rand(size=(batch_size, problem_size, 2), device=device, dtype=torch.double) * coord_factor  # (batch, problem, 2)
        travel_time = torch.cdist(node_xy, node_xy,p=2)  # , compute_mode='donot_use_mm_for_euclid_dist')  # (batch, problem, problem)

        random_solution = torch.arange(1, problem_size).repeat(batch_size, 1)
        idx = [torch.randperm(random_solution.size(1)) for _ in range(batch_size)]
        out = torch.zeros(size=(0, problem_size - 1)).long()
        for i in range(batch_size):
            out = torch.cat([out, random_solution[i][idx[i]].unsqueeze(0)], dim=0)
        zeros = torch.zeros(size=(batch_size, 1)).long()
        random_solution = torch.cat([zeros, out], dim=1)

        time_windows = torch.zeros((batch_size, problem_size, 2))
        time_windows[:, 0, :] = torch.tensor([0, 1000. * coord_factor]).repeat(batch_size, 1)

        total_dist = torch.zeros(batch_size)
        for i in range(1, problem_size):

            prev_city = random_solution[:, i - 1]
            cur_city = random_solution[:, i]

            cur_dist = travel_time[torch.arange(batch_size), prev_city, cur_city]

            tw_lb_min = time_windows[torch.arange(batch_size), prev_city, 0] + cur_dist
            total_dist += cur_dist

            # print(tw_type)
            if tw_type == "da_silva":
                # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
                rand_tw_lb = torch.rand(batch_size) * (max_tw_size / 2) + (total_dist - max_tw_size / 2)
                rand_tw_ub = torch.rand(batch_size) * (max_tw_size / 2) + total_dist
            elif tw_type == "cappart":
                # Cappart et al. style 'propagates' the time windows resulting in little overlap / easier instances
                rand_tw_lb = torch.rand(batch_size) * (max_tw_gap) + tw_lb_min
                rand_tw_ub = torch.rand(batch_size) * (max_tw_size) + rand_tw_lb

            time_windows[torch.arange(batch_size), cur_city, :] = torch.cat(
                [rand_tw_lb.unsqueeze(1), rand_tw_ub.unsqueeze(1)], dim=1)
    elif tw_type == "zhang":
        TSPTW_SET = namedtuple("TSPTW_SET",
                               ["node_loc",  # Node locations 1
                                "node_tw",  # node time windows 5
                                "durations",  # service duration per node 6
                                "service_window",  # maximum of time units 7
                                "time_factor", "loc_factor"])
        # tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=1.42,
        #                          tw_type="naive/hard", tw_duration=self.tw_duration) # 1.42 = sqrt()
        tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=problem_size * 55,
                                 tw_type="naive/hard", tw_duration=tw_duration)
        node_xy = torch.tensor(tw.node_loc).float()
        travel_time = torch.cdist(node_xy, node_xy, p=2)
        time_windows = torch.tensor(tw.node_tw)
    elif tw_type == "random":
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)
        travel_time = torch.cdist(node_xy, node_xy, p=2)
        service_window = int(1.42 * problem_size)
        tw_start = torch.rand(size=(batch_size, problem_size, 1)) * (service_window / 2)
        episilon = ((torch.rand(size=(batch_size, problem_size, 1)) * 0.8) + 0.1) * (service_window / 2)  # [0.1,0.9]
        tw_end = tw_start + episilon
        # Normalize as in DPDP (Kool et. al)
        # Upper bound for depot = max(node ub + dist to depot), to make this tight
        tw_start[:, 0, 0] = 0.
        tw_end[:, 0, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:, 0]).max(dim=1)[0]
        time_windows = torch.cat([tw_start, tw_end], dim=-1)
    else:
        raise NotImplementedError

    if normalize:
        # Normalize as in DPDP (Kool et. al)
        node_xy = node_xy / coord_factor  # Normalize
        travel_time = travel_time / coord_factor
        # Normalize same as coordinates to keep unit the same, not that these values do not fall in range [0,1]!
        # Todo: should we additionally normalize somehow, e.g. by expected makespan/tour length?
        time_windows = time_windows / coord_factor
        # Upper bound for depot = max(node ub + dist to depot), to make this tight
        time_windows[:, 0, 1] = (travel_time[:, 0, 1:]/coord_factor + time_windows[:, 1:, 1]).max(dim=-1)[0]
        # nodes_timew = nodes_timew / nodes_timew[0, 1]

    # travel_time[torch.arange(problem_size), torch.arange(problem_size)] = 1e9 # note here
    diag_mask = torch.eye(problem_size, problem_size).unsqueeze(0) * (1e9)
    travel_time += diag_mask

    if batch_size == 1:
        return time_windows[0], travel_time[0], node_xy[0]
    return time_windows, travel_time, node_xy


def gen_distance_matrix(tsp_coordinates):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2, dtype=torch.double)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10  # note here
    return distances


def gen_pyg_data(tw, position, distances, k_sparse, device, start_node=None):
    '''
    Args:
        tw: torch tensor [n_nodes,2] for time window
        distances: torch tensor [n_nodes, n_nodes] for distance matrix
        k_sparse: int, number of edges to keep for each node
        start_node: int, index of the start node, if None, use random start node
    Returns:
        pyg_data: pyg Data instance
    '''
    n_nodes = tw.size(0)
    # sparsify
    # part 1: k nearest distance (n_nodes * k_sparse)
    topk_values1, topk_indices1 = torch.topk(distances, k=k_sparse, dim=1, largest=False)
    edge_index1 = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(device), repeats=k_sparse),
        torch.flatten(topk_indices1)
        ])
    # edge_attr1 = topk_values1.reshape(-1, 1)
    # part 2: k nearest tw_start for start node (if any) (1 * k_sparse)
    if start_node is not None:
        start_node_tw_start = tw[start_node, 0]
        tw_start_differences = tw[:, 0] - start_node_tw_start
        tw_start_differences[tw_start_differences <= 0] = float('inf')
        topk_values2, topk_indices2 = torch.topk(tw_start_differences, k=k_sparse, largest=False)
        edge_index2 = torch.stack([
            torch.repeat_interleave(torch.tensor(start_node).to(device), repeats=k_sparse),
            torch.flatten(topk_indices2)
        ])
        # edge_attr2 = topk_values2.reshape(-1, 1)
    # part 3: k tw overlap most ((n_nodes-1) * k_sparse)
    # overlap = min(end_A, end_B) - max(start_A, start_B)
    if start_node is not None:
        assert start_node==0, "only start_node=0 is supported!"
        tw_remove_start_node = torch.cat([tw[:start_node,:], tw[start_node+1:,:]], dim=0)
        start_index = 1
    else:
        tw_remove_start_node = tw
        start_index = 0
    start_times = tw_remove_start_node[:, 0].unsqueeze(1).expand(-1, n_nodes-1)
    end_times = tw_remove_start_node[:, 1].unsqueeze(1).expand(-1, n_nodes-1)
    start_max = torch.max(start_times, start_times.transpose(0, 1))
    end_min = torch.min(end_times, end_times.transpose(0, 1))
    overlap_matrix = torch.clamp(end_min - start_max, min=0)
    overlap_matrix.fill_diagonal_(0) # ignore self
    topk_values3, topk_indices3 = torch.topk(overlap_matrix, k=k_sparse, dim=1)
    topk_indices3 += 1 # since we remove the first node (start node) in overlap_matrix
    edge_index3 = torch.stack([
        torch.repeat_interleave(torch.arange(start_index, n_nodes).to(device), repeats=k_sparse),
        torch.flatten(topk_indices3)
        ])
    # edge_attr3 = topk_values3.reshape(-1, 1)

    if start_node is not None:
        edge_index = torch.concat([edge_index1, edge_index2, edge_index3], dim=1)
    else:
        edge_index = torch.concat([edge_index1, edge_index3], dim=1)
    # # remove the duplicated edge_index
    # unique_indices, _, counts = torch.unique(edge_index, dim=0, return_inverse=True, return_counts=True)
    # edge_index = unique_indices[counts == 1]
    edge_attr = distances[edge_index[0, :], edge_index[1, :]].flatten().unsqueeze(1)

    node_features = torch.cat([position, tw], dim=1)
    pyg_data = Data(x=node_features, edge_attr=edge_attr.float(), edge_index=edge_index)
    return pyg_data


def load_test_dataset(val_dataset, n_node, k_sparse, device):
    filename = val_dataset
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )
    dataset = torch.load(filename, map_location=device)

    test_list = []
    for i in range(len(dataset)):
        tw, position, distances = dataset[i, :, 0:2], dataset[i, :, 2:4], dataset[i, :, 4:]
        pyg_data = gen_pyg_data(tw, position, distances, k_sparse, device, start_node=0)
        test_list.append((pyg_data, tw, distances, position))
    return test_list


def load_val_dataset(n_node, tw_type, tw_duration, k_sparse, device, start_node = None):
    filename = f"../data/tsptw/valDataset-{n_node}-{tw_type}-{tw_duration}.pt"
    if not os.path.isfile(filename):
        tw, dist, position = gen_instance(50, n_node, tw_type, tw_duration, normalize=True, device=device)  # type: ignore
        dataset = torch.cat([tw, position, dist], dim=-1)
        torch.save(dataset, filename)
    else:
        dataset = torch.load(filename, map_location=device)

    val_list = []
    for i in range(len(dataset)):
        tw, position, distances = dataset[i, :, 0:2], dataset[i, :, 2:4], dataset[i, :, 4:]
        pyg_data = gen_pyg_data(tw, position, distances, k_sparse, device, start_node)
        val_list.append((pyg_data, tw, distances, position))
    return val_list


def load_vrplib_dataset(n_nodes, k_sparse_factor, device, dataset_name="X", filename=None):
    if dataset_name == "X":
        scale_map = {100: "100_299", 200: "100_299", 400: "300_699", 500: "300_699", 1000: "700_1001"}
    elif dataset_name == "M":
        scale_map = {100: "100_200", 200: "100_200"}
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    filename = filename or f"../data/cvrp/vrplib/vrplib_{dataset_name}_{scale_map[n_nodes]}.pkl"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )
    with open(filename, "rb") as f:
        vrplib_list = pickle.load(f)

    test_list = []
    int_dist_list = []
    name_list = []
    for normed_demand, position, distance, name in vrplib_list:
        # demand is already normalized by capacity
        # normalize the position and distance into [0.01, 0.99] range
        scale = (position.max(0) - position.min(0)).max() / 0.98
        position = position - position.min(0)
        position = position / scale + 0.01
        normed_dist = distance / scale
        np.fill_diagonal(normed_dist, 1e-10)
        # convert all to torch
        normed_demand = torch.tensor(normed_demand, device=device, dtype=torch.float64)
        normed_dist = torch.tensor(normed_dist, device=device, dtype=torch.float64)
        position = torch.tensor(position, device=device, dtype=torch.float64)
        pyg_data = gen_pyg_data(normed_demand, normed_dist, device, k_sparse=position.shape[0] // k_sparse_factor)

        test_list.append((pyg_data, normed_demand, normed_dist, position))
        int_dist_list.append(distance)
        name_list.append(name)
    return test_list, int_dist_list, name_list


if __name__ == '__main__':
    import pathlib
    pathlib.Path('../data/cvrp').mkdir(exist_ok=True)

    # TAM dataset
    for n in [100, 400, 1000]:  # problem scale
        torch.manual_seed(123456)
        inst_list = []
        for _ in range(100):
            demand, dist, position = gen_instance(n, 'cpu', tam=True)  # type: ignore
            instance = torch.vstack([demand, position.T, dist])
            inst_list.append(instance)
        testDataset = torch.stack(inst_list)
        torch.save(testDataset, f'../data/cvrp/testDataset-tam-{n}.pt')

    # main Dataset
    for scale in [200, 500, 1000]:
        with open(f"../data/cvrp/vrp{scale}_128.pkl", "rb") as f:
            dataset = pickle.load(f)

        inst_list = []
        for instance in dataset:
            depot_position, positions, demands, capacity = instance

            demands_torch = torch.tensor([0] + [d / capacity for d in demands], dtype=torch.float64)
            positions_torch = torch.tensor([depot_position] + positions, dtype=torch.float64)
            distmat_torch = gen_distance_matrix(positions_torch)
            inst_list.append(torch.vstack([demands_torch, positions_torch.T, distmat_torch]))

            test_dataset = torch.stack(inst_list)
            torch.save(test_dataset, f"../data/cvrp/testDataset-{scale}.pt")
