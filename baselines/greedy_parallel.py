'''
This is the implementation of the greedy heuristics for TSPTW and TSPDL
Two heuristics are supported:
1. Greedy-L heuristic selects the candidate with the shortest distance (length)
2. Greedy-C heuristic selects a node based on the satisfaction of constraints,
   which is the candidate with the soonest time window end w.r.t current time in TSPTW
   and the candidate with the minimal draft limit in TSPDL
'''

import pickle
import numpy as np
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def save_results(results, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def greedy_tsptw_instance(instance, heuristics):
    node_xy, service_time, tw_start, tw_end = instance
    num_nodes = len(node_xy)
    visited = [False] * num_nodes
    visited[0] = True  # Start at node 0
    current_node = 0
    tour = [current_node]
    total_distance = 0
    current_time = 0
    feasible = True

    while len(tour) < num_nodes:
        next_node = None
        min_tw_end = float('inf')
        min_distance = float('inf')
        for i in range(1, num_nodes):  # Start from 1 as 0 is the depot
            if not visited[i]:
                if heuristics == "constraint":
                    if tw_end[i] < min_tw_end:
                        min_tw_end = tw_end[i]
                        next_node = i
                elif heuristics == "length":
                    distance = calculate_distance(node_xy[current_node], node_xy[i])
                    if distance < min_distance:
                        min_distance = distance
                        next_node = i
                else:
                    raise NotImplementedError
        if next_node is None:
            break
        visited[next_node] = True
        tour.append(next_node)
        travel_time = calculate_distance(node_xy[current_node], node_xy[next_node]) if heuristics != "length" else min_distance
        current_time = max(current_time + travel_time, tw_start[next_node])
        if current_time > tw_end[next_node] + 0.000001:
            feasible = False
        total_distance += travel_time
        current_node = next_node

    # Return to start node to complete the tour
    total_distance += calculate_distance(node_xy[current_node], node_xy[0])
    tour.append(0)  # Append start node to complete the cycle

    return (tour, total_distance, feasible)

def greedy_tsptw_algorithm(instances, heuristics, num_workers=16):
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_instance = {executor.submit(greedy_tsptw_instance, instance, heuristics): instance for instance in instances}
        for future in as_completed(future_to_instance):
            result = future.result()
            results.append(result)
    return results

def greedy_tspdl_instance(instance, heuristics):
    node_xy, demand, draft_limit = instance
    num_nodes = len(node_xy)
    visited = [False] * num_nodes
    visited[0] = True  # Start at node 0
    current_node = 0
    tour = [current_node]
    total_distance = 0
    current_load = 0
    feasible = True

    while len(tour) < num_nodes:
        next_node = None
        min_draft_limit = float('inf')
        min_distance = float('inf')
        for i in range(1, num_nodes):  # Start from 1 as 0 is the depot
            if not visited[i]:
                if heuristics == "constraint":
                    if draft_limit[i] < min_draft_limit:
                        min_draft_limit = draft_limit[i]
                        next_node = i
                elif heuristics == "length":
                    distance = calculate_distance(node_xy[current_node], node_xy[i])
                    if distance < min_distance:
                        min_distance = distance
                        next_node = i
                else:
                    raise NotImplementedError
        if next_node is None:
            break
        visited[next_node] = True
        tour.append(next_node)
        travel_time = calculate_distance(node_xy[current_node], node_xy[next_node]) if heuristics != "length" else min_distance
        current_load += demand[next_node]
        if current_load > draft_limit[next_node] + 0.000001:
            feasible = False
        total_distance += travel_time
        current_node = next_node

    # Return to start node to complete the tour
    total_distance += calculate_distance(node_xy[current_node], node_xy[0])
    tour.append(0)  # Append start node to complete the cycle

    return (tour, total_distance, feasible)


def greedy_tspdl_algorithm(instances, heuristics, num_workers=16):
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_instance = {executor.submit(greedy_tspdl_instance, instance, heuristics): instance for instance in instances}
        for future in as_completed(future_to_instance):
            result = future.result()
            results.append(result)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy algorithm for TSPTW and TSPDL")
    parser.add_argument("--problem", type=str, default="TSPTW", choices=["TSPTW", "TSPDL"])
    parser.add_argument("--heuristics", type=str, default="constraint", choices=["constraint", "length"])
    parser.add_argument("--datasets", type=str, default="../data/TSPTW/tsptw50_medium.pkl")
    parser.add_argument("--cal_gap", action='store_true', help="Set true to calculate optimality gap")
    parser.add_argument("--optimal_solution_path", type=str, default='../data/TSPTW/lkh_tsptw50_medium.pkl')
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--print_results", action='store_true',
                        help="print tours, objectives, optimal obj., gap and feasibility for each instance")

    args = parser.parse_args()

    start = time.time()

    # load data
    data = load_data(args.datasets)
    # start greedily search
    if args.problem == "TSPTW":
        results = greedy_tsptw_algorithm(data, args.heuristics, args.num_workers)
    elif args.problem == "TSPDL":
        results = greedy_tspdl_algorithm(data, args.heuristics, args.num_workers)
    else:
        raise NotImplementedError

    duration = time.time() - start

    # Post-processing
    save_path = args.datasets.split(".pkl")[0] + "_greedy_{}.pkl".format(args.heuristics)
    save_results(results, save_path)
    print(">> Results are saved to {}".format(save_path))

    if args.cal_gap:
        assert args.optimal_solution_path is not None, "Optimal solution path is not provided. Unable to calculate optimality gap."

        with open(args.optimal_solution_path, 'rb') as file:
            opt_sol = pickle.load(file)

        gaps = np.array([])
        distances = np.array([])
        opt = np.array([])
        feasible_cnt = 0

        for i, result in enumerate(results):
            opt = np.append(opt, opt_sol[i][0]/ 100)
            gap = (result[1] - opt_sol[i][0]) / opt_sol[i][0] *100
            if args.print_results:
                print("Tour {}:".format(i), result[0],
                      "Total Distance:", result[1],
                      "Optimal:",opt_sol[i][0],
                      "Gap:{}%".format(gap),
                      "Feasible:", result[2])
            if result[2]:
                distances = np.append(distances, result[1] / 100)
                gaps = np.append(gaps, gap)
                feasible_cnt += 1

        print(">> Duration: {}".format(duration))
        print(">> Average distance: {}".format(np.mean(distances)))
        print(">> Average distance in LKH3: {}".format(np.mean(opt)))
        print(">> Average gap: {}%".format(np.mean(gaps)))
        print(">> Infeasible count: {}%".format((len(results)-feasible_cnt)/len(results)*100))

    else:
        distances = np.array([])
        feasible_cnt = 0

        for i, result in enumerate(results):
            if args.print_results:
                print("Tour {}:".format(i), result[0],
                      "Total Distance:", result[1],
                      "Feasible:", result[2])
            if result[2]:
                distances = np.append(distances, result[1] / 100)
                feasible_cnt += 1

        print(">> Duration: {}".format(duration))
        print(">> Average distance: {}".format(np.mean(distances)))
        print(">> Infeasible count: {}%".format((len(results)-feasible_cnt)/len(results)*100))
