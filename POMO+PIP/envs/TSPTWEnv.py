from dataclasses import dataclass
import torch
import os, pickle
import sys
import numpy as np
from collections import namedtuple

__all__ = ['TSPTWEnv']
dg = sys.modules[__name__]

TSPTW_SET = namedtuple("TSPTW_SET",
                       ["node_loc",  # Node locations 1
                        "node_tw",  # node time windows 5
                        "durations",  # service duration per node 6
                        "service_window",  # maximum of time units 7
                        "time_factor", "loc_factor"])


@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_service_time: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_start: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_end: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem)
    finished: torch.Tensor = None
    infeasible: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)


class TSPTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "TSPTW"
        self.env_params = env_params
        self.hardness = env_params['hardness']
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else env_params['device']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.node_xy = None
        # shape: (batch, problem, 2)
        self.node_service_time = None
        # shape: (batch, problem)
        self.node_tw_start = None
        # shape: (batch, problem)
        self.node_tw_end = None
        # shape: (batch, problem)
        self.speed = 1.0

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.timestamps = None
        self.infeasibility_list = None
        self.timeout_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.visited_ninf_flag = None
        self.simulated_ninf_flag = None
        self.global_mask = None
        self.global_mask_ninf_flag = None
        self.out_of_tw_ninf_flag = None
        # shape: (batch, pomo, problem)
        self.ninf_mask = None
        # shape: (batch, pomo, problem)
        self.finished = None
        self.infeasible = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, problems=None, aug_factor=1, normalize=True):
        if problems is not None:
            node_xy, service_time, tw_start, tw_end = problems
        else:
            node_xy, service_time, tw_start, tw_end = self.get_random_problems(batch_size, self.problem_size, max_tw_size=100)

        if normalize:
            # Normalize as in DPDP (Kool et. al)
            loc_factor = 100
            node_xy = node_xy / loc_factor  # Normalize
            # Normalize same as coordinates to keep unit the same, not that these values do not fall in range [0,1]!
            # Todo: should we additionally normalize somehow, e.g. by expected makespan/tour length?
            tw_start = tw_start / loc_factor
            tw_end = tw_end / loc_factor
            # Upper bound for depot = max(node ub + dist to depot), to make this tight
            tw_end[:, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:]).max(dim=-1)[0]
            # nodes_timew = nodes_timew / nodes_timew[0, 1]

        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
            else:
                raise NotImplementedError

        self.node_xy = node_xy
        # shape: (batch, problem, 2)
        self.node_service_time = service_time
        # shape: (batch, problem)
        self.node_tw_start = tw_start
        # shape: (batch, problem)
        self.node_tw_end = tw_end
        # shape: (batch, problem)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.node_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size + 1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem

        #only calculate PIP masking for k nearest neighbors
        k_sparse = self.env_params["k_sparse"]
        node_xy_expanded = node_xy[:, :, None, :]  # (B, N, 1, 2)
        node_xy_expanded_T = node_xy[:, None, :, :]  # (B, 1, N, 2)
        distances = torch.sqrt(torch.sum((node_xy_expanded - node_xy_expanded_T) ** 2, dim=-1))  # (B, N, N)
        diag_mask = torch.eye(self.problem_size).unsqueeze(0).repeat(self.batch_size,1,1) * (1e9)
        distances += diag_mask

        if k_sparse < self.problem_size:
            self.is_sparse = True
            print("Sparse, ", k_sparse)
            # sparsify
            # Part 1: k nearest distance (n_nodes * k_sparse)
            _, topk_indices1 = torch.topk(distances, k=k_sparse, dim=-1, largest=False)
            dist_neighbors_index = torch.cat([
                torch.repeat_interleave(torch.arange(self.problem_size), repeats=k_sparse).reshape(1, self.problem_size,-1).repeat(self.batch_size,1,1).unsqueeze(-1),
                topk_indices1.unsqueeze(-1)
            ], dim=-1)
            # edge_attr1 = topk_values1.reshape(-1, 1)

            # Part 2: k nearest tw_start for start node (0 by default) (1 * k_sparse)
            start_node_tw_start = tw_start[:, :1]
            tw_start_differences = tw_start - start_node_tw_start
            tw_start_differences[tw_start_differences <= 0] = float('inf')
            _, topk_indices2 = torch.topk(tw_start_differences, k=k_sparse, dim=-1, largest=False)
            edge_index0 = torch.cat([
                torch.repeat_interleave(torch.tensor(0), repeats=k_sparse).reshape(1,-1).repeat(self.batch_size,1).unsqueeze(-1),
                topk_indices2.unsqueeze(-1)
            ], dim=-1)
            # edge_attr2 = topk_values2.reshape(-1, 1)

            # Part 3: k tw overlap most ((n_nodes-1) * k_sparse)
            # overlap = min(end_A, end_B) - max(start_A, start_B)
            start_times = tw_start[:,1:].unsqueeze(-1).expand(-1, -1, self.problem_size - 1)
            end_times = tw_end[:,1:].unsqueeze(-1).expand(-1, -1, self.problem_size - 1)
            start_max = torch.max(start_times, start_times.transpose(1,2))
            end_min = torch.min(end_times, end_times.transpose(1, 2))
            overlap_matrix = torch.clamp(end_min - start_max, min=0)
            eye_matrix = torch.eye(self.problem_size-1).unsqueeze(0).repeat(self.batch_size, 1, 1).bool()
            overlap_matrix[eye_matrix] = 0.  # ignore self
            del eye_matrix
            _, topk_indices3 = torch.topk(overlap_matrix, k=k_sparse, dim=-1)
            topk_indices3 += 1  # since we remove the first node (start node) in overlap_matrix
            edge_index1 = torch.cat([
                torch.repeat_interleave(torch.arange(1, self.problem_size), repeats=k_sparse).reshape(1, self.problem_size-1,-1).repeat(self.batch_size,1,1).unsqueeze(-1),
                topk_indices3.unsqueeze(-1)
            ], dim=-1)
            tw_neighbors_index = torch.concat([edge_index0.unsqueeze(1), edge_index1], dim=1)
            self.neighbour_index = torch.concat([dist_neighbors_index, tw_neighbors_index], dim=2)
            self.k_neigh_ninf_flag =torch.full((self.batch_size, self.problem_size, self.problem_size), float('-inf'))
            # set nerighbor position to 0
            indices = self.neighbour_index.view(self.batch_size, -1, 2)
            self.k_neigh_ninf_flag[torch.arange(self.batch_size).view(-1, 1).expand_as(indices[:, :, 0]), indices[:, :, 0], indices[:, :, 1]] = 0
            self.k_neigh_ninf_flag[torch.arange(self.batch_size).view(-1, 1).expand_as(indices[:, :, 0]), indices[:, :, 1], indices[:, :, 0]] = 0
        else:
            self.is_sparse = False

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.timestamps = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.timeout_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(self.device) # True for causing infeasibility
        # shape: (batch, pomo, 0~)

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.out_of_tw_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        self.infeasible = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.current_coord = self.node_xy[:, :1, :]  # depot
        # shape: (batch, pomo, 2)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, visit_mask_only =True, out_reward = False, generate_PI_mask=False, use_predicted_PI_mask=False, pip_step=1):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        current_coord = self.node_xy[torch.arange(self.batch_size)[:, None], selected]
        # shape: (batch, pomo, 2)
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # shape: (batch, pomo)
        self.length = self.length + new_length
        self.current_coord = current_coord

        # Mask
        ####################################
        # visited mask
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem)

        # wait until time window starts
        self.current_time = (torch.max(self.current_time + new_length / self.speed,
                                       self.node_tw_start[torch.arange(self.batch_size)[:, None], selected])
                             + self.node_service_time[torch.arange(self.batch_size)[:, None], selected])
        # shape: (batch, pomo)
        self.timestamps = torch.cat((self.timestamps, self.current_time[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # time window constraint
        #   current_time: the end time of serving the current node
        #   max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        round_error_epsilon = 0.00001
        next_arrival_time = torch.max(self.current_time[:, :, None] + (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed,
                                 self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
        out_of_tw = next_arrival_time > self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
        # shape: (batch, pomo, problem)
        self.out_of_tw_ninf_flag[out_of_tw] = float('-inf')

        # simulate the PIP masking
        if generate_PI_mask and self.selected_count < self.problem_size -1:
            self._calculate_PIP_mask(pip_step, selected)

        # timeout value of the selected node = current time - tw_end
        total_timeout = self.current_time - self.node_tw_end[torch.arange(self.batch_size)[:, None], selected]
        # negative value means current time < tw_end, turn it into 0
        total_timeout = torch.where(total_timeout<0, torch.zeros_like(total_timeout), total_timeout)
        # shape: (batch, pomo)
        self.timeout_list = torch.cat((self.timeout_list, total_timeout[:, :, None]), dim=2)

        self.ninf_mask = self.visited_ninf_flag.clone()
        if not visit_mask_only:
            self.ninf_mask[out_of_tw] = float('-inf')
        if generate_PI_mask and self.selected_count < self.problem_size -1 and (not use_predicted_PI_mask): # use PIP mask once not using mask from PIP-D
            self.ninf_mask = torch.where(self.simulated_ninf_flag==float('-inf'), float('-inf'), self.ninf_mask)
            all_infsb = ((self.ninf_mask == float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1, -1, self.problem_size)
            # all_infsb = ((self.simulated_ninf_flag==float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1,-1,self.problem_size)
            self.ninf_mask = torch.where(all_infsb, self.visited_ninf_flag, self.ninf_mask)

        # visited == 0 means not visited
        # out_of_tw_ninf_flag == -inf means already can not be visited bacause current_time + travel_time > tw_end
        newly_infeasible = (((self.visited_ninf_flag == 0).int() + (self.out_of_tw_ninf_flag == float('-inf')).int()) == 2).any(dim=2)

        self.infeasible = self.infeasible + newly_infeasible
        # once the infeasibility occurs, no matter which node is selected next, the route has already become infeasible
        self.infeasibility_list = torch.cat((self.infeasibility_list, self.infeasible[:, :, None]), dim=2)
        infeasible = 0.
        # infeasible_rate = self.infeasible.sum() / (self.batch_size * self.pomo_size)
        # print(">> Cause Infeasibility: Illegal rate: {}".format(infeasible_rate))

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        # returning values
        done = self.finished.all()
        if done:
            if not out_reward:
                reward = -self._get_travel_distance()  # note the minus sign!
            else:
                # shape: (batch, pomo)
                dist_reward = -self._get_travel_distance() # note the minus sign
                total_timeout_reward = -self.timeout_list.sum(dim=-1)
                timeout_nodes_reward = -torch.where(self.timeout_list>0, torch.ones_like(self.timeout_list), self.timeout_list).sum(-1).int()
                reward = [dist_reward, total_timeout_reward, timeout_nodes_reward]
            # not visited but can not reach
            # infeasible_rate = self.infeasible.sum() / (self.batch_size*self.pomo_size)
            infeasible = self.infeasible
            # print(">> Cause Infeasibility: Inlegal rate: {}".format(infeasible_rate))
        else:
            reward = None

        return self.step_state, reward, done, infeasible

    def _calculate_PIP_mask(self, pip_step, selected):

        round_error_epsilon = 0.00001
        next_arrival_time = torch.max(self.current_time[:, :, None] + (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed,
                                 self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
        node_tw_end = self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1)

        if pip_step == 0:
            if self.is_sparse:
                print("Warning! Performing zero-step PIP masking on k nearest neighbors is not supported! Consider all the unvisited nodes instead!")
            out_of_tw = next_arrival_time > node_tw_end + round_error_epsilon  # shape: (batch, pomo, problem)
            self.simulated_ninf_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
            self.simulated_ninf_flag[out_of_tw] = float('-inf')
        elif pip_step == 1:

            if self.is_sparse: # calculate the k_sparse_mask
                self.not_neigh_ninf_flag = self.k_neigh_ninf_flag[:, None, :, :].repeat(1, self.pomo_size, 1, 1)

                self.not_neigh_ninf_flag = self.not_neigh_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected]
                # shape: (batch, pomo, problem)

                # Mark the unvisited neighbors as True
                self.visited_and_notneigh_ninf_flag = (self.not_neigh_ninf_flag == 0) & (self.visited_ninf_flag == 0)  # (B, P, N)
                # calculate the count of the unvisited neighbors for each instance
                unvisited_and_neigh_counts = self.visited_and_notneigh_ninf_flag.sum(dim=-1)  # (B, P)
                max_count = unvisited_and_neigh_counts.max().item()  # (self.batch_size, P, N)
                # extract the unvisited neighbors
                _, unvisited_and_neigh = torch.topk(self.visited_and_notneigh_ninf_flag.int(), dim=-1,
                                                    largest=True, k=max_count)  # (B, P, N)
                unvisited = unvisited_and_neigh.sort(dim=-1)[0]  # shape: (batch, pomo, max_count)
                del unvisited_and_neigh, unvisited_and_neigh_counts

            else: # all the unvisited nodes will be considered
                unvisited = torch.masked_select(torch.arange(self.problem_size).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.pomo_size, self.problem_size),
                    self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)

            simulate_size = unvisited.size(-1)
            two_step_unvisited = unvisited.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            diag_element = torch.eye(simulate_size).view(1, 1, simulate_size, simulate_size).repeat(self.batch_size, self.pomo_size, 1, 1)
            two_step_idx = torch.masked_select(two_step_unvisited, diag_element == 0).reshape(self.batch_size,self.pomo_size,simulate_size, -1)

            # add arrival_time of the first-step nodes
            first_step_current_coord = self.node_xy.unsqueeze(1).repeat(1, self.pomo_size, 1, 1).gather(dim=2, index=unvisited.unsqueeze(3).expand(-1, -1, -1, 2))
            # first_step_new_length = (first_step_current_coord - current_coord.unsqueeze(2).repeat(1,1,self.problem_size-self.selected_count,1)).norm(p=2, dim=-1)
            # current_time = self.current_time.unsqueeze(-1).repeat(1, 1, simulate_size)
            # first_step_tw_end = torch.masked_select(node_tw_end, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
            # node_tw_start = self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1)
            # first_step_tw_start = torch.masked_select(node_tw_start, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
            # node_service_time = self.node_service_time[:, None, :].expand(-1, self.pomo_size, -1)
            # first_step_node_service_time= torch.masked_select(node_service_time, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
            # first_step_current_time = torch.max(current_time + first_step_new_length / self.speed, first_step_tw_start) + first_step_node_service_time
            first_step_arrival_time = next_arrival_time.gather(dim=-1, index=unvisited)

            # add arrival_time of the second-step nodes
            two_step_tw_end = node_tw_end.gather(dim=-1, index=unvisited)
            two_step_tw_end = two_step_tw_end.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_tw_end = torch.masked_select(two_step_tw_end, diag_element == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1)

            node_tw_start = self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1)
            two_step_tw_start = node_tw_start.gather(dim=-1, index=unvisited)
            two_step_tw_start = two_step_tw_start.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_tw_start = torch.masked_select(two_step_tw_start, diag_element == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1)

            node_service_time = self.node_service_time[:, None, :].expand(-1, self.pomo_size, -1)
            two_step_node_service_time = node_service_time.gather(dim=-1, index=unvisited)
            two_step_node_service_time = two_step_node_service_time.unsqueeze(2).repeat(1, 1, simulate_size, 1)
            two_step_node_service_time = torch.masked_select(two_step_node_service_time, diag_element == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1)

            two_step_current_coord = first_step_current_coord.unsqueeze(2).repeat(1, 1, simulate_size, 1, 1)
            two_step_current_coord = torch.masked_select(two_step_current_coord, diag_element.unsqueeze(-1).expand(-1, -1, -1, -1, 2) == 0).reshape(self.batch_size, self.pomo_size, simulate_size, -1, 2)
            second_step_new_length = (two_step_current_coord - first_step_current_coord.unsqueeze(3).repeat(1, 1, 1, simulate_size - 1, 1)).norm(p=2, dim=-1)
            first_step_arrival_time = first_step_arrival_time.unsqueeze(-1).repeat(1, 1, 1, simulate_size - 1)
            second_step_arrival_time = torch.max(first_step_arrival_time + second_step_new_length / self.speed, two_step_tw_start) + two_step_node_service_time

            # time window constraint
            #   current_time: the end time of serving the current node
            #   max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
            # feasibility judgement
            infeasible_mark = (second_step_arrival_time > two_step_tw_end + round_error_epsilon)
            selectable = (infeasible_mark == False).all(dim=-1)

            # mark the selectable unvisited nodes
            self.simulated_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
            selected_indices = selectable.nonzero(as_tuple=False)
            unvisited_indices = unvisited[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
            self.simulated_ninf_flag[selected_indices[:, 0], selected_indices[:, 1], unvisited_indices] = 0.

        elif pip_step == 2:
            if self.selected_count < self.problem_size - 2:
                # unvisited nodes
                unvisited = torch.masked_select(torch.arange(self.problem_size).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.pomo_size, self.problem_size), self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                unvisited_size = unvisited.size(-1)
                # add arrival_time of the first-step nodes
                first_step_current_coord = self.node_xy.unsqueeze(1).repeat(1, self.pomo_size, 1, 1).gather(dim=2, index=unvisited.unsqueeze(3).expand(-1, -1, -1, 2))
                first_step_arrival_time = torch.masked_select(next_arrival_time, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)

                # unvisited nodes in the second step
                two_step_unvisited = unvisited.unsqueeze(2).repeat(1, 1, unvisited.size(2), 1)
                diag_element = torch.diag_embed(torch.diagonal(two_step_unvisited, dim1=-2, dim2=-1))
                two_step_idx = torch.masked_select(two_step_unvisited, diag_element == 0).reshape(self.batch_size, self.pomo_size, unvisited_size, -1)

                # add arrival_time of the second-step nodes
                two_step_tw_end = torch.masked_select(node_tw_end, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                two_step_tw_end = two_step_tw_end.unsqueeze(2).repeat(1, 1, self.problem_size - self.selected_count, 1)
                two_step_tw_end = torch.masked_select(two_step_tw_end, diag_element == 0).reshape(self.batch_size, self.pomo_size, unvisited_size, -1)

                node_tw_start = self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1)
                two_step_tw_start = torch.masked_select(node_tw_start, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                two_step_tw_start = two_step_tw_start.unsqueeze(2).repeat(1, 1, self.problem_size - self.selected_count, 1)
                two_step_tw_start = torch.masked_select(two_step_tw_start, diag_element == 0).reshape(self.batch_size, self.pomo_size, unvisited_size, -1)

                # node_service_time = self.node_service_time[:, None, :].expand(-1, self.pomo_size, -1)
                # two_step_node_service_time = torch.masked_select(node_service_time, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
                # two_step_node_service_time = two_step_node_service_time.unsqueeze(2).repeat(1, 1, self.problem_size - self.selected_count, 1)
                # two_step_node_service_time = torch.masked_select(two_step_node_service_time, diag_element == 0).reshape(self.batch_size, self.pomo_size, self.problem_size - self.selected_count, -1)
                two_step_node_service_time = 0.

                two_step_current_coord = first_step_current_coord.unsqueeze(2).repeat(1, 1, unvisited_size, 1, 1)
                two_step_current_coord = torch.masked_select(two_step_current_coord, diag_element.unsqueeze(-1).expand(-1, -1, -1, -1, 2) == 0).reshape(self.batch_size, self.pomo_size, unvisited_size, -1, 2)
                second_step_new_length = (two_step_current_coord - first_step_current_coord.unsqueeze(3).repeat(1, 1, 1, unvisited_size - 1, 1)).norm(p=2, dim=-1)
                first_step_arrival_time = first_step_arrival_time.unsqueeze(-1).repeat(1, 1, 1, unvisited_size - 1)
                second_step_arrival_time = torch.max(first_step_arrival_time + second_step_new_length / self.speed, two_step_tw_start) + two_step_node_service_time

                # Free the memory
                del first_step_arrival_time, first_step_current_coord, diag_element, two_step_node_service_time, second_step_new_length

                two_step_infeasible_mark = (second_step_arrival_time > two_step_tw_end + round_error_epsilon)
                two_step_selectable = (two_step_infeasible_mark == False).all(dim=-1)

                # unvisited nodes in the third step
                three_step_unvisited = two_step_idx.unsqueeze(3).repeat(1, 1, 1, two_step_idx.size(-1), 1)
                del two_step_idx
                diag_element = torch.diag_embed(torch.diagonal(three_step_unvisited, dim1=-2, dim2=-1))
                # three_step_idx = torch.masked_select(three_step_unvisited, diag_element == 0).reshape(self.batch_size, self.pomo_size, unvisited_size, unvisited_size - 1, -1)
                # add arrival_time of the third-step nodes
                three_step_tw_start = two_step_tw_start.unsqueeze(3).expand_as(three_step_unvisited)
                del two_step_tw_start
                three_step_tw_start = torch.masked_select(three_step_tw_start, diag_element == 0).reshape(self.batch_size, self.pomo_size, unvisited_size, unvisited_size - 1, -1)
                three_step_tw_end = two_step_tw_end.unsqueeze(3).expand_as(three_step_unvisited)
                del two_step_tw_end
                three_step_tw_end = torch.masked_select(three_step_tw_end, diag_element == 0).reshape(self.batch_size, self.pomo_size, unvisited_size, unvisited_size - 1, -1)
                three_step_node_service_time = 0.

                three_step_current_coord = two_step_current_coord.unsqueeze(3).expand(-1, -1, -1, unvisited_size - 1, -1, -1)
                three_step_current_coord = torch.masked_select(three_step_current_coord, diag_element.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 2) == 0).reshape(self.batch_size, self.pomo_size, unvisited_size, unvisited_size - 1, -1, 2)
                third_step_new_length = (three_step_current_coord - two_step_current_coord.unsqueeze(4).expand(-1, -1, -1, -1, unvisited_size - 2, -1)).norm(p=2,  dim=-1)
                second_step_arrival_time = second_step_arrival_time.unsqueeze(-1).repeat(1, 1, 1, 1, unvisited_size - 2)
                third_step_arrival_time = torch.max(second_step_arrival_time + third_step_new_length / self.speed, three_step_tw_start) + three_step_node_service_time
                del second_step_arrival_time, two_step_current_coord, diag_element, three_step_node_service_time, third_step_new_length, three_step_tw_start

                # time window constraint
                #   current_time: the end time of serving the current node
                #   max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
                # feasibility judgement
                delta_t = self.env_params["random_delta_t"] * torch.rand(size=third_step_arrival_time.size())
                third_step_arrival_time += delta_t
                infeasible_mark = (third_step_arrival_time > three_step_tw_end + round_error_epsilon)
                selectable = (infeasible_mark == False).all(dim=-1).any(dim=-1)
                selectable = two_step_selectable & selectable  # Fixed the bug of ignoring the infeasible nodes in the last step

                self.simulated_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
                selected_indices = selectable.nonzero(as_tuple=False)
                unvisited_indices = unvisited[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
                self.simulated_ninf_flag[selected_indices[:, 0], selected_indices[:, 1], unvisited_indices] = 0.
            else:
                pass
        else:
            raise NotImplementedError

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        if self.loc_scaler:
            segment_lengths = torch.round(segment_lengths * self.loc_scaler) / self.loc_scaler

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def generate_dataset(self, num_samples, problem_size, path):
        data = self.get_random_problems(num_samples, problem_size)
        dataset = [attr.cpu().tolist() for attr in data]
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(path, 'wb') as f:
            pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)
        print("Save TSPTW dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=10000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
        node_xy, service_time, tw_start, tw_end = torch.Tensor(node_xy), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)

        data = (node_xy, service_time, tw_start, tw_end)
        return data

    def get_random_problems(self, batch_size, problem_size, coord_factor=100, max_tw_size=100):

        if self.hardness == "hard":
            # Taken from DPDP (Kool et. al)
            # Taken from https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/tsptw/environment/tsptw.py
            # max_tw_size = 1000 if tw_type == "da_silva" else 100
            # max_tw_size = problem_size * 2 if tw_type == "da_silva" else 100
            """
            :param problem_size: number of cities
            :param grid_size (=1): x-pos/y-pos of cities will be in the range [0, grid_size]
            :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
            :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
            :return: a feasible TSPTW instance randomly generated using the parameters
            """
            node_xy = torch.rand(size=(batch_size, problem_size, 2)) * coord_factor  # (batch, problem, 2)
            travel_time = torch.cdist(node_xy, node_xy, p=2, compute_mode='donot_use_mm_for_euclid_dist') / self.speed # (batch, problem, problem)

            random_solution = torch.arange(1, problem_size).repeat(batch_size, 1)
            for i in range(batch_size):
                random_solution[i] = random_solution[i][torch.randperm(random_solution.size(1))]
            zeros = torch.zeros(size=(batch_size, 1)).long()
            random_solution = torch.cat([zeros, random_solution], dim=1)

            time_windows = torch.zeros((batch_size, problem_size, 2))
            time_windows[:, 0, :] = torch.tensor([0, 1000. * coord_factor]).repeat(batch_size, 1)

            total_dist = torch.zeros(batch_size)
            for i in range(1, problem_size):
                prev_city = random_solution[:, i - 1]
                cur_city = random_solution[:, i]

                cur_dist = travel_time[torch.arange(batch_size), prev_city, cur_city]

                # tw_lb_min = time_windows[torch.arange(batch_size), prev_city, 0] + cur_dist
                total_dist += cur_dist

                # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
                rand_tw_lb = torch.rand(batch_size) * (max_tw_size / 2) + (total_dist - max_tw_size / 2)
                rand_tw_ub = torch.rand(batch_size) * (max_tw_size / 2) + total_dist

                time_windows[torch.arange(batch_size), cur_city, :] = torch.cat([rand_tw_lb.unsqueeze(1), rand_tw_ub.unsqueeze(1)], dim=1)

        elif self.hardness in ["easy", "medium"]:

            tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=problem_size*55, tw_duration="5075" if self.hardness == "easy" else "1020")
            node_xy = torch.tensor(tw.node_loc).float()
            time_windows = torch.tensor(tw.node_tw)

        else:
            raise NotImplementedError

        service_time = torch.zeros(size=(batch_size,problem_size))
        # Don't store travel time since it takes up much
        return node_xy, service_time, time_windows[:,:,0], time_windows[:,:,1]

    def augment_xy_data_by_8_fold(self, xy_data):
        # xy_data.shape: (batch, N, 2)

        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        # x,y shape: (batch, N, 1)

        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, N, 2)

        return aug_xy_data


def gen_tw(size, graph_size, time_factor, dura_region, rnds):
    """
    Copyright (c) 2020 Jonas K. Falkner
    Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py
    """

    service_window = int(time_factor * 2)

    horizon = np.zeros((size, graph_size, 2))
    horizon[:] = [0, service_window]

    # sample earliest start times
    tw_start = rnds.randint(horizon[..., 0], horizon[..., 1] / 2)
    tw_start[:, 0] = 0

    # calculate latest start times b, which is
    # tw_start + service_time_expansion x normal random noise, all limited by the horizon
    # and combine it with tw_start to create the time windows
    epsilon = rnds.uniform(dura_region[0], dura_region[1], (tw_start.shape))
    duration = np.around(time_factor * epsilon)
    duration[:, 0] = service_window
    tw_end = np.minimum(tw_start + duration, horizon[..., 1]).astype(int)

    tw = np.concatenate([tw_start[..., None], tw_end[..., None]], axis=2).reshape(size, graph_size, 2)

    return tw


def generate_tsptw_data(size, graph_size, rnds=None, time_factor=100.0, loc_factor=100, tw_duration="5075"):
    """
    Copyright (c) 2020 Jonas K. Falkner
    Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py
    """

    rnds = np.random if rnds is None else rnds
    service_window = int(time_factor * 2)

    # sample locations
    nloc = rnds.uniform(size=(size, graph_size, 2)) * loc_factor  # node locations

    # tw duration
    dura_region = {
         "5075": [.5, .75],
         "1020": [.1, .2],
    }
    if isinstance(tw_duration, str):
        dura_region = dura_region[tw_duration]
    else:
        dura_region = tw_duration

    tw = gen_tw(size, graph_size, time_factor, dura_region, rnds)

    return TSPTW_SET(node_loc=nloc,
                     node_tw=tw,
                     durations=tw[..., 1] - tw[..., 0],
                     service_window=[service_window] * size,
                     time_factor=[time_factor] * size,
                     loc_factor=[loc_factor] * size, )