from dataclasses import dataclass
import torch
import os, pickle
import numpy as np

__all__ = ['TSPDLEnv']


@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None 
    node_draft_limit: torch.Tensor = None # draft limit
    # shape: (batch, problem)
    prob_emb: torch.Tensor = None
    # shape: (num_training_prob)


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
    # shape: (batch, pomo)
    infeasible: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)


class TSPDLEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "TSPDL"
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.hardness = env_params['hardness']
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
        self.lib_node_xy = None
        # shape: (batch, problem, 2)
        self.node_demand = None
        self.node_draft_limit = None
        # shape: (batch, problem)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.infeasibility_list = None
        self.out_of_draft_limit_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        self.simulated_ninf_flag = None
        self.global_mask = None
        self.global_mask_ninf_flag = None
        self.out_of_dl_ninf_flag = None
        # shape: (batch, pomo, problem)
        self.ninf_mask = None
        # shape: (batch, pomo, problem)
        self.finished = None
        self.infeasible = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, problems=None, aug_factor=1, normalize=False):
        if problems is not None:
            node_xy, node_demand, node_draft_limit = problems
        else:
            node_xy, node_demand, node_draft_limit = self.get_random_problems(batch_size, self.problem_size, normalized=normalize)
        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                node_draft_limit = node_draft_limit.repeat(8, 1)
            else:
                raise NotImplementedError
        # print(node_xy.size())
        self.node_xy = node_xy
        if self.lib_node_xy is not None:
            self.lib_node_xy = self.lib_node_xy.repeat(8,1,1)
            # shape: (8*batch, N, 2)
        # shape: (batch, problem, 2)
        self.node_demand = node_demand
        self.node_draft_limit = node_draft_limit
        # shape: (batch, problem)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_draft_limit = node_draft_limit

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size+1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.out_of_draft_limit_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(self.device) # True for causing infeasibility
        # shape: (batch, pomo, 0~)

        self.load = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.out_of_dl_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        self.infeasible = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
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
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
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
        demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load += selected_demand

        current_coord = self.node_xy[torch.arange(self.batch_size)[:, None], selected]
        # shape: (batch, pomo, 2)
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # shape: (batch, pomo)
        self.length = self.length + new_length
        self.current_coord = current_coord

        # Mask
        ####################################
        # visit constraint
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem)

        # draft limit constraint
        round_error_epsilon = 0.00001
        dl_list = self.node_draft_limit[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem)

        # obtain one-step PI masking
        if generate_PI_mask and self.selected_count < self.problem_size -1:
            unvisited = torch.masked_select(
                torch.arange(self.problem_size).unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.pomo_size, self.problem_size),
                self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
            two_step_unvisited = unvisited.unsqueeze(2).repeat(1, 1, self.problem_size-self.selected_count, 1)
            diag_element = torch.diag_embed(torch.diagonal(two_step_unvisited, dim1=-2, dim2=-1))
            two_step_idx = torch.masked_select(two_step_unvisited, diag_element==0).reshape(self.batch_size, self.pomo_size, self.problem_size-self.selected_count, -1)

            two_step_dl = torch.masked_select(dl_list, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
            two_step_dl = two_step_dl.unsqueeze(2).repeat(1, 1, self.problem_size-self.selected_count, 1)
            two_step_dl = torch.masked_select(two_step_dl, diag_element==0).reshape(self.batch_size, self.pomo_size, self.problem_size-self.selected_count, -1)

            current_load = self.load.unsqueeze(-1).repeat(1, 1, self.problem_size-self.selected_count)
            # add demand of the first-step nodes
            first_step_demand = torch.masked_select(demand_list, self.visited_ninf_flag != float('-inf')).reshape(self.batch_size, self.pomo_size, -1)
            current_load += first_step_demand
            current_load = current_load.unsqueeze(-1).repeat(1, 1,1,two_step_dl.size(-1))
            # add demand of the second-step nodes
            second_step_demand = first_step_demand.unsqueeze(2).repeat(1, 1, self.problem_size-self.selected_count, 1)
            second_step_demand = torch.masked_select(second_step_demand, diag_element==0).reshape(self.batch_size, self.pomo_size, self.problem_size-self.selected_count, -1)
            current_load += second_step_demand

            # feasibility judgement
            infeasible_mark = (current_load > two_step_dl + round_error_epsilon)
            selectable = (infeasible_mark == False).all(dim=-1)
            self.simulated_ninf_flag = torch.full((self.batch_size, self.pomo_size, self.problem_size), float('-inf'))
            selected_indices = selectable.nonzero(as_tuple=False)
            unvisited_indices = unvisited[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
            self.simulated_ninf_flag[selected_indices[:, 0], selected_indices[:, 1], unvisited_indices] = 0.

        # (current load + demand of next node > draft limit of next node) means infeasible
        out_of_dl = self.load[:, :, None] + demand_list > dl_list + round_error_epsilon
        # shape: (batch, pomo, problem)
        self.out_of_dl_ninf_flag[out_of_dl] = float('-inf')
        # shape: (batch, pomo, problem)
        # value that exceeds draft limit of the selected node = current load - node_draft_limit
        total_out_of_dl = self.load - self.node_draft_limit[torch.arange(self.batch_size)[:, None], selected]
        # negative value means current load < node_draft_limit, turn it into 0
        total_out_of_dl = torch.where(total_out_of_dl<0, torch.zeros_like(total_out_of_dl), total_out_of_dl)
        # shape: (batch, pomo)
        self.out_of_draft_limit_list = torch.cat((self.out_of_draft_limit_list, total_out_of_dl[:, :, None]), dim=2)

        self.ninf_mask = self.visited_ninf_flag.clone()
        if not visit_mask_only:
            self.ninf_mask[out_of_dl] = float('-inf')
        if generate_PI_mask and self.selected_count < self.problem_size -1 and (not use_predicted_PI_mask):
            self.ninf_mask = torch.where(self.simulated_ninf_flag==float('-inf'), float('-inf'), self.ninf_mask)
            all_infsb = ((self.ninf_mask==float('-inf')).all(dim=-1)).unsqueeze(-1).expand(-1,-1,self.problem_size)
            self.ninf_mask = torch.where(all_infsb, self.visited_ninf_flag, self.ninf_mask)

        # visited == 0 means not visited
        # out_of_dl_ninf_flag == -inf means already can not be visited bacause current_load + node_demand > node_draft_limit
        newly_infeasible = (((self.visited_ninf_flag == 0).int() + (self.out_of_dl_ninf_flag == float('-inf')).int()) == 2).any(dim=2)
        self.infeasible = self.infeasible + newly_infeasible
        # once the infeasibility occurs, no matter which node is selected next, the route has already become infeasible
        self.infeasibility_list = torch.cat((self.infeasibility_list, self.infeasible[:, :, None]), dim=2)
        infeasible = 0.
        # infeasible_rate = self.infeasible.sum() / (self.batch_size * self.pomo_size)
        # print(">> Cause Infeasibility: Inlegal rate: {}".format(infeasible_rate))

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.load = self.load
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        # returning values
        done = self.finished.all()
        if done:
            if not out_reward:
                reward = -self._get_travel_distance()  # note the minus sign!
            else:
                # shape: (batch, pomo)
                dist_reward = -self._get_travel_distance()  # note the minus sign
                total_out_of_dl_reward = - self.out_of_draft_limit_list.sum(dim=-1)
                out_of_dl_nodes_reward = - torch.where(self.out_of_draft_limit_list > 0, torch.ones_like(self.out_of_draft_limit_list), self.out_of_draft_limit_list).sum(-1).int()
                reward = [dist_reward, total_out_of_dl_reward, out_of_dl_nodes_reward]
                # not visited but can not reach
                # infeasible_rate = self.infeasible.sum() / (self.batch_size*self.pomo_size)
            infeasible = self.infeasible
            # print(">> Cause Infeasibility: Inlegal rate: {}".format(infeasible_rate))
        else:
            reward = None

        return self.step_state, reward, done, infeasible

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        if self.lib_node_xy is not None:
            all_xy = self.lib_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        else:
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

        data = self.get_random_problems(num_samples, problem_size, normalized=False)
        dataset = [attr.cpu().tolist() for attr in data]
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)

        with open(path, 'wb') as f:
            pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)

        print("Save TSPDL dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):

        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))

        node_xy, node_demand, node_draft_limit = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data]
        node_xy, node_demand, node_draft_limit = torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(node_draft_limit)

        # scale it to [0,1]
        demand_sum = node_demand.sum(-1).view(-1, 1)
        node_demand = node_demand / demand_sum
        node_draft_limit = node_draft_limit / demand_sum
        data = (node_xy, node_demand, node_draft_limit)

        return data

    def get_random_problems(self, batch_size, problem_size, normalized=True):

        dl_percent_DICT = {
            "hard": 90,
            "medium": 75,
            "easy": 50,
        }
        dl_percent = dl_percent_DICT[self.hardness]

        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)
        node_demand = torch.cat([torch.zeros((batch_size, 1)), torch.ones((batch_size, problem_size - 1))], dim=1)
        # (batch, problem) 0,1,1,1,1,....
        demand_sum = node_demand.sum(dim=1).unsqueeze(1)
        # currently, demand_sum == problem_size-1; if not, the program needs to be revised (todo)
        node_draft_limit = torch.ones((batch_size, problem_size)) * demand_sum
        for i in range(batch_size):
            # randomly choose w% of the nodes (except depot) to lower their draft limit (range: [1, demand_sum))
            lower_dl_idx = np.random.choice(range(1, problem_size), size=problem_size * dl_percent // 100, replace=False)
            feasible_dl = False
            while not feasible_dl:
                lower_dl = torch.randint(1, demand_sum[i].int().item(), size=(problem_size * dl_percent // 100,))
                cnt = torch.bincount(lower_dl)
                cnt_cumsum = torch.cumsum(cnt, dim=0)
                feasible_dl = (cnt_cumsum <= torch.arange(0, cnt.size(0))).all()
            node_draft_limit[i, lower_dl_idx] = lower_dl.float()
        if normalized:
            node_demand = node_demand / demand_sum
            node_draft_limit = node_draft_limit / demand_sum

        return node_xy, node_demand, node_draft_limit

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
