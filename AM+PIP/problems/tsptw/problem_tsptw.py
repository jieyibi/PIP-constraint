import math
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import torch
import os
import pickle
from .state_tsptw import StateTSPTWInt
from collections import namedtuple

TSPTW_SET = namedtuple("TSPTW_SET",
                       ["node_loc",  # Node locations 1
                        "node_tw",  # node time windows 5
                        "durations",  # service duration per node 6
                        "service_window",  # maximum of time units 7
                        "time_factor", "loc_factor"])


class TSPTW(object):

    NAME = 'tsptw'  # TSP with Time Windows

    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param pi: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        # Check that tours are valid, i.e. contain 1 to n (0 is depot and should not be included)
        if (pi[:, 0] == 0).all():
            pi = pi[:, 1:]  # Strip of depot
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) + 1 ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Distance must be provided in dataset since way of rounding can vary
        coords = dataset[:, :, :2]
        dist = torch.cdist(coords, coords, p=2, compute_mode='donot_use_mm_for_euclid_dist')

        batch_size, graph_size, _ = dataset.size()

        # Check the time windows
        t = dist.new_zeros((batch_size, ))
        #assert (pi[:, 0] == 0).all()  # Tours must start at depot
        batch_zeros = pi.new_zeros((batch_size, ))
        cur = batch_zeros
        batch_ind = torch.arange(batch_size).long()
        lb, ub = torch.unbind(dataset[:, :, 2:], -1)
        timeout = []
        for i in range(graph_size - 1):
            next = pi[:, i]
            t = torch.max(t + dist[batch_ind, cur, next], lb[batch_ind, next])
            timeout.append(torch.clamp(t - ub[batch_ind, next], min=0))
            # assert (t <= ub[batch_ind, next]).all()
            cur = next

        length = dist[batch_ind, 0, pi[:, 0]] + dist[batch_ind[:, None], pi[:, :-1], pi[:, 1:]].sum(-1) + dist[batch_ind, pi[:, -1], 0]
        # We want to maximize total prize but code minimizes so return negative
        return length, torch.stack(timeout,1)

    # @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSPTWInt.initialize(*args, **kwargs)


class TSPTWDataset(Dataset):

    def __init__(self, filename=None, size=100, num_samples=1000000, offset=0, normalize=True, hardness=None):
        super(TSPTWDataset, self).__init__()


        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)[offset: offset+num_samples]
            node_xy, tw_start, tw_end = [i[0] for i in data], [i[2] for i in data], [i[3] for i in data]
            node_xy, tw = torch.Tensor(node_xy), torch.cat([torch.Tensor(tw_start).unsqueeze(-1), torch.Tensor(tw_end).unsqueeze(-1)], dim=-1)
        else:
            node_xy, tw = get_random_problems(num_samples, size, hardness)
            #(batch, problem, 2)

        if normalize:
            # Normalize as in DPDP (Kool et. al)
            loc_factor = 100
            node_xy = node_xy / loc_factor  # Normalize
            tw = tw / loc_factor
            tw_end_max = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw[:, 1:, 1]).max(dim=-1)[0]
            tw[:, 0, 1] = tw_end_max
            # nodes_timew = nodes_timew / nodes_timew[0, 1]

        self.data = torch.cat([node_xy, tw], dim=-1)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def get_rounded_distance_matrix(coord):
    return cdist(coord, coord).round().astype(np.int)


def get_random_problems(batch_size, problem_size, hardness, normalized=True, coord_factor=100, max_tw_size = 100):

    if hardness == "hard":
        # Taken from DPDP (Kool et. al)
        # Taken from https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/tsptw/environment/tsptw.py
        """
        :param problem_size: number of cities
        :param grid_size (=1): x-pos/y-pos of cities will be in the range [0, grid_size]
        :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
        :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
        :return: a feasible TSPTW instance randomly generated using the parameters
        """
        node_xy = torch.rand(size=(batch_size, problem_size, 2)) * coord_factor  # (batch, problem, 2)
        travel_time = torch.cdist(node_xy, node_xy, p=2, compute_mode='donot_use_mm_for_euclid_dist')  # (batch, problem, problem)

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
            total_dist += cur_dist

            # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
            rand_tw_lb = torch.rand(batch_size) * (max_tw_size / 2) + (total_dist - max_tw_size / 2)
            rand_tw_ub = torch.rand(batch_size) * (max_tw_size / 2) + total_dist

            time_windows[torch.arange(batch_size), cur_city, :] = torch.cat([rand_tw_lb.unsqueeze(1), rand_tw_ub.unsqueeze(1)], dim=1)

    elif hardness in ['easy', 'medium']:

        tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=problem_size*55, tw_duration="5075" if hardness == "easy" else "1020")
        node_xy = torch.tensor(tw.node_loc).float()
        time_windows = torch.tensor(tw.node_tw)

    else:
        raise NotImplementedError

    return node_xy, time_windows


def generate_instance(batch_size=1, problem_size=100, coord_factor=100, max_tw_gap=10, max_tw_size=100, tw_type=None, tw_duration=None):

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
        node_xy = torch.rand(size=(batch_size, problem_size, 2)) * coord_factor  # (batch, problem, 2)
        travel_time = torch.cdist(node_xy, node_xy, p=2, compute_mode='donot_use_mm_for_euclid_dist') # (batch, problem, problem)

        random_solution = torch.arange(1, problem_size).repeat(batch_size, 1)
        idx = [torch.randperm(random_solution.size(1)) for _ in range(batch_size)]
        out = torch.zeros(size=(0, problem_size-1)).long()
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
            if tw_type=="da_silva":
                # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for TSPTW"
                rand_tw_lb = torch.rand(batch_size) * (max_tw_size / 2) + (total_dist - max_tw_size / 2)
                rand_tw_ub = torch.rand(batch_size) * (max_tw_size / 2) + total_dist
            elif tw_type == "cappart":
                # Cappart et al. style 'propagates' the time windows resulting in little overlap / easier instances
                rand_tw_lb = torch.rand(batch_size) * (max_tw_gap) + tw_lb_min
                rand_tw_ub = torch.rand(batch_size) * (max_tw_size) + rand_tw_lb

            time_windows[torch.arange(batch_size), cur_city, :] = torch.cat([rand_tw_lb.unsqueeze(1), rand_tw_ub.unsqueeze(1)], dim=1)
    elif tw_type == "zhang":
        TSPTW_SET = namedtuple("TSPTW_SET",
                               ["node_loc",  # Node locations 1
                                "node_tw",  # node time windows 5
                                "durations",  # service duration per node 6
                                "service_window",  # maximum of time units 7
                                "time_factor", "loc_factor"])
        # tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=1.42,
        #                          tw_type="naive/hard", tw_duration=self.tw_duration) # 1.42 = sqrt()
        tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=problem_size*55, tw_type="naive/hard", tw_duration=tw_duration)
        node_xy = torch.tensor(tw.node_loc).float()
        time_windows = torch.tensor(tw.node_tw)
    elif tw_type == "random":
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)
        service_window = int(1.42 * problem_size)
        tw_start = torch.rand(size=(batch_size, problem_size, 1)) * (service_window/2)
        episilon = ((torch.rand(size=(batch_size, problem_size, 1)) * 0.8) + 0.1) * (service_window/2) #[0.1,0.9]
        tw_end = tw_start + episilon
        # Normalize as in DPDP (Kool et. al)
        # Upper bound for depot = max(node ub + dist to depot), to make this tight
        tw_start[:, 0, 0] = 0.
        tw_end[:, 0, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:, 0]).max(dim=1)[0]
        time_windows = torch.cat([tw_start, tw_end], dim=-1)
    else:
        raise NotImplementedError

    return node_xy[:, 0], node_xy[:, 1:], time_windows, coord_factor


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



