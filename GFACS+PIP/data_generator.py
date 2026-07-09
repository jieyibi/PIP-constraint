"""
Copyright (c) 2020 Jonas K. Falkner
"""
# Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from collections import namedtuple

dg = sys.modules[__name__]

TSPTW_SET = namedtuple("TSPTW_SET",
                       ["node_loc",  # Node locations 1
                        "node_tw",  # node time windows 5
                        "durations",  # service duration per node 6
                        "service_window",  # maximum of time units 7
                        "time_factor", "loc_factor"])


def get_random_tour(edge_matrix, rnds=None):
    rnds = np.random if rnds is None else rnds

    size = edge_matrix.shape[0]
    graph_size = edge_matrix.shape[1]

    tours = np.concatenate([rnds.permutation((graph_size)) for _ in range(size)], axis=-1).reshape(size, graph_size)
    tours = np.concatenate([tours, tours[:, 0:1]], axis=1)

    # (size, gs)
    d0 = np.arange(size)[:, None].repeat(graph_size, axis=1)
    tours_e = edge_matrix[d0, tours[:, :-1], tours[:, 1:]]

    len = np.sum(tours_e, axis=-1)
    return len


def get_edge_matrix(nodes):
    # (b, ns, 2) -> (b, ns, ns, 2)
    # bs = nodes.shape[0]
    # ns = nodes.shape[1]
    # edges = np.zeros((bs, ns, ns))
    # # nodes[:, :, None]
    # for i in range(ns):
    #     t = (nodes[:, i:i+1] - nodes)
    #     t = np.linalg.norm(t, axis=-1)
    #     # print(t.shape)
    #     edges[:, i, :] = t
    if isinstance(nodes, np.ndarray):
        edges = np.linalg.norm(nodes[:, None, :, :] - nodes[:, :, None, :], axis=-1)
    else:
        edges = torch.norm(nodes[:, None, :, :] - nodes[:, :, None, :], dim=-1)
    # edges = np.linalg.norm(nodes[:, None, :, :] - nodes[:, :, None, :], axis=-1)

    return edges


def gen_tw_naive(size, graph_size, time_factor, dura_region, rnds, tw_type="easy"):
    service_window = int(time_factor * 2)

    # horizon allows for the feasibility of reaching nodes / returning from nodes within the global tw (service window)
    horizon = np.zeros((size, graph_size, 2))
    horizon[:] = [0, service_window]

    # epsilon = np.maximum(np.abs(rnds.standard_normal([size, graph_size])), 1 / time_factor)

    # sample earliest start times
    a = rnds.randint(horizon[..., 0], horizon[..., 1] / 2)
    sp = rnds.randint(0, graph_size, (size, 1))
    # a[np.arange(size)[:, None], sp] = 0
    a[:, 0] = 0
    # print(a.shape, sp.shape, a[np.arange(size)[:, None], sp].shape)

    # calculate latest start times b, which is
    # a + service_time_expansion x normal random noise, all limited by the horizon
    # and combine it with a to create the time windows
    # epsilon = rnds.uniform(0.25, .5, (a.shape))
    epsilon = rnds.uniform(dura_region[0], dura_region[1], (a.shape))
    # epsilon = rnds.uniform(.75, 1, (a.shape))
    duration = np.around(time_factor * epsilon)
    duration[:, 0] = service_window
    # print("duration.shape", duration.shape)
    # print("dura", epsilon.mean(), duration.mean(), time_factor)
    tw_high = np.minimum(a + duration, horizon[..., 1]).astype(int)

    tw = np.concatenate([a[..., None], tw_high[..., None]], axis=2).reshape(size, graph_size, 2)

    # print("tw_type", tw_type)
    if tw_type == "easy":
        tw[:, :, 0] = 0
    if tw_type == "none":
        tw[:, :, 0] = 0
        tw[:, :, 1] = service_window
    # print(a.shape, tw_high.shape)
    return tw


def gen_tw_window(size, graph_size, time_factor, dura_region, rnds, rand_lt=False,
                  tw=None, sp_rate=1., grp_sizes=2, expand_rate=1, tw_type="eval"):
    if tw is None:
        tw = np.zeros((size, graph_size, 2))
    if dura_region is None:
        dura_region = [.5, .75]

    for i in range(size):
        grp = rnds.permutation(graph_size - 1) + 1  # except node 0
        sp_size = int((graph_size - 1) * sp_rate)

        if isinstance(grp_sizes, int):
            grp_size = grp_sizes
        else:
            grp_size = rnds.choice(grp_sizes).item()

        grp_right = rnds.choice(sp_size - 1, grp_size - 1, replace=False).tolist()  # except n-1
        grp_right.sort()
        grp_right.append(sp_size - 1)
        left = 0
        grps = []
        grp_tfs = []
        grp_hs = []
        lt = 0
        for j in range(grp_size):
            cur_grp_size = grp_right[j] - left + 1
            tf = time_factor * (cur_grp_size / graph_size)
            if rand_lt:
                lt = rnds.randint(0, time_factor)
            rt = lt + tf * (1 + dura_region[1])
            grp_tfs.append(tf)
            grp_hs.append([lt, rt])

            # expand the random region
            mt = (lt + rt) / 2
            llt = mt - (mt - lt) * expand_rate
            rrt = mt + (rt - mt) * expand_rate
            ttf = tf * expand_rate

            st = rnds.randint(llt, llt + ttf, (cur_grp_size,))
            epsilon = rnds.uniform(0.5, .75, (st.shape))
            dura = np.round(epsilon * ttf)

            if tw_type == "eval":
                st[:] = llt
                dura[:] = rrt - llt

            gidx = grp[left: grp_right[j] + 1]

            # print("gidx", gidx.shape, tw.shape, cur_grp_size, left, grp_right[j]+1)
            tw[i, gidx, 0] = st
            tw[i, gidx, 1] = st + dura

            lt = rt
            left = grp_right[j] + 1

    tw[:, 0] = [0, time_factor * 2]
    return tw


def generate_tsptw_data(size, graph_size, rnds=None,
                        time_factor=100.0,
                        loc_factor=100,
                        tw_type=None, tw_duration="5075",
                        **kwargs):
    """
    """
    # print("real tf!!!", time_factor)
    rnds = np.random if rnds is None else rnds
    service_window = int(time_factor * 2)

    # sample locations
    # dloc = rnds.uniform(size=(size, 2))  # depot location
    nloc = rnds.uniform(size=(size, graph_size, 2)) * loc_factor  # node locations

    # static analyze
    # em = get_edge_matrix(nloc)
    # l = get_random_tour(em)
    # print("random tour length: ", np.min(l), np.max(l), np.mean(l))

    tw_type = tw_type.split("/")


    dura_region = {"5075": [.5, .75],
         "2550": [.25, .50],
         "1020": [.1, .2],
         "75100": [.75, 1.0],
        }
    if tw_duration == "random":
        tw_duration = np.random.choice(["5075", "1020", "2550"])
    if isinstance(tw_duration, str):
        dura_region = dura_region[tw_duration]
    else:
        dura_region = tw_duration
    # print(">> tw duration range from {} to {}".format(str(dura_region[0]), str(dura_region[1])))


    if tw_type[0] == "window":
        if tw_type[1] == "easy":
            er = 1.3
        else:
            er = 1.0
        if graph_size <= 21:
            grp_sizes = [2, 3]
        else:
            grp_sizes = list(range(2, 11))
        tw = gen_tw_window(size, graph_size, time_factor, dura_region, rnds,
                           expand_rate=er, grp_sizes=grp_sizes, tw_type=tw_type[1])
    elif tw_type[0] == "real":

        if tw_type[1] == "easy":
            tw = gen_tw_naive(size, graph_size, time_factor, dura_region, rnds, tw_type="hard")
        elif tw_type[1] == "eval":
            tw = np.zeros((size, graph_size, 2))
            tw[:, :, 0] = 0
            tw[:, :, 1] = time_factor * 2

        if graph_size <= 21:
            grp_sizes = 2
        else:
            grp_sizes = list(range(2, 7))
        tw = gen_tw_window(size, graph_size, time_factor, dura_region, rnds,
                           rand_lt=True, tw=tw, grp_sizes=grp_sizes, sp_rate=0.3, tw_type=tw_type[1])

    elif tw_type[0] == "naive":
        tw = gen_tw_naive(size, graph_size, time_factor, dura_region, rnds, tw_type=tw_type[1])
    elif tw_type[0] == "easy":
        dura_region = [.75, 1.75]
        tw = gen_tw_naive(size, graph_size, time_factor, dura_region, rnds, tw_type=tw_type[1])
    else:
        assert False, f"unkonwn tw type {tw_type}"
    # tw[:, :, 0] = 0
    # print("tw!!", tw[:, 0])
    # return [TSPTW_SET(*data) for data in zip(
    #     nloc.tolist(),
    #     tw,
    #     tw[..., 1] - tw[..., 0],
    #     [service_window] * size,
    #     [time_factor] * size,
    # )]
    return TSPTW_SET(node_loc=nloc,
                     node_tw=tw,
                     durations=tw[..., 1] - tw[..., 0],
                     service_window=[service_window] * size,
                     time_factor=[time_factor] * size,
                     loc_factor=[loc_factor] * size, )


def format_save_path(directory, args=None, note=''):
    """Formats the save path for saving datasets"""
    directory = os.path.normpath(os.path.expanduser(directory))

    fname = ''
    if args is not None:
        for k, v in args.items():
            if isinstance(v, str):
                fname += f'_{v}'
            else:
                fname += f'_{k}_{v}'

    fpath = os.path.join(directory, str(note) + fname + '.pkl')

    if os.path.isfile(fpath):
        print('Dataset file with same name exists already. Overwrite file? (y/n)')
        # a = input()
        # if a != 'y':
        #     print('Could not write to file. Terminating program...')
        #     sys.exit()

    return fpath


def save_dataset(dataset, filepath):
    """Saves data set to file path"""
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # check file extension
    assert os.path.splitext(filepath)[1] == '.pkl', "Can only save as pickle. Please add extension '.pkl'!"

    # save with pickle
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


# ## MAIN ## #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='./data', help="Create datasets in dir/problem (default 'data')")
    parser.add_argument("--name", default="train", type=str, help="Name of dataset (test, validation, ...)")
    parser.add_argument("--problem", type=str, default='tsptw',
                        help="Problem to sample: 'cvrp', 'cvrptw' or 'all' to generate all")
    parser.add_argument("--size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[50, 11, 21],
                        help="Sizes of problem instances (default: 20, 50, 100)")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--service_window', type=int, default=1000, help="Global time window of CVRP-TW")
    parser.add_argument('--service_duration', type=int, default=10, help="Global duration of service")
    parser.add_argument('--time_factors', type=float, nargs='+', default=[2500, 500, 1000],
                        help="Value to map from distances in [0, 1] to time units (transit times)")
    parser.add_argument('--tw_type', type=str, default="naive/hard", )
    # # medium
    # train: "naive/hard"  # uniform coord and tw
    # valid: "naive/hard"
    # # hard
    # train: "real/easy"  # mixed coord and shifted tw
    # valid: "real/eval"
    # # supplementary
    # train: "easy/none"  # TSP instances
    # train: "naive/easy"  # removes the constraint of earliest accessing time, i.e. tw_start = 0
    # train: "window/hard"  # Cluster coord and time windows from two different groups do not cover each other.
    parser.add_argument('--tw_expansion', type=float, default=3.0,
                        help="Expansion factor of service tw compared to service duration")

    args = parser.parse_args()

    problem = args.problem
    problems = ['cvrp', 'cvrptw'] if problem == 'all' else [problem]
    time_factors = args.time_factors
    time_factors = [s * 55 for s in args.graph_sizes]
    # print(time_factors, "tffff")

    for problem in problems:
        for i, graph_size in enumerate(args.graph_sizes):
            ddir = os.path.join(args.dir, problem)
            # filename = format_save_path(ddir, note=f"{problem}{graph_size}_{args.name}_seed{args.seed}")
            filename = format_save_path(ddir, note=f"{problem}{graph_size}_zhang_uniform")
            rnds = np.random.RandomState(args.seed)
            dataset = getattr(dg, f"generate_{problem}_data")(graph_size=graph_size, time_factor=time_factors[i],
                                                              **vars(args))
            service_time = np.zeros(shape=(args.size, graph_size))
            # Don't store travel time since it takes up much
            data = (dataset.node_loc, service_time, dataset.node_tw[:, :, 0], dataset.node_tw[:, :, 1])
            dataset = [attr.tolist() for attr in data]
            filedir = os.path.split(filename)[0]
            if not os.path.isdir(filedir):
                os.makedirs(filedir)
            with open(filename, 'wb') as f:
                pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)
            print("Save TSPTW dataset to {}".format(filename))

            # save_dataset(dataset, filename)
            # print(f"Dataset saved to {filename}")