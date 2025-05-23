import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
import pickle
from datetime import timedelta
from utils.functions import parse_softmax_temperature
mp = torch.multiprocessing.get_context('spawn')
import warnings
warnings.filterwarnings("ignore")

def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    print('  [*] Loading dataset from {}'.format(dataset_path))
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, ins_infsb, sol_infsb, durations = zip(*results)  # Not really costs since they should be negative
    costs = torch.tensor(costs)

    if opts.val_solution_path:
        print('  [*] Loading optimal solution from {}'.format(opts.val_solution_path))
        with open(opts.val_solution_path, 'rb') as f:
            opt_sol = pickle.load(f)[opts.offset: opts.offset+opts.val_size]
        grid_factor = 100.
        opt_sol = torch.tensor([i[0] / grid_factor for i in opt_sol])
        gap = ((costs[~torch.isinf(costs)] - opt_sol[~torch.isinf(costs)]) / opt_sol[~torch.isinf(costs)] * 100).mean() if (~torch.isinf(costs)).any() else 1000
    else:
        gap = 1000

    total_samples = len(dataset)
    ins_infsb_rate = torch.stack(ins_infsb).sum() / total_samples * 100
    sol_infsb_rate = (torch.stack(sol_infsb).sum() / (total_samples * width)) * 100
    print("Feasible average cost: {} +- {} (Gap: {}%)".format(costs[~torch.isinf(costs)].mean(), 2 * costs[~torch.isinf(costs)].std() / np.sqrt(costs[~torch.isinf(costs)].size(-1)), gap))
    print("Infeasible rate: Instance-level: {:.4f}% ({}/{}); Solution-level: {:.4f}% ({}/{})".format(ins_infsb_rate, torch.stack(ins_infsb).sum(), total_samples,
                                                                                                     sol_infsb_rate, torch.stack(sol_infsb).sum(), total_samples*1280))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs), ext
        ))
    else:
        out_file = opts.o

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)

    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    # assert opts.eval_batch_size == 1
                    # assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs, ins_infsb_num, sol_infsb_num  = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )

        duration = time.time() - start
        for seq, cost, ins_infsb, sol_infsb in zip(sequences, costs, ins_infsb_num, sol_infsb_num):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            elif model.problem.NAME in ("tspdl", "tsptw"):
                seq = [0] + np.trim_zeros(seq).tolist() # Add depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, ins_infsb, sol_infsb, duration))

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hardness", type=str, default="hard", help="constraint hardness level")
    parser.add_argument("--graph_size", type=int, default=50, help="problem size")
    parser.add_argument("--datasets", default=None, help="Filename of the dataset(s) to evaluate")
    parser.add_argument('--val_solution_path', type=str, default=None)
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help="Batch size to use during (baseline) evaluation")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, default=1280,
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str, default='sample',
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--generate_PI_mask', action='store_true')
    parser.add_argument('--model', type=str, default="pretrained/tsptw50_hard/AM_star_PIP-D/epoch-99.pt")

    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=1280, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--CUDA_VISIBLE_ID', default="0",
                        help='Make specific id of cuda visible and use them instead of all available cuda')


    opts = parser.parse_args()

    if opts.datasets is None:
        opts.datasets = f"../data/TSPTW/tsptw{opts.graph_size}_{opts.hardness}.pkl"
    if opts.val_solution_path is None:
        opts.val_solution_path = f"../data/TSPTW/lkh_tsptw{opts.graph_size}_{opts.hardness}.pkl"

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_ID
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device("cuda" if use_cuda else "cpu")
    # opts.device = torch.device("cuda:{}".format(opts.CUDA_VISIBLE_ID) if use_cuda else "cpu")
    # if use_cuda:
    #     torch.cuda.set_device(int(opts.CUDA_VISIBLE_ID))
    print('device: ', opts.device)
    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    width = opts.width if opts.width is not None else 0

    # for dataset_path in opts.datasets:
    dataset_path = opts.datasets
    eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
