import torch
from torch.distributions import Categorical
import random
import itertools
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from functools import cached_property
import concurrent.futures
from itertools import combinations
import os
from tsptw_baseline import get_lkh_executable, get_gvns_executable, solve_lkh_log, solve_gvns_log, run_all_in_pool
from sklearn.metrics import confusion_matrix

def run_lkh(args):
    directory, name, *args = args
    timew, dist, route = args
    executable = get_lkh_executable()
    return solve_lkh_log(
        executable,
        directory, name,
        timew, dist, route,
        runs=1, max_trials=1,
        seed = 2023,
        makespan=False
    )

def run_gvns(args):
    # FIXME: not suuported yet
    directory, name, *args = args
    timew, dist, route = args
    executable = get_gvns_executable()
    return solve_gvns_log(
        executable,
        directory, name,
        timew, dist, route,
        runs=1,
    )

class ACO():
    def __init__(
        self,  # 0: depot
        distances, # (n, n)
        tw,   # (n, 2)
        generate_simulated_mask,
        n_ants=20, 
        decay=0.9,
        alpha=1,
        beta=1,
        elitist=False,
        min_max=False,
        pheromone=None,
        heuristic=None,
        min=None,
        adaptive=False,
        device='cpu',
        local_search="lkh",
        positions = None,
        normalize_phe = False,
        dual_decoder = None,
        embedding = None,
        pyg_data = None,
        sl_mask = None,
        sl_mask_type = "AR"
    ):
        
        self.problem_size = len(distances)
        self.distances = distances
        self.tw = tw
        self.generate_simulated_mask = generate_simulated_mask
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or adaptive
        self.min_max = min_max
        self.adaptive = adaptive
        self.positions = positions
        
        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = None
        
        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            if min_max: # fixme
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

        self.normalize_phe = self.problem_size if normalize_phe else 1.

        # fixme
        if self.adaptive:
            self.elite_pool = []

        assert local_search in [None, "gvns", "lkh"]
        self.local_search_type = local_search

        self.heuristic = 1 / (distances + 1e-10) if heuristic is None else heuristic

        self.dual_decoder = dual_decoder
        self.embedding = embedding
        if sl_mask is not None:
            self.sl_mask = sl_mask
        else:
            if dual_decoder is not None and embedding is not None:
                if sl_mask_type == "NAR":
                    self.sl_mask = dual_decoder(embedding)
                    self.sl_mask = dual_decoder.reshape(pyg_data, self.sl_mask) + 1e-10
                else:
                    self.pyg = pyg_data
                    self.sl_mask = None

        self.shortest_path = None
        self.lowest_cost = float('inf')
        self.best_total_timeout = float('inf')
        self.best_timeout_nodes = self.problem_size

        self.device = device

    def sample(self, invtemp=1.0, return_auc=False):
        paths, log_probs, timeout, out = self.gen_path(require_prob=True, invtemp=invtemp, return_auc=return_auc)  # type: ignore
        timeout_penalty, timeout_nodes_penalty = self.calculate_penalty(timeout)
        costs = self.gen_path_costs(paths)
        return costs, timeout_penalty, timeout_nodes_penalty, log_probs, paths, out

    def sample_local_search(self, invtemp=1.0, return_loss=False, return_auc=False):
        paths, log_probs, _,  out = self.gen_path(require_prob=True, invtemp=invtemp, return_loss=return_loss, return_auc=return_auc)
        paths_raw = paths.clone()
        timeout_raw, timeout_nodes_raw = self.gen_timeout_penalty(paths_raw)
        costs_raw = self.gen_path_costs(paths_raw).detach()

        paths = self.local_search(paths)
        costs = self.gen_path_costs(paths).detach()
        timeout, timeout_nodes = self.gen_timeout_penalty(paths)

        return costs, log_probs, paths, costs_raw, paths_raw, timeout, timeout_nodes, timeout_raw, timeout_nodes_raw, out

    @cached_property
    @torch.no_grad()
    def heuristic_dist(self):
        heu = self.heuristic.detach().cpu().numpy()  # type: ignore
        return (1 / (heu/heu.max(-1, keepdims=True) + 1e-5))

    @torch.no_grad()
    def run(self, n_iterations, local_search =True):
        for _ in range(n_iterations):
            paths, timeout, _ = self.gen_path(require_prob=False)
            _paths = paths.clone()   # type: ignore

            # local search
            if local_search:
                paths = self.local_search(paths)

            costs = self.gen_path_costs(paths)
            total_timeout, timeout_nodes = self.gen_timeout_penalty(paths) # shape: (n_ants, )

            # improved = False
            is_feasible = (total_timeout == 0)
            if is_feasible.sum() != 0: # have feasible solutions
                best_cost, best_idx = costs[is_feasible].min(dim=0) # feasible only
                original_indices = torch.where(is_feasible)[0]
                best_idx = original_indices[best_idx]
                if best_cost < self.lowest_cost:
                    self.shortest_path = paths[:, best_idx].clone()  # type: ignore
                    self.lowest_cost = best_cost.item()

                    if self.min_max:
                        max = self.problem_size / self.lowest_cost
                        if self.max is None:
                            self.pheromone *= max / self.pheromone.max()
                        self.max = max
                # improved = True

            # if improved:
            self.update_pheromone(paths, (costs+total_timeout+timeout_nodes)/ self.normalize_phe)

        # Pairwise Jaccard similarity between paths
        edge_sets = []
        _paths = _paths.T.cpu().numpy()  # type: ignore
        for _p in _paths:
            edge_sets.append(set(map(frozenset, zip(_p[:-1], _p[1:]))))

        # Diversity
        jaccard_sum = 0
        for i, j in combinations(range(len(edge_sets)), 2):
            jaccard_sum += len(edge_sets[i] & edge_sets[j]) / len(edge_sets[i] | edge_sets[j])
        diversity = 1 - jaccard_sum / (len(edge_sets) * (len(edge_sets) - 1) / 2)

        return self.lowest_cost, diversity, total_timeout

    @torch.no_grad()
    def update_pheromone(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 

        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += 1.0/best_cost
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += 1.0/best_cost

        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
                self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost

        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max  # type: ignore

    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        # assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    @torch.no_grad()
    def calculate_penalty(self, timeout):
        total_timeout = timeout.sum(dim=-1)
        timeout_nodes = torch.where(timeout > 0, torch.ones_like(timeout),timeout).sum(-1).int()
        return total_timeout, timeout_nodes

    @torch.no_grad()
    def gen_timeout_penalty(self, paths):
        paths = paths.T # shape: (n_ants, problem_size)
        current_time = torch.zeros(self.n_ants)
        timeouts = torch.zeros(self.n_ants, 1)
        for i in range(self.problem_size-1):
            current_node = paths[:, i+1]
            travel_time = self.distances[paths[:, i], paths[:, i+1]]
            tw_start = self.tw[current_node, 0]
            current_time = torch.max(current_time + travel_time, tw_start)
            tw_end = self.tw[current_node, 1]
            timeout = torch.clamp(current_time - tw_end, min=0)
            timeouts = torch.cat((timeouts, timeout.unsqueeze(-1)), dim=-1)
        total_timeout = timeouts.sum(dim=-1)
        timeout_nodes = torch.where(timeouts > 0, torch.ones_like(timeouts),timeouts).sum(-1).int()
        return total_timeout, timeout_nodes

    def gen_path(self, require_prob=False, invtemp=1.0, paths=None, remove_infsb_ants=False, return_loss = False, return_auc= False):
        # start from the first node
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # mask == 1 means selectable
        visit_mask = self.update_visit_mask(visit_mask, actions)

        simulated_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # sm_matrix = torch.zeros(size=(self.n_ants, self.problem_size, self.problem_size))
        current_time = torch.zeros(size=(self.n_ants,), device=self.device)
        sm_flag = 0
        infsb_num = 0
        fsb_num = 0
        true_infsb = 0
        true_fsb = 0
        sl_loss_list = torch.zeros(size=(0,))
        if self.generate_simulated_mask:
            simulated_mask = self.update_simulated_mask(current_time, visit_mask, actions)
            # sm_matrix = self.update_sm_matrix(sm_matrix, simulated_mask, actions)
            sm_flag = 1
        # print(self.generate_simulated_mask)

        prob_mat = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
        prev = actions

        paths_list = [actions]  # paths_list[i] is the ith move (tensor) for all ants
        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        timeout_list = [torch.zeros(size=(self.n_ants,), device=self.device)]
        # done = self.check_done(visit_mask, actions)

        ##################################################
        # given paths
        feasible_idx = torch.arange(self.n_ants, device=self.device)
        ##################################################
        for i in range(self.problem_size - 1):
            selected = paths[i + 1] if paths is not None else None
            actions, log_probs = self.pick_move(prob_mat[prev], visit_mask,
                                                simulated_mask if sm_flag==1 else torch.ones(size=(self.n_ants, self.problem_size)),
                                                require_prob, invtemp, selected)
            sm_flag = 0 # already use the generated simulated mask
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            current_time = self.update_current_time(current_time, prev, actions)
            if self.generate_simulated_mask and i < self.problem_size-3:
                sm_flag = 1
                with torch.no_grad():
                    simulated_mask = self.update_simulated_mask(current_time, visit_mask, actions)
                    # sm_matrix = self.update_sm_matrix(sm_matrix, simulated_mask, actions)
                if (return_loss or return_auc):
                    if self.sl_mask is not None:
                        predicted = self.sl_mask[actions]
                    else:
                        context_embedding, candidate_embedding = self.dual_decoder.get_embedding(self.pyg, self.embedding, prev, actions)
                        context_embedding = context_embedding.unsqueeze(-2).repeat(1, self.problem_size, 1)
                        current_time_batch = current_time.unsqueeze(-1).unsqueeze(-1).repeat(1, self.problem_size,1)
                        context = torch.cat([context_embedding, candidate_embedding, current_time_batch], dim=-1) # n_ants * problem_size * (32+32+1)
                        predicted = self.dual_decoder(context)
                    probs_sl = torch.masked_select(predicted, predicted > 1e-5) # remove those not-considered edges
                    labels = (torch.masked_select(simulated_mask, predicted > 1e-5) == 0).float() # 1 means infeasible (unselectable)
                    if labels.sum() != labels.view(-1).shape[0] and labels.sum() != 0:
                        if return_loss:
                            edge_labels = (labels != 0).int().cpu().numpy().flatten()
                            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
                            probs_sl = torch.clamp(probs_sl, min=1e-7, max=1 - 1e-7)  # add a clamp to avoid numerical instability
                            sl_loss = - edge_cw[1] * (labels * torch.log(probs_sl)) - edge_cw[0] * ( ((1 - labels) * torch.log(1 - probs_sl)))
                            sl_loss_list = torch.cat([sl_loss_list, sl_loss.mean().unsqueeze(-1)])
                        if return_auc:
                            tn, fp, fn, tp = confusion_matrix((labels > 0.5).int().cpu().numpy(), (probs_sl > 0.5).int().cpu().numpy()).ravel()
                            infsb_num += (fn + tp)
                            fsb_num += (tn + fp)
                            true_infsb +=  tp
                            true_fsb += tn
            elif (not self.generate_simulated_mask and self.dual_decoder is not None) and i < self.problem_size - 3:
                # use sl mask
                sm_flag = 1
                with torch.no_grad():
                    if self.sl_mask is not None:
                        simulated_mask = self.sl_mask[actions]
                    else:
                        context_embedding, candidate_embedding = self.dual_decoder.get_embedding(self.pyg, self.embedding, prev,actions)
                        context_embedding = context_embedding.unsqueeze(-2).repeat(1, self.problem_size, 1)
                        current_time_batch = current_time.unsqueeze(-1).unsqueeze(-1).repeat(1, self.problem_size, 1)
                        context = torch.cat([context_embedding, candidate_embedding, current_time_batch],dim=-1)  # n_ants * problem_size * (32+32+1)
                        simulated_mask = (self.dual_decoder(context) <= 0.5).float() # 1 means feasible (selectable)
                # print(i, simulated_mask.sum(-1))

            timeout = self.update_timeout(current_time, actions)
            timeout_list.append(timeout)

            ##################################################
            # may generate infeasible solutions
            if remove_infsb_ants:
                infeasible_number = (timeout > 0).sum()
                # remove infeasible ants
                if infeasible_number > 0:
                    is_feasible = timeout == 0
                    feasible_idx = feasible_idx[is_feasible]  # type: ignore

                    actions = actions[is_feasible]
                    visit_mask = visit_mask[is_feasible]
                    simulated_mask = simulated_mask[is_feasible]
                    current_time = current_time[is_feasible]

                    paths_list = [p[is_feasible] for p in paths_list]
                    timeout_list = [tl[is_feasible] for tl in timeout_list]
                    if require_prob:
                        log_probs_list = [l_p[is_feasible] for l_p in log_probs_list]
                    if paths is not None:
                        paths = paths[:, is_feasible]

                    self.n_ants -= infeasible_number
            ##################################################

            prev = actions
        out=[]
        if return_auc:
            out = [true_infsb, infsb_num, true_fsb, fsb_num]
        if return_loss:
            out.append(sl_loss_list.mean())
        if require_prob:
            if paths is not None:
                return torch.stack(paths_list), torch.stack(log_probs_list), feasible_idx, out   # type: ignore
                # FIXME
            return torch.stack(paths_list), torch.stack(log_probs_list), torch.stack(timeout_list).T, out
        else:
            return torch.stack(paths_list), torch.stack(timeout_list), out

    def pick_move(self, dist, visit_mask, simulated_mask, require_prob, invtemp=1.0, selected=None):
        all_infsb = ((visit_mask * simulated_mask) == 0).all(-1).unsqueeze(-1).expand(self.n_ants, self.problem_size)
        simulated_mask = torch.where(all_infsb, visit_mask, simulated_mask)
        dist = (dist ** invtemp) * visit_mask * simulated_mask  # shape: (n_ants, p_size)
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        dist = Categorical(probs=dist)
        actions = selected if selected is not None else dist.sample()  # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None  # shape: (n_ants,)
        return actions, log_probs

    @torch.no_grad()
    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        return visit_mask

    @torch.no_grad()
    def update_simulated_mask(self,current_time, visit_mask, actions):
        # unvisited index
        nonzero_indices = torch.nonzero(visit_mask == 1)
        num_unvisited_per_path = (visit_mask == 1).sum(1)[0]
        first_idx = nonzero_indices[:, 1].view(self.n_ants, num_unvisited_per_path)
        first_arrival_times = self.distances[actions[:, None], first_idx] + current_time.unsqueeze(1)
        first_tw_start = self.tw[:, 0].expand(self.n_ants, self.problem_size).gather(1, first_idx)
        first_arrival_times = torch.max(first_tw_start, first_arrival_times)

        second_idx = first_idx.unsqueeze(1).repeat(1, num_unvisited_per_path, 1)
        diag_element = torch.diag_embed(torch.diagonal(second_idx, dim1=-2, dim2=-1))
        second_idx = torch.masked_select(second_idx, diag_element == 0).reshape(self.n_ants, num_unvisited_per_path, -1)
        second_tw_end = self.tw[:, 1].expand(self.n_ants, self.problem_size).gather(1, first_idx)
        second_tw_end = second_tw_end.unsqueeze(1).repeat(1, num_unvisited_per_path, 1)
        second_tw_end = torch.masked_select(second_tw_end, diag_element == 0).reshape(self.n_ants, num_unvisited_per_path, -1)
        second_tw_start = first_tw_start.unsqueeze(1).repeat(1, num_unvisited_per_path, 1)
        second_tw_start = torch.masked_select(second_tw_start, diag_element == 0).reshape(self.n_ants, num_unvisited_per_path, -1)
        second_arrival_times = self.distances[first_idx.unsqueeze(-1), second_idx]
        second_arrival_times = second_arrival_times + first_arrival_times.unsqueeze(-1)
        second_arrival_times = torch.max(second_tw_start, second_arrival_times)

        round_error_epsilon = 0.00001
        infeasible_mark = (second_arrival_times > second_tw_end + round_error_epsilon)
        selectable = (infeasible_mark == False).all(dim=-1)

        simulated_mask = torch.zeros_like(visit_mask)
        selected_indices = selectable.nonzero(as_tuple=False)
        unvisited_indices = first_idx[selected_indices[:, 0], selected_indices[:, 1]]
        simulated_mask[selected_indices[:, 0], unvisited_indices] = 1.
        # first index: ant idx; second index: candidate idx

        # simulated_mask = torch.ones_like(visit_mask)
        # unselectable = (selectable == False)
        # unselectable_indices = unselectable.nonzero(as_tuple=False)
        # unvisited_indices = first_idx[unselectable_indices[:, 0], unselectable_indices[:, 1]]
        # simulated_mask[unselectable_indices[:, 0], unvisited_indices] = 0.

        return simulated_mask

    @torch.no_grad()
    def update_sm_matrix(self, sm_matrix, simulated_mask, actions):
        sm_matrix[torch.arange(self.n_ants), actions] = simulated_mask
        return sm_matrix
    @torch.no_grad()
    def update_current_time(self, current_time, prev, actions):
        tw_start = self.tw[actions, 0]
        new_length = self.distances[prev, actions]
        # wait until time window starts
        return torch.max(current_time + new_length, tw_start)

    @torch.no_grad()
    def update_timeout(self, current_time, actions):
        # timeout value of the selected node = current time - tw_end
        # negative value means current time < tw_end, turn it into 0
        timeout = torch.clamp(current_time - self.tw[actions, 1], min=0.)
        return timeout

    @cached_property
    @torch.no_grad()
    def distances_cpu(self):
        return self.distances.cpu().numpy()

    @cached_property
    @torch.no_grad()
    def time_window_cpu(self):
        return self.tw.cpu().numpy()
    
    @cached_property
    @torch.no_grad()
    def positions_cpu(self):
        return self.positions.cpu().numpy() if self.positions is not None else None

    @torch.no_grad()
    def local_search(self, paths):

        paths = paths.T
        method = self.local_search_type
        directory = "tmp" + method
        if not os.path.exists(directory):
            os.makedirs(directory)
        use_multiprocessing = True

        if method in ("lkh", "lkhms"):
            # Note: only processing n items is handled by run_all_in_pool
            n_ants = len(paths)
            dataset = [(self.tw.cpu().numpy(), self.distances.cpu().numpy(), paths[i].cpu().numpy()) for i in range(n_ants)]
            results, parallelism = run_all_in_pool(
                run_lkh,
                directory, dataset,
                n_cpus = 28,
                use_multiprocessing=use_multiprocessing,
            )
        elif method == "gvns":
            assert 0, "not supported yet!!!!"
        try:
            return torch.tensor(np.array(results)).T
        except:
            return paths.T







