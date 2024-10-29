import math
import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
# from utils.tensor_functions import compute_in_batches


class StateTSPTWInt(NamedTuple):
    # Fixed input
    timew: torch.Tensor  # Depot + loc
    dist: torch.Tensor  # (n + 1, n + 1), rounded to integer values with triangle inequality hack

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency the
    # timew, dist and before tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    PI_mask: torch.Tensor
    lengths: torch.Tensor
    current_time: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.timew.size(-1))  # n + 1 ! TODO look into this

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            i=self.i[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        assert visited_dtype == torch.uint8, \
            "Compressed mask not yet supported for TSPTW, first check code if depot is handled correctly"

        loc = input[:, : ,:2]
        timew = input[:, : , 2:]

        batch_size, n_loc, _ = loc.size()
        dist = torch.cdist(loc, loc, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        return StateTSPTWInt( #start from the depot
            timew=timew,
            dist=dist,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
                torch.cat([torch.ones((batch_size, 1, 1), dtype=torch.uint8, device=loc.device),
                torch.zeros( batch_size, 1, n_loc-1, dtype=torch.uint8, device=loc.device)], dim=-1)
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            PI_mask=(torch.cat([torch.ones((batch_size, 1, 1), dtype=torch.uint8, device=loc.device),
                           torch.zeros(batch_size, 1, n_loc - 1, dtype=torch.uint8, device=loc.device)], dim=-1)
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            current_time=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.ones(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.
        # We are at the depot so no need to add remaining distance
        return torch.where(
            self.prev_a == 0,  # If prev_a == 0, we have visited the depot prematurely and the solution is infeasible
            self.lengths.new_tensor(math.inf, dtype=torch.float),
            (self.lengths + self.dist[self.ids, self.prev_a, 0]).float()  # Return to first step which is always 0
        )

    def update(self, selected, ):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length (only needed for DP)
        d = self.dist[self.ids, self.prev_a, prev_a]

        # Compute new time
        lb, ub = torch.unbind(self.timew[self.ids, prev_a], -1)
        t = torch.max(self.current_time + d, lb)
        timeout = torch.clamp(t - ub, min=0)
        # assert (t <= ub).all()

        # Compute lengths (costs is equal to length)
        lengths = self.lengths + d  # (batch_dim, 1)

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, current_time=t, i=self.i + 1
        ), timeout

    def all_finished(self):
        # Exactly n steps since node 0 depot is visited before the first step already
        return self.i.item() >= self.timew.size(-2)

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_current_time(self):
        return self.current_time

    def get_mask(self):
        return self.visited > 0

    def get_PI_mask(self):
        batch_size, _, problem_size = self.visited.size()
        unvisited = torch.masked_select(torch.arange(problem_size).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, problem_size).to(self.visited.device), self.visited != 1).reshape(batch_size, 1, -1)
        two_step_unvisited = unvisited.unsqueeze(2).repeat(1, 1, problem_size - self.i, 1)
        diag_element = torch.diag_embed(torch.diagonal(two_step_unvisited, dim1=-2, dim2=-1))
        two_step_idx = torch.masked_select(two_step_unvisited, diag_element == 0).reshape(batch_size, 1, problem_size - self.i,  -1)

        # add arrival_time of the first-step nodes
        node_tw_start = self.timew[:, None, :, 0]
        first_step_tw_start = torch.masked_select(node_tw_start, self.visited != 1).reshape(batch_size, 1, -1)
        prev_action_expanded = self.prev_a.unsqueeze(2).expand(-1, -1, unvisited.shape[2])
        first_step_new_length = self.dist[self.ids.unsqueeze(2), prev_action_expanded, unvisited]
        first_step_arrival_time = torch.max(self.current_time.unsqueeze(2).expand(-1, -1, unvisited.shape[2]) + first_step_new_length, first_step_tw_start)

        # add arrival_time of the second-step nodes
        node_tw_end = self.timew[:, None, :, 1]
        two_step_tw_end = torch.masked_select(node_tw_end, (self.visited != 1)).reshape(batch_size, 1, -1)
        two_step_tw_end = two_step_tw_end.unsqueeze(2).repeat(1, 1, problem_size - self.i, 1)
        two_step_tw_end = torch.masked_select(two_step_tw_end, diag_element == 0).reshape(batch_size, 1,problem_size - self.i,-1)

        two_step_tw_start = torch.masked_select(node_tw_start, self.visited != 1).reshape(batch_size, 1, -1)
        two_step_tw_start = two_step_tw_start.unsqueeze(2).repeat(1, 1, problem_size - self.i, 1)
        two_step_tw_start = torch.masked_select(two_step_tw_start, diag_element == 0).reshape(batch_size, 1,problem_size - self.i,-1)

        first_action_expanded = unvisited.unsqueeze(-1).expand(-1, -1, -1, two_step_idx.shape[-1])
        second_step_new_length = self.dist[self.ids.unsqueeze(2).unsqueeze(-1), first_action_expanded, two_step_idx]
        first_step_arrival_time = first_step_arrival_time.unsqueeze(-1).repeat(1, 1, 1, problem_size - self.i - 1)
        second_step_arrival_time = torch.max(first_step_arrival_time + second_step_new_length, two_step_tw_start)

        # time window constraint
        #   current_time: the end time of serving the current node
        #   max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        # feasibility judgement
        round_error_epsilon = 1e-4
        infeasible_mark = (second_step_arrival_time > two_step_tw_end + round_error_epsilon)
        selectable = (infeasible_mark == False).all(dim=-1)

        PI_mask0 = torch.zeros((batch_size, 1, problem_size)).to(self.visited.device)
        unselectable = ~selectable
        unselectable_indices = unselectable.nonzero(as_tuple=False)
        unvisited_indices = unvisited[unselectable_indices[:, 0], unselectable_indices[:, 1], unselectable_indices[:, 2]]
        PI_mask0[unselectable_indices[:, 0], unselectable_indices[:, 1], unvisited_indices] = 1

        self = self._replace(PI_mask=PI_mask0)
        # return PI_mask0 > 0
        return self.PI_mask > 0

    def _get_mask(self, implementation):

        # A node can NOT be visited if it is already visited or arrival there directly would not be within time interval
        # Additionally 1: check BEFORE conditions (all befores must be visited)
        # Additionally 2: check 1 step ahead

        # Note: this always allows going to the depot, but that should always be suboptimal so be ok
        # Cannot visit if already visited or if length that would be upon arrival is too large to return to depot
        # If the depot has already been visited then we cannot visit anymore (and this solution is infeasible!)
        visited_ = self.visited

        verbose = False
        if verbose:
            from utils.profile import mem_report_cache
            print("Mem report before")
            mem_report_cache()

        if implementation == 'feasible_only':
            mask = (
                    visited_ | visited_[:, :, 0:1] |
                    # Check that time upon arrival is OK (note that this is implied by test3 as well)
                    (self.t[:, :, None] + self.dist[self.ids, self.prev_a, :] > self.timew[self.ids, :, 1])
            )
        elif implementation == 'no_test3':
            mask = (
                    visited_ | visited_[:, :, 0:1] |
                    # Check that time upon arrival is OK (note that this is implied by test3 as well)
                    (self.t[:, :, None] + self.dist[self.ids, self.prev_a, :] > self.timew[self.ids, :, 1]) |
                    # Check Test2 from Dumas et al. 1995: all predecessors must be visited
                    # I.e. if there is a node that is not visited and should be visited before, then it becomes infeasible
                    (((visited_ == 0)[:, :, None, :] & (self.before[self.ids])).sum(-1) > 0)
            )
        elif implementation == 'cache_before':

            mask = (
                    visited_ | visited_[:, :, 0:1] |
                    # Check Test3 from Dumas et al. 1995: via j, for all unvisited k i -> j -> k must be feasible
                    # So all k must either be visited or have t + t_ij + t_jk <= ub_k
                    # The lower bound for j is not used, if using this would make i -> j -> k infeasible then k must be before j
                    # and this is already checked by Test2
                    ((
                             (
                                     (
                                             (self.t[:, :, None, None] + self.dist[self.ids, self.prev_a, :,
                                                                         None] + self.dist[self.ids, :, :])
                                             > self.timew[self.ids, None, :, 1]
                                     ) | self.before[self.ids]
                             ) & (visited_ == 0)[:, :, None, :]
                     ).sum(-1) > 0)
            )

        elif implementation == 'compute_before':
            mask = (
                visited_ | visited_[:, :, 0:1] |
                # Check Test3 from Dumas et al. 1995: via j, there must not be some k such that i -> j -> k is infeasible
                # So all k must either be visited or have max(t + t_ij, lb_j) + t_jk < ub_k
                ((
                    (
                        (
                            torch.max(
                                self.t[:, :, None, None] + self.dist[self.ids, self.prev_a, :, None],
                                self.timew[self.ids, :, None, 0]
                            ) + self.dist[self.ids, :, :]
                        ) > self.timew[self.ids, None, :, 1]
                    ) & (visited_ == 0)[:, :, None, :]
                 ).sum(-1) > 0)
            )
        elif implementation == 'test2_separate':
            mask = (
                visited_ | visited_[:, :, 0:1] |
                # Check that time upon arrival is OK (note that this is implied by test3 as well)
                # (self.t[:, :, None] + self.dist[self.ids, self.prev_a, :] > self.timew[self.ids, :, 1]) |
                # Check Test2 from Dumas et al. 1995: all predecessors must be visited
                # I.e. if there is a node that is not visited and should be visited before, then it becomes infeasible
                (((visited_ == 0)[:, :, None, :] & (self.before[self.ids])).sum(-1) > 0) |
                # Check Test3 from Dumas et al. 1995: via j, there must not be some k such that i -> j -> k is infeasible
                # So all k must either be visited or have t + t_ij + t_jk < ub_k
                # The lower bound for j is not used, if using this would make i -> j -> k infeasible then k must be before j
                # and this is already checked by Test2
                ((
                    (
                        (self.t[:, :, None, None] + self.dist[self.ids, self.prev_a, :, None] + self.dist[self.ids, :, :])
                        > self.timew[self.ids, None, :, 1]
                    ) & (visited_ == 0)[:, :, None, :]
                ).sum(-1) > 0)
            )
        else:
            assert False, "Unkown implementation"

        if verbose:
            print("Mem report after")
            mem_report_cache()

        # Depot can always be visited, however this means that the solution is infeasible but we do to prevent nans
        # (so we do not hardcode knowledge that this is strictly suboptimal if other options are available)
        # mask[:, :, 1:].all(-1) == 0
        # If all non-depot actions are infeasible, we allow the depot to be visited. This is infeasible and therefore
        # ends the tour making it infeasible, but this way we do not get nan-problems etc. Simply we should make sure
        # in the cost function that this is accounted for
        infeas = mask[:, :, 1:].sum(-1) == mask[:, :, 1:].size(-1)
        # Allow depot as infeasible action if no other option, to prevent dead end when sampling
        # we must always have at least one feasible action
        mask[:, :, 0] = (infeas == 0)
        return mask, infeas

    def construct_solutions(self, actions):
        return actions
