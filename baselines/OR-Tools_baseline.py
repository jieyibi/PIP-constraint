import os, sys
import time
import argparse
import numpy as np
from datetime import timedelta
from functools import partial
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
sys.path.insert(0, "..")  # for utils
from utils import check_extension, load_dataset, save_dataset, run_all_in_pool

SPEED = 1.0
SCALE = 1000  # EAS uses 1000, while AM uses 100000
TIME_HORIZON = 3  # the tw_end for the depot node, all vehicles should return to depot before T
# TIME_HORIZON = 5500  # the tw_end for the depot node, all vehicles should return to depot before T
DRAFT_HORIZON = 1000  # the maximal draft limit
CNT = 0

def create_data_model(depot, loc, demand=None, capacity=None, route_limit=None, service_time=None, tw_start=None, tw_end=None, draft=None, grid_size=1, problem="CVRP"):
    """
        Stores the data for the problem
    """
    data = {}
    to_int = lambda x: int(x / grid_size * SCALE + 0.5)

    data['depot'] = 0
    locations = [depot] + loc
    data['locations'] = [(to_int(x), to_int(y)) for (x, y) in locations]
    data['real_locations'] = locations
    data['num_locations'] = len(data['locations'])  # the number of customer nodes + depot (exclude dummy_depot if any)

    if problem not in ["TSPTW", "TSPDL"]:
        data['demands'] = [0] + list(demand)
        data['num_vehicles'] = len(loc)
        data['vehicle_capacity'] = int(capacity)
    else:
        data['num_vehicles'] = 1

    # For Open Route (e.g., OVRP)
    data['dummy_depot'] = None
    if problem in ["OVRP", "OVRPB", "OVRPL", "OVRPTW", "OVRPBL", "OVRPLTW", "OVRPBTW", "OVRPBLTW"]:
        data['dummy_depot'] = data['num_locations']  # len(loc) + 1

    # For TW
    if problem in ["VRPTW", "VRPBTW", "VRPLTW", "OVRPTW", "VRPBLTW", "OVRPLTW", "OVRPBTW", "OVRPBLTW"]:
        # for depot: [0., 0.] -> Cumul(depot) = 0: vehicle must be at time 0 at depot
        data['time_windows'] = [(to_int(e), to_int(l)) for e, l in zip([0]+tw_start, [0]+tw_end)]
        data['service_time'] = to_int(service_time[0])

    if problem in ["TSPDL"]:
        data['demands'] = [int(i) for i in demand]
        data['draft'] = [int(i) for i in draft]
        # data['draft'] = [50 for i in draft]
    elif problem in ["TSPTW"]:
        # for depot: [0., 0.] -> Cumul(depot) = 0: vehicle must be at time 0 at depot
        data['time_windows'] = [(to_int(e), to_int(l)) for e, l in zip([0] + tw_start[1:], [0] + tw_end[1:])]
        data['service_time'] = 0

    # For duration limit
    if problem in ["VRPL", "VRPBL", "VRPLTW", "OVRPL", "VRPBLTW", "OVRPBL", "OVRPLTW", "OVRPBLTW"]:
        data['distance'] = to_int(route_limit)

    return data


#######################
# Problem Constraints #
#######################
def Euc_distance(position_1, position_2):
    return int(np.sqrt((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2))


def create_distance_evaluator(data):
    """
        Creates callback to return distance between points.
    """
    _distances = {}
    # precompute distance between location to have distance callback in O(1)
    for from_node in xrange(data['num_locations']):
        _distances[from_node] = {}
        for to_node in xrange(data['num_locations']):
            if from_node == to_node:
                _distances[from_node][to_node] = 0
            else:
                _distances[from_node][to_node] = (Euc_distance(data['locations'][from_node], data['locations'][to_node]))
    data['distance_matrix'] = _distances

    def distance_evaluator(manager, from_index, to_index):
        """
            Returns the manhattan distance between the two nodes.
        """
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node == data['dummy_depot']:  # for open route
            return 0
        else:
            return _distances[from_node][to_node]

    return distance_evaluator


def add_distance_constraints(routing, data, distance_evaluator_index):
    """
        Adds duration limit constraint.
    """
    routing.AddDimension(
        distance_evaluator_index,
        0,  # null distance slack
        data['distance'],
        True,  # start cumul to zero
        'Distance')


def create_demand_evaluator(data):
    """
        Creates callback to get demands at each location.
    """
    _demands = data['demands']

    def demand_evaluator(manager, from_index):
        """
            Returns the demand of the current node.
        """
        from_node = manager.IndexToNode(from_index)
        return _demands[from_node]

    return demand_evaluator


def add_capacity_constraints(routing, data, demand_evaluator_index, problem="CVRP"):
    """
        Adds capacity constraint.
    """
    if problem in ["VRPB", "OVRPB", "VRPBL", "VRPBTW", "VRPBLTW", "OVRPBL", "OVRPBTW", "OVRPBLTW"]:  # Optional for VRPB and VRPBL
        # Note (Only for the problems with backhauls): need to relax the capacity constraint, otherwise OR-Tools cannot find initial feasible solution;
        # However, it may be problematic since the vehicle could decide how many loads to carry from depot in this case.
        routing.AddDimension(
            demand_evaluator_index,
            0,  # null capacity slack
            data['vehicle_capacity'],
            False,  # don't force start cumul to zero
            'Capacity')
    else:
        routing.AddDimension(
            demand_evaluator_index,
            0,  # null capacity slack
            data['vehicle_capacity'],
            True,  # start cumul to zero
            'Capacity')


def create_time_evaluator(data):
    """
        Creates callback to get total times between locations.
    """

    def travel_time(data, from_node, to_node):
        """
            Gets the travel times between two locations.
        """
        return int(data['distance_matrix'][from_node][to_node] / SPEED)

    _total_time = {}
    # precompute total time to have time callback in O(1)
    for from_node in xrange(data['num_locations']):
        _total_time[from_node] = {}
        for to_node in xrange(data['num_locations']):
            if from_node == to_node:
                _total_time[from_node][to_node] = 0
            elif from_node == data['depot']:  # depot node -> customer node
                _total_time[from_node][to_node] = travel_time(data, from_node, to_node)
            else:
                _total_time[from_node][to_node] = int(data['service_time'] + travel_time(data, from_node, to_node))
    data['time_matrix'] = _total_time

    def time_evaluator(manager, from_index, to_index):
        """
            Returns the total time (service_time + travel_time) between the two nodes.
        """
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node == data['dummy_depot']:
            return 0
        else:
            return _total_time[from_node][to_node]

    return time_evaluator

def add_draft_constraints(routing, manager, data, demand_evaluator_index, problem="TSPDL"):
    """
        Add Draft Limit constraint.
    """
    capacity = 'Capacity'
    routing.AddDimension(
        demand_evaluator_index,
        0,  # null capacity slack
        DRAFT_HORIZON,  # maximum draft limit per vehicle
        True,  # start cumul to zero
        capacity)
    capacity_dimension = routing.GetDimensionOrDie(capacity)

    # Add draft limit constraints for each location
    for location_idx, draft in enumerate(data['draft']):
        if location_idx == data['depot'] or location_idx == data['dummy_depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        capacity_dimension.CumulVar(index).SetRange(0, int(draft))

        # routing.AddToAssignment(time_dimension.SlackVar(index))

    for vehicle_id in xrange(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        # Cumul(depot).SetRange(0, 0) -> vehicle must be empty at depot
        capacity_dimension.CumulVar(index).SetRange(0, 0)

def add_time_window_constraints(routing, manager, data, time_evaluator_index, grid_size=1, time_horizon=TIME_HORIZON):
    """
        Add Global Span constraint.
    """
    time = 'Time'
    horizon = int(time_horizon / grid_size * SCALE + 0.5)
    routing.AddDimension(
        time_evaluator_index,
        horizon,  # allow waiting time
        horizon,  # maximum time per vehicle
        False,  # don't force start cumul to zero since we are giving TW to start nodes
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Add time window constraints for each location except depot
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot'] or location_idx == data['dummy_depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
        routing.AddToAssignment(time_dimension.SlackVar(index))

    # Add time window constraints for each vehicle start and end node
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for vehicle_id in xrange(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        # Cumul(depot).SetRange(0, 0) -> vehicle must be at time 0 at depot
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
        # for open route
        if data['dummy_depot']:
            index = routing.End(vehicle_id)
            time_dimension.CumulVar(index).SetRange(0, horizon)
            # Warning: Slack var is not defined for vehicle's end node
            # routing.AddToAssignment(time_dimension.SlackVar(index))


###########
# Printer #
###########
def print_solution(data, manager, routing, assignment, problem="CVRP", log_file=None):
    """
        Only print route, and calculate cost (total distance).
    """

    def calc_vrp_cost(depot, loc, tour, problem):
        assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
        loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
        sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
        if problem in ["CVRP", "VRPB", "VRPL", "VRPTW", "VRPBL", "VRPLTW", "VRPBTW", "VRPBLTW", "TSPDL", "TSPTW"]:
            return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
        elif problem in ["OVRP", "OVRPB", "OVRPL", "OVRPTW", "OVRPBL", "OVRPLTW", "OVRPBTW", "OVRPBLTW"]:  # no need to return to depot
            full_tour = [0] + tour + [0]
            not_to_depot = np.array(full_tour)[1:] != 0
            return (np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1) * not_to_depot).sum()
        else:
            raise NotImplementedError

    route = []
    total_distance, total_load = 0, 0
    # distance_dimension = routing.GetDimensionOrDie('Distance')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    for vehicle_id in xrange(data['num_vehicles']):
        if not routing.IsVehicleUsed(vehicle=vehicle_id, assignment=assignment):
            continue
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            if problem == "TSPDL":
                plan_output += ' {0} Load({1}) <= Draft({2})->'.format(
                    manager.IndexToNode(index),
                    assignment.Value(load_var),
                    data["draft"][manager.IndexToNode(index)])
            else:
                plan_output += ' {0} Load({1}) ->'.format(
                    manager.IndexToNode(index),
                    assignment.Value(load_var))
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            # distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)  # Bugs: always output 0 if given variable index, don't know why
            from_node, to_node = manager.IndexToNode(previous_index), manager.IndexToNode(index)
            to_node = to_node if to_node != data['dummy_depot'] else data['depot']
            distance += data['distance_matrix'][from_node][to_node]  # use distance matrix instead

        load_var = capacity_dimension.CumulVar(index)
        # dist_var = distance_dimension.CumulVar(index)
        plan_output += ' {0} Load({1})\n'.format(
            manager.IndexToNode(index),
            assignment.Value(load_var))
        # assert distance == assignment.Value(dist_var), ">> Distance not match!"
        plan_output += 'Distance of the route: {}\n'.format(distance)
        plan_output += 'Load of the route: {}\n'.format(assignment.Value(load_var))
        if log_file:
            print(plan_output, file=log_file)
        total_distance += distance
        total_load += assignment.Value(load_var)

    # double check
    cost = calc_vrp_cost(data['real_locations'][0], data['real_locations'][1:], route[1:], problem)
    if log_file:
        print('Route: {}'.format(route + [data['depot']]), file=log_file)
        print('Total Load of all routes: {}'.format(total_load), file=log_file)
        print('Total Distance of all routes: {} (Routing Error may exist)'.format(total_distance / SCALE), file=log_file)
        print('Final Result - Cost of the obtained solution: {}'.format(cost), file=log_file)

    return cost, route[1:]


def print_solution_tsptw(data, manager, routing, assignment, problem="TSPTW", log_file=None):
    """
        Only print route, and calculate cost (total distance).
    """

    def calc_vrp_cost(depot, loc, tour, problem):
        assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
        loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
        sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
        if problem in ["TSPTW"]:
            return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
        else:
            raise NotImplementedError

    route = []
    total_distance = 0
    # distance_dimension = routing.GetDimensionOrDie('Distance')
    # capacity_dimension = routing.GetDimensionOrDie('Capacity')
    for vehicle_id in xrange(data['num_vehicles']):
        if not routing.IsVehicleUsed(vehicle=vehicle_id, assignment=assignment):
            continue
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            # load_var = capacity_dimension.CumulVar(index)
            # plan_output += ' {0} Load({1}) ->'.format(manager.IndexToNode(index), assignment.Value(load_var))
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            # distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)  # Bugs: always output 0 if given variable index, don't know why
            from_node, to_node = manager.IndexToNode(previous_index), manager.IndexToNode(index)
            to_node = to_node if to_node != data['dummy_depot'] else data['depot']
            distance += data['distance_matrix'][from_node][to_node]  # use distance matrix instead

        # load_var = capacity_dimension.CumulVar(index)
        # dist_var = distance_dimension.CumulVar(index)
        # plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index), assignment.Value(load_var))
        # assert distance == assignment.Value(dist_var), ">> Distance not match!"
        plan_output += 'Distance of the route: {}\n'.format(distance)
        # plan_output += 'Load of the route: {}\n'.format(assignment.Value(load_var))
        if log_file:
            print(plan_output, file=log_file)
        total_distance += distance
        # total_load += assignment.Value(load_var)

    # double check
    cost = calc_vrp_cost(data['real_locations'][0], data['real_locations'][1:], route[1:], problem)
    if log_file:
        print('Route: {}'.format(route + [data['depot']]), file=log_file)
        # print('Total Load of all routes: {}'.format(total_load), file=log_file)
        print('Total Distance of all routes: {} (Routing Error may exist)'.format(total_distance / SCALE), file=log_file)
        print('Final Result - Cost of the obtained solution: {}'.format(cost), file=log_file)

    return cost, route[1:]


def solve_or_tools_log(directory, name, depot, loc, demand, capacity, route_limit=None, service_time=None, tw_start=None, tw_end=None, draft=None,
                       timelimit=3600, grid_size=1, seed=1234, problem="CVRP"):
    """
        OR-Tools to solve VRP variants, Ref to:
            https://developers.google.com/optimization/routing/vrptw
            https://developers.google.com/optimization/routing/routing_options
            https://github.com/google/or-tools/issues/1051
            https://github.com/google/or-tools/issues/750
    """

    tour_filename = os.path.join(directory, "{}.or_tools.tour".format(name))
    output_filename = os.path.join(directory, "{}.or_tools.pkl".format(name))
    log_filename = os.path.join(directory, "{}.or_tools.log".format(name))

    data = create_data_model(depot, loc, demand, capacity, route_limit=route_limit, service_time=service_time,
                             tw_start=tw_start, tw_end=tw_end, draft=draft, grid_size=grid_size, problem=problem)

    # Create the routing index manager
    if problem in ["OVRP", "OVRPB", "OVRPL", "OVRPTW", "OVRPBL", "OVRPLTW", "OVRPBTW", "OVRPBLTW"]:
        manager = pywrapcp.RoutingIndexManager(data['num_locations'] + 1, data['num_vehicles'], [data['depot']] * data['num_vehicles'], [data['dummy_depot']] * data['num_vehicles'])
    else:
        manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Define weight of each edge
    distance_evaluator_index = routing.RegisterTransitCallback(partial(create_distance_evaluator(data), manager))
    routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)

    # Make sure to first minimize number of vehicles
    # routing.SetFixedCostOfAllVehicles(0)

    # Add Capacity constraint
    if problem not in ["TSPTW"]:
        demand_evaluator_index = routing.RegisterUnaryTransitCallback(partial(create_demand_evaluator(data), manager))
        if problem not in ["TSPDL"]:
            add_capacity_constraints(routing, data, demand_evaluator_index, problem=problem)
        else:
            add_draft_constraints(routing, manager, data, demand_evaluator_index, problem=problem)

    # Add Time Window (TW) constraint
    if problem in ["TSPTW", "VRPTW", "VRPBTW", "VRPLTW", "OVRPTW", "VRPBLTW", "OVRPLTW", "OVRPBTW", "OVRPBLTW"]:
        time_evaluator_index = routing.RegisterTransitCallback(partial(create_time_evaluator(data), manager))
        add_time_window_constraints(routing, manager, data, time_evaluator_index, grid_size=grid_size, time_horizon = tw_end[0])

    # Add Duration Limit (L) constraint
    if problem in ["VRPL", "VRPBL", "VRPLTW", "OVRPL", "VRPBLTW", "OVRPBL", "OVRPLTW", "OVRPBLTW"]:
        add_distance_constraints(routing, data, distance_evaluator_index)

    # Setting first solution heuristic (cheapest addition).
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.log_search = False  # print log
    search_parameters.time_limit.seconds = timelimit
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION)  # PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    # Solve the problem
    start = time.time()
    assignment = routing.SolveWithParameters(search_parameters)
    duration = time.time() - start

    if routing.status() not in [1, 2]:
        global CNT
        CNT += 1
        # print("sb", CNT)
        print(">> OR-Tools failed to solve instance {} - Solver status: {}".format(name, routing.status()))
        # exit(0)
        return None, None, duration

    if problem not in ["TSPTW"]:
        cost, route = print_solution(data, manager, routing, assignment, problem=problem, log_file=open(log_filename, 'w'))  # route does not include the first and last node (i.e., depot)
    else:
        cost, route = print_solution_tsptw(data, manager, routing, assignment, problem=problem, log_file=open(log_filename, 'w'))  # route does not include the first and last node (i.e., depot)
    print("\n".join(["{}".format(r) for r in ([data['depot']] + route + [data['depot']])]), file=open(tour_filename, 'w'))
    save_dataset((route, duration), output_filename, disable_print=True)

    return cost, route, duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OR-Tools baseline")
    parser.add_argument('--problem', type=str, default="TSPTW", choices=["TSPDL", "TSPTW", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW",
                                                                         "OVRPTW", "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                                                         "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument("--datasets", nargs='+', default=["../data/TSPTW/tsptw50_medium.pkl", ], help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_false', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, default=16, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, default=10000, help="Number of instances to process")
    parser.add_argument('-timelimit', type=int, default=20,
                        help="timelimit (seconds) for OR-Tools to solve an instance, we use 40s for n=100; 20s for n=50")
    parser.add_argument('-seed', type=int, default=1234, help="random seed")
    parser.add_argument('--offset', type=int, default=0, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='baseline_results', help="Name of results directory")
    parser.add_argument('--cal_gap', action='store_true', help="enable to calculate optimality gap")
    parser.add_argument('--opt_sol_path', default='../data/TSPTW/lkh_tsptw50_medium.pkl', help="directory to optimal solutions")

    opts = parser.parse_args()
    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:
        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"
        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "{}_or_tools".format(opts.problem))
            os.makedirs(results_dir, exist_ok=True)
            dir, filename = os.path.split(dataset_path)
            out_file = os.path.join(dir, "or_tools_{}s_{}".format(opts.timelimit, filename))
        else:
            out_file = opts.o
        assert opts.f or not os.path.isfile(out_file), "File already exists! Try running with -f option to overwrite."
        start_t = time.time()
        use_multiprocessing = True

        def run_func(args):
            directory, name, *args = args
            depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end, all_loc, draft = None, None, None, None, None, None, None, None, None, None
            if opts.problem in ["CVRP", "OVRP", "VRPB", "OVRPB"]:
                depot, loc, demand, capacity, *args = args
            elif opts.problem in ["VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"]:
                depot, loc, demand, capacity, service_time, tw_start, tw_end, *args = args
            elif opts.problem in ["VRPL", "VRPBL", "OVRPL", "OVRPBL"]:
                depot, loc, demand, capacity, route_limit, *args = args
            elif opts.problem in ["VRPLTW", "VRPBLTW", "OVRPLTW", "OVRPBLTW"]:
                depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end, *args = args
            elif opts.problem in ["TSPTW"]:
                all_loc, service_time, tw_start, tw_end, *args = args
                depot, loc = all_loc[:1], all_loc[1:]
            elif opts.problem in ["TSPDL"]:
                all_loc, demand, draft, *args = args
                depot, loc = all_loc[:1], all_loc[1:]
            else:
                raise NotImplementedError

            depot = depot[0] if len(depot) == 1 else depot  # if depot: [[x, y]] -> [x, y]
            grid_size = 1

            return solve_or_tools_log(
                directory, name,
                depot=depot, loc=loc, demand=demand, capacity=capacity, route_limit=route_limit, service_time=service_time, tw_start=tw_start, tw_end=tw_end, draft=draft,
                timelimit=opts.timelimit, grid_size=grid_size, seed=opts.seed, problem=opts.problem
            )

        target_dir = os.path.join(results_dir, "{}_or_tools_tl{}s".format(dataset_basename, opts.timelimit))
        print(">> Target dir: {}".format(target_dir))
        assert opts.f or not os.path.isdir(target_dir), "Target dir already exists! Try running with -f option to overwrite."
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        dataset = load_dataset(dataset_path)
        # Note: only processing n items is handled by run_all_in_pool
        results, parallelism = run_all_in_pool(
            run_func,
            target_dir, dataset, opts, use_multiprocessing=use_multiprocessing
        )

        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        filtered_cost = [x / 100 for x,_,_ in results if x is not None]
        print(">> Infeasible rate: {}%".format((1-len(filtered_cost)/opts.n)*100))
        print(">> Solving {} instances within {:.2f}s using OR-Tools".format(opts.n, time.time() - start_t))
        print(">> Average cost: {} +- {}".format(np.mean(filtered_cost), 2 * np.std(filtered_cost) / np.sqrt(len(filtered_cost))))
        print(">> Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print(">> Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print(">> Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        if opts.cal_gap:
            results = [(i[0], i[1]) for i in results]
            with open(opts.opt_sol_path, 'rb') as file:
                opt_sol = pickle.load(file)
            gaps = []
            for i in range(opts.n):
                if results[i][0] is not None:
                    gap = (results[i][0] - opt_sol[i][0]) / opt_sol[i][0]
                    gaps.append(gap)
            print(">> Optimality gap: {}%".format(np.mean(gaps)*100))

        save_dataset(results, out_file)  # [(obj, route), ...]
        os.system("rm -rf {}".format(target_dir))
