from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math

def solve_tsp(locations, start_index=0, end_index=None):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    import math

    def distance(a, b):
        return math.sqrt((a["lat"] - b["lat"])**2 + (a["lon"] - b["lon"])**2)

    size = len(locations)
    manager = pywrapcp.RoutingIndexManager(size, 1, start_index, end_index if end_index is not None else start_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance(locations[from_node], locations[to_node]) * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(locations[node]["name"])
        index = solution.Value(routing.NextVar(index))

    node = manager.IndexToNode(index)
    route.append(locations[node]["name"])

    return route

