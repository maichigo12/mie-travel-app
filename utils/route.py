from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math


def calc_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


def solve_tsp(locations):
    size = len(locations)

    distance_matrix = [
        [
            calc_distance(
                locations[i]["lat"], locations[i]["lon"],
                locations[j]["lat"], locations[j]["lon"]
            )
            for j in range(size)
        ]
        for i in range(size)
    ]

    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(distance_matrix[f][t] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))

    return [locations[i]["name"] for i in route]

def make_google_map_url(route):
    return "https://www.google.com/maps/dir/" + "/".join(route)
