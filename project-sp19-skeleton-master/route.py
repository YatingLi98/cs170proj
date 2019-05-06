from MST import *
from priority_queue import *

def route(G, S1, S2):
    """
    This modified dijsktra's takes in a graph G and a set S.
    It runs dijkstra's from the node with index 0.

    :param G: A dictionary indexed by nodes. The value of G[v] is also a dict indexed
    by adjacent nodes. For any edge (u,v), G[u][v]['weight'] is the edge length.
    :param S1: Source nodes set.
    :param S2: Destination nodes set.
    :return: A path stored in list from home to any nodes in S.
    """

    for home in S1:
        dist, prev = dijkstra(G, home)
        dist_s1 = {} # dist_s1 is used to store the shortest path from a specific node in S1 to S2.
        dist_s2 = {}
        for u in dist:
            if u in S2:
                entry = [home, u]
                dist_s2[entry] = dist[u]
            dist_s1[min(dist_s2, key=dist_s2.get)] = dist[u]
    final_entry = min(dist_s1, key=dist_s1.get)
    start = final_entry[0]
    end = final_entry[1]
    path = []
    n = end
    while prev.get(n) != start:
        path.insert(0, prev[n])
        n = prev[n]
    path.insert(start)
    return path

def dijkstra(G, home):
    """
    :param G:
    :param home: starting vertex.
    :return: two dictionaries that contains distances and predecessors.
    """

    inf = float('infinity')
    dist = {vertex: inf for vertex in G}
    dist[0] = 0
    prev = {}
    entry_lookup = {}
    PQ = PriorityQueue()

    for vertex, distance in dist.items():
        entry = [distance, vertex]
        entry_lookup[vertex] = entry
        PQ.push(vertex, distance)

    while len(PQ) > 0:
        current_distance, current_vertex = PQ.pop()

        for neighbor, neighbor_distance in G[current_vertex]:
            distance = dist[current_distance] + neighbor_distance['weight']
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                PQ.update(neighbor, distance)
                prev[neighbor] = current_vertex
    return (dist, prev)


    """
    # demo2
G2 = {0: {1: {'weight': 50}, 2: {'weight': 20},  3: {'weight': 150}, 4: {'weight': 200}, 6: {'weight': 30}},
      1: {0: {'weight': 50}, 2: {'weight': 50}, 3: {'weight': 50}, 4: {'weight': 100}},
      2: {0: {'weight': 20}, 1: {'weight': 50}, 5: {'weight': 20}},
      3: {0: {'weight': 150}, 1: {'weight': 50}},
      4: {0: {'weight': 200}, 1: {'weight': 100}},
      5: {2: {'weight': 20}, 6: {'weight': 30}},
      6: {0: {'weight': 30}, 5: {'weight': 30}}}
set1_2 = [0, 1, 2, 3, 4, 5]
set2_2 = [6]
res_2 = 0-1-3, 0-1-4, 0-2-5    
    """