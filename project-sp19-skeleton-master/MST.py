import networkx as nx
import random
from queue import *


class realPQ(PriorityQueue):

    def __init__(self):
        PriorityQueue.__init__(self)
        self.size = 0

    def pop(self, *args, **kwargs):
        x, y, obj = PriorityQueue.get(self, *args, **kwargs)
        return obj

    def put(self, obj, val):
        PriorityQueue.put(self, (val, self.size, obj))
        self.size += 1


def bfs(tree, start):
    result = []
    qq = Queue()
    visited = {}
    for x in tree.nodes:
        visited[x] = False

    visited[start] = True ## 'start' is to be replaced by the home vertex in the actual graph

    for v in list(tree.adj[start]):
        qq.put((start, v))

    while (not qq.empty()):
        (u,v) = qq.get()
        visited[v] = True
        result.append((v,u))

        for node in tree.adj[v]:
            if not visited[node]:
                qq.put((v, node))

    return result


def findmst(g, home):

    pq = realPQ()
    cost = {}
    prev = {}
    visited = {}
    start = home

    for x in g.nodes:
        cost[x] = float('inf')
        prev[x] = -1
        visited[x] = False
    visited[start] = True
    cost[start] = 0
    prev[start] = -1

    for v in g.adj[start]:

        pq.put((start,v), g[start][v]['weight'])

    while not pq.empty():
        (u,v) = pq.pop()
        if (not visited[v]) and cost[v] > g[u][v]['weight']:
            cost[v] = g[u][v]['weight']
            prev[v] = u
            visited[v] = True
            for node in g.adj[v]:
                pq.put((v, node), g[v][node]['weight'])

    tree = nx.Graph()
    for x in prev:
        if prev[x] != -1:
            tree.add_nodes_from([x, prev[x]])
            tree.add_edge(x, prev[x])

    return bfs(tree, home)  #this returns a list with (u,v) pairs, which gives the reverse of the correct order of remote


def solve(client):
    client.end()
    client.start()
    """
    all_students = list(range(1, client.students + 1))
    non_home = list(range(1, client.home)) + list(range(client.home + 1, client.v + 1))
    client.scout(random.choice(non_home), all_students)

    for _ in range(100):
        u, v = random.choice(list(client.G.edges()))
        client.remote(u, v)

    
    """

    order = findmst(client.G, client.home)

    while order:
        (frm, to) = order.pop()
        client.remote(frm, to)
    client.end()


if __name__ == '__main__':
    findmst()


"""
gr = nx.Graph()
    gr.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f'])
    gr.add_edges_from([('a', 'b', {'wei': 2}), ('a', 'c', {'wei': 1}), ('e', 'c', {'wei': 3})
                         , ('e', 'f', {'wei': 1}), ('d', 'f', {'wei': 4}), ('d', 'b', {'wei': 1})
                         , ('d', 'c', {'wei': 2}), ('b', 'c', {'wei': 2}),
                      ('e', 'd', {'wei': 3})])
    g = nx.Graph()
    g.add_nodes_from(['a', 'b','c','d','e','f'])
    g.add_edges_from([('a', 'b', {'wei': 5}), ('a', 'c', {'wei': 6}), ('e', 'c', {'wei': 5})
                      , ('e', 'f', {'wei': 4}), ('d', 'f', {'wei': 4}), ('d', 'b', {'wei': 2})
                      , ('d', 'c', {'wei': 2}), ('f', 'c', {'wei': 3}), ('b', 'c', {'wei': 1}),
                      ('a', 'd', {'wei': 4})])
    s = 0
    for x in prev:
        print(str(x) + "  :  " + str(prev[x]))
        if (prev[x] == -1):
            continue
        s += g.edges[x, prev[x]]['wei']
    print(s)
"""
