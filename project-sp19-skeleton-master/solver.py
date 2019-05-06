# Put your solution here.
"""
import networkx as nx
import random
from queue import *

git
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

    all_students = list(range(1, client.students + 1))
    non_home = list(range(1, client.home)) + list(range(client.home + 1, client.v + 1))
    client.scout(random.choice(non_home), all_students)

    for _ in range(100):
        u, v = random.choice(list(client.G.edges()))
        client.remote(u, v)



    all_students = list(range(1, client.students + 1))
    non_home = list(range(1, client.home)) + list(range(client.home + 1, client.v + 1))

    order = findmst(client.G, client.home)

    while order:
        (frm, to) = order.pop()
        client.remote(frm, to)

    client.end()


if __name__ == '__main__':
    findmst()



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


       gr.add_nodes_from([0,1,2,3,4,5,6,7,8])
    gr.add_edges_from([(0, 1, {'weight': 4}), (1, 2, {'weight': 8}), (2, 3, {'weight': 7})
                          , (3, 4, {'weight': 9})
                          , (4, 5, {'weight': 10}), (5, 6, {'weight': 2}),
                       (6, 7, {'weight': 1}), (7, 0, {'weight': 8}), (1, 7, {'weight': 11}), (7, 8, {'weight': 7}), (8, 6, {'weight': 6})
                       , (2, 8, {'weight': 2}), (2, 5, {'weight': 4}), (5, 3, {'weight': 14})])
"""
import networkx as nx
import random
import math
import operator
import heapq


class realnode:
    score = 0
    vertex = None

    def __init__(self, sc, node):
        self.score = sc
        self.vertex = node


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0
        self.contain = set()

    def push(self, item, priority):
        entry = (-priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1
        self.contain.add(item)

    def pop(self):
        (val, _, item) = heapq.heappop(self.heap)
        self.contain.remove(item)
        return item, -val

    def key_set(self):
        return self.contain

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        priority = -priority
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


def dijkstra(client):  # Dijsktra
    gr = client.G

    start = client.home
    pq = PriorityQueue()
    dist = {}
    prev = {}
    for v in gr.nodes:
        prev[v] = -1
        dist[v] = float('inf')
    dist[start] = 0
    for v in gr.adj[start]:
        dist[v] = gr.edges[start, v]['weight']
        prev[v] = start
        pq.push((start, v), gr.edges[start, v]['weight'])
    while not pq.isEmpty():
        item, priority = pq.pop()
        v = item[1]
        for node in gr.adj[v]:
            if dist[node] > dist[v] + gr.edges[v, node]['weight']:
                dist[node] = dist[v] + gr.edges[v, node]['weight']
                prev[node] = v
                pq.push((v, node), dist[node])

    return prev, dist


def solve(client):
    client.end()
    client.start()

    # variance
    all_edges = []
    for v1 in range(1, client.v + 1):
        all_edges.extend([client.G[v1][v2]['weight'] for v2 in client.G[v1] if v2 > v1])
    mean = sum(all_edges) / client.v
    sd = math.sqrt(sum([pow(e - mean, 2) for e in all_edges]) / (client.v - 1))
    rsd = sd / mean * 100

    # scout_all
    non_home = list(range(1, client.home)) + list(range(client.home + 1, client.v + 1))
    all_students = list(range(1, client.students + 1))
    dictionary = dict()
    prob = PriorityQueue()

    # initialize the confidence with 10
    confidence = {student: 10 for student in all_students}

    for v in non_home:
        dictionary[v] = client.scout(v, list(all_students))

    # initialize prob
    for v in non_home:
        score = sum([confidence[s] for s, res in dictionary[v].items() if res is True]) + \
                sum([-confidence[s] for s, res in dictionary[v].items() if res is False])

        prob.push(v, score)

    bot_at_home = 0
    path, dist = dijkstra(client)
    while bot_at_home < client.bots:
        same_priority = {}
        v, priority = prob.pop()
        same_priority[v] = client.G.edges[v][path[v]]['weight']
        while True:
            nex, nprio = prob.pop()
            if nprio != priority:
                prob.push(nex, nprio)
                break
            same_priority[nex] = client.G.edges[nex][path[nex]]['weight']
        minnode = min(same_priority.items(), key=lambda x: x[1])
        for node in same_priority:
            if node != minnode:
                prob.push(node, priority)
        v = minnode
        num = client.remote(v, path[v])
        v = path[v]
        if num != 0:
            update(True, confidence, prob, dictionary, v)
            while v != client.home:
                numbots = client.remote(v, path[v])
                if numbots != num:
                    update(True, confidence, prob, dictionary, v)
                    num = numbots
                else:
                    update(False, confidence, prob, dictionary, v)

                v = path[v]
            bot_at_home += num
        else:
            update(False, confidence, prob, dictionary, v)

    client.end()


def update(res, confidence, prob, dictionary, vertex):
    for v in dictionary.keys():
        if v != vertex:
            continue
        for sid, r in dictionary[v].items():
            if r == res:
                continue
            else:
                confidence[sid] += 3.5
    for v in prob.key_set():
        score = sum([confidence[s] for s, res in dictionary[v].items() if res is True]) + \
                sum([-confidence[s] for s, res in dictionary[v].items() if res is False])

        prob.update(v, score)
