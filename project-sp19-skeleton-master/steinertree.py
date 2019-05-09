import networkx as nx
import random
import heapq
from queue import *
from networkx.algorithms import approximation


class PQ:

    def __init__(self):
        self.heap = []
        self.count = 0
        self.contain = set()

    def size(self):
        return len(self.heap)

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
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
         self.push(item, -priority)
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
    allcost = 0
    for x in tree.nodes:
        visited[x] = False

    visited[start] = True ## 'start' is to be replaced by the home vertex in the actual graph

    for v in list(tree.adj[start]):
        qq.put((start, v))

    while (not qq.empty()):
        (u,v) = qq.get()
        visited[v] = True
        result.append((v,u))
        allcost += tree[v][u]['weight']

        helper = PQ()
        for node in list(tree.adj[v]):
            helper.push(node, tree[node][v]['weight'])
        while not helper.isEmpty():
            (node, _) = helper.pop()
            if not visited[node]:
                qq.put((v, node))

    return result, allcost

def dijkstra(client):
    gr = client.G
    start = client.home
    pq = PQ()
    dist = {}
    prev = {}
    for v in gr.nodes:
        prev[v] = -1
        dist[v] = float('inf')
    dist[start] = 0
    for v in gr.adj[start]:
        dist[v] = gr.edges[start,v]['weight']
        prev[v] = start
        pq.push((start, v), gr.edges[start, v]['weight'])
    while not pq.isEmpty():
        ((u,v),_) = pq.pop()
        for node in gr.adj[v]:
            if dist[node] > dist[v] + gr.edges[v,node]['weight']:
                dist[node] = dist[v] + gr.edges[v, node]['weight']
                prev[node] = v
                pq.push((v, node), dist[node])

    return prev,dist





def solve(client):
    client.end()
    client.start()

    non_home = list(range(1, client.home)) + list(range(client.home + 1, client.v + 1))
    all_students = list(range(1, client.students + 1))
    dictionary = dict()
    prob = PQ()
    remoted = set()
    havebots = []

    # initialize the confidence with 10
    confidence = {student: 10 for student in all_students}

    for v in non_home:
        dictionary[v] = client.scout(v, all_students)

    # initialize prob
    for v in non_home:
        score = sum([confidence[s] for s, res in dictionary[v].items() if res is True]) + \
                sum([-confidence[s] for s, res in dictionary[v].items() if res is False])

        prob.push(v, score)

    bot_at_home = 0
    path, dist = dijkstra(client)

    while bot_at_home < client.bots:
        #pop the nodes with highest probability of having bots, find the shortest dijstras remote of
        temp = []
        v, priority = prob.pop()
        temp.append(v)

        while True:
            if prob.isEmpty():
                break
            nex, nprio = prob.pop()
            if nprio != priority:
                prob.push(nex, nprio)
                break
            temp.append(nex)
        minv = float('inf')
        minnode = None
        for x in temp:
            comp = client.G[x][path[x]]['weight']
            if comp < minv:
                minv = comp
                minnode = x
        for x in temp:
            if x != minnode:
                prob.push(x , priority)
        v = minnode
        # pop the nodes with highest probability of having bots, find the shortest dijstras remote ABOVE

        num = client.remote(v, path[v])
        remoted.add(v)
        v = path[v]
        if num != 0:
            update(True, confidence, prob, dictionary, v, len(all_students), remoted)
            if v == client.home:
                bot_at_home += num
            else :
                havebots.append(v)
                print(havebots)
            if bot_at_home + len(havebots) >= client.bots:
                if len(havebots) == 0:
                   break
                havebots.append(client.home)
                order, b = bfs(approximation.steiner_tree(client.G, havebots), client.home)
                while order:
                    if bot_at_home >= client.bots:
                        break
                    (frm, to) = order.pop()
                    num = client.remote(frm, to)
                    if to == client.home:
                        bot_at_home += num
        else:
            update(False, confidence, prob, dictionary, v, len(all_students), remoted)






    client.end()

def update(res, confidence, prob, dictionary, vertex, n, seen):
    for v in dictionary.keys():
        if v != vertex:
            continue
        for sid, r in dictionary[v].items():
            if r == res:
                confidence[sid] -= 0
            else:
                if n > 30:
                    confidence[sid] += 1
                else:
                    confidence[sid] += 2.5
    for v in prob.key_set():
        if v in seen:
            continue
        score = sum([confidence[s] for s, res in dictionary[v].items() if res is True]) +\
                sum([-confidence[s] for s, res in dictionary[v].items() if res is False])
        prob.update(v, score)


def powerset(s):
    pset = []
    x = len(s)
    for i in range(1 << x):
        pset.append([s[j] for j in range(x) if (i & (1 << j))])
    return pset


if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from([0,1,2,3,4,5])
    g.add_edges_from([(0, 1, {'weight': 100}), (2, 1, {'weight': 410}), (2, 5, {'weight':100 })
                         , (2, 3, {'weight': 50}), (3, 5, {'weight': 400}), (1, 3, {'weight': 105})
                         , (3, 4, {'weight': 100}), (4, 5, {'weight': 210}), (0, 3, {'weight': 100})])

    a,b = bfs(approximation.steiner_tree(g, [0,2,5]), 0)
    while a:
        print(a.pop())
    print(b)
