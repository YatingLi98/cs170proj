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
    path = {}
    for x in tree.nodes:
        visited[x] = False

    visited[start] = True ## 'start' is to be replaced by the home vertex in the actual graph

    for v in list(tree.adj[start]):
        qq.put((start, v))

    while (not qq.empty()):
        (u,v) = qq.get()
        visited[v] = True
        result.append((v,u))
        path[v] = u

        helper = PQ()
        for node in list(tree.adj[v]):
            helper.push(node, tree[node][v]['weight'])
        while not helper.isEmpty():
            (node, _) = helper.pop()
            if not visited[node]:
                qq.put((v, node))

    return result, path

def dicetrust(client):
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
    havebots = {}


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

    while bot_at_home < client.bots:
        toberemoted = set()
        newadd = set()
        numfound = 0

        for v, n in havebots.items():
            if n > 0:
                numfound += n
                toberemoted.add(v)
            else:
                prob.update(v, float('-inf'))
        while bot_at_home + numfound < client.bots:
            a, b = prob.pop()
            numfound += 1
            toberemoted.add(a)
            newadd.add(a)

        toberemoted.add(client.home)



        order, path = bfs(approximation.steiner_tree(client.G, list(toberemoted)), client.home)
        if bot_at_home + sum(havebots.values()) == client.bots:
            remotelist = order

        else:
            remotelist = []
            for (me, you) in order:
                if me in newadd:
                    remotelist.append((me, you))

      
        while remotelist:
            if bot_at_home >= client.bots:
                break
            (frm, to) = remotelist.pop()
            num = client.remote(frm, to)
            remoted.add(frm)
            if num != havebots.get(frm, num):
                update(True, confidence, prob, dictionary, v, len(all_students), remoted)
            else:
                update(False, confidence, prob, dictionary, v, len(all_students), remoted)
            havebots[frm] = 0
            prob.update(frm, float('-inf'))
            if to == client.home:
                bot_at_home += num
            else:
                havebots[to] = havebots.get(to, 0) + num
            if bot_at_home >= client.bots:
                break


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
                    confidence[sid] += 3.5
                else:
                    confidence[sid] += 3.5
    for v in prob.key_set():
        if v in seen:
            continue
        score = sum([confidence[s] for s, res in dictionary[v].items() if res is True]) +\
                sum([-confidence[s] for s, res in dictionary[v].items() if res is False])
        prob.update(v, score)
