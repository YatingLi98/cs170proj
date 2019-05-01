import networkx as nx
import random
import math
import operator
from priority_queue import PriorityQueue


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
        dictionary[v] = client.scout(v, all_students)

    # initialize prob
    for v in non_home:
        score = sum([confidence[s] for s, res in dictionary[v].items() if res is True]) +\
                sum([-confidence[s] for s, res in dictionary[v].items() if res is False])
        prob.push(v, score)


    bot_at_home = 0
    while bot_at_home < client.bots:
        v = prob.pop()
        num = client.remote(v, client.home)
        if num != 0:
            bot_at_home += num
            update(True, confidence, prob, dictionary[v])
        else:
            update(False, confidence, prob, dictionary[v])

    client.end()


def update(res, confidence, prob, dictionary):
    for student, r in dictionary.items():
        if r == res:
            confidence[student] += 0.5
        else:
            confidence[student] -= 0.5
    for v in prob.key_set():
        score = sum([confidence[s] for s, res in dictionary.items() if res is True]) +\
                sum([-confidence[s] for s, res in dictionary.items() if res is False])
        prob.update(v, score)
