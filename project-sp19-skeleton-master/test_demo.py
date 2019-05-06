# demo1
G1 = {0: {1: {'weight': 100}, 3: {'weight': 100}},
      1: {0: {'weight': 100}, 2: {'weight': 410}, 3: {'weight': 105}},
      2: {1: {'weight': 410}, 3: {'weight': 50}, 5: {'weight': 100}},
      3: {0: {'weight': 100}, 1: {'weight': 105}, 2: {'weight': 50}, 4: {'weight': 100}, 5: {'weight': 400}},
      4: {3: {'weight': 100}, 5: {'weight': 210}},
      5: {2: {'weight': 100}, 3: {'weight': 400}, 4: {'weight': 210}}}
set1 = [0, 2, 5]
set2 = [1, 3, 4]
res = 0-3-2-5

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