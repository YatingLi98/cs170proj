class tree:
    def __init__(self, val):
        self.val = val
        self.degree = 0
        self.children = []
        self.parent = None

    def add_children(self, child):
        self.children.append(child)
    
    def set_degree(self, d):
        self.degree = d

    def get_degree(self):
        return self.degree
    
    def get_children(self):
        return self.children

    def set_parent(self, p):
        self.parent = p

    def get_parent(self):
        return self.parent


def change_into_tree(paths):
    root = tree(-1)
    dictionary = {-1: root}
    flag = True
    for path in paths:
        for i in range(1, len(path)):
            curr = tree(path[i])
            dictionary[path[i]] = curr
            if flag:
                root.add_children(curr)
                flag = False
            else:
                dictionary[path[i-1]].add_children(curr)
    complete_tree(root, 0)
    return root.get_children()[0]

def complete_tree(root, degree):
    for children in root.get_children():
        children.set_degree(degree)
        children.set_parent(root)
        complete_tree(children, degree + 1)



        
            

