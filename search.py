import networkx as nx
import numpy as np
import bisect
import random
import math
import sys
class NavigationProblem:
    def __init__(self, initial, goal, connections, locations=None, directed=False):
        self.initial = initial
        self.goal = goal
        self.locations = locations
        self.graph = nx.DiGraph() if directed else nx.Graph()
        for cityA, cityB, distance in connections:
            self.graph.add_edge(cityA, cityB, cost=distance)            
    def successors(self, state):
        ## Exactly as defined in Lecture slides, 
        return [("go to %s" % city, connection['cost'], city) for city, connection in self.graph[state].items()]
    
    
    def goal_test(self, state):
        return state == self.goal


class Node:
    def __init__(self, state=None, parent=None, action=None, path_cost=0,level=0,depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = depth        ###One additional paramter is added to node class to maintain the depth information.


    def getPath(self):
        """getting the path of parents up to the root"""
        currentNode = self
        path = [self]
        while currentNode.parent:  ## stops when parent is None, ie root
            path.append(currentNode.parent)
            currentNode = currentNode.parent
        path.reverse()             #from root to this node
        return path
    def expand(self, problem):
        successors = problem.successors(self.state)
        return [Node(newState, self, action, self.path_cost+cost) for (action, cost, newState) in successors]
    def __gt__(self, other): ## needed for tie breaks in priority queues
        return True
    def __repr__(self):
        return (self.state, self.action, self.path_cost)
    
class FIFO:
    def __init__(self):
        self.list = []
    def push(self, item):
        self.list.insert(0, item)  
    def pop(self):
        return self.list.pop()


class LIFO:             ### fill out yourself! 
    def __init__(self):
        self.list = []
    def push(self, item):
        self.list.append(item)
    def pop(self):
        return self.list.pop()


class PriorityQueue:
    def __init__(self, f):
        self.list = []
        self.f = f
    def push(self, item):
        priority = self.f(item)
        bisect.insort(self.list, (priority, random.random(), item))
    def pop(self):
        return self.list.pop(0)[-1]
        

def graph_search(problem, frontier,depth=sys.maxsize):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    If two paths reach a state, only use the best one. [Fig. 3.18]"""
    closed = set() ## can store hashable objects, thats why we need to define a hash code for states
    p = Node(problem.initial)
    p.depth=0                                 ### Set Root node depth = 0.
    frontier.push(p)
    maxFrontierLength=1                       ### Set max froniter lenght=1 because we have pushed root node only.
    explorationHistory = []
    while frontier:
        if not frontier.list:
            return None,None,None
        node = frontier.pop()

        explorationHistory.append(node)
        if problem.goal_test(node.state): 
            return node, explorationHistory,maxFrontierLength
        if node.state not in closed:
            closed.add(node.state)
            successors = node.expand(problem)
            if len(frontier.list)> maxFrontierLength: ### Right after adding nodes to froniter. Compare maxFrontier to length of list in frontier.
                maxFrontierLength = len(frontier.list) ### Update maxFrontiter based on condition.
            for snode in successors:
                snode.depth=snode.parent.depth+1 ### Here we have to define depth for each node which is parentDepth + 1.
                if snode.depth<=depth:          ### This line is important. Each node has its depth if depth of node > then depth limit then do not add that node to frontiter 
                    frontier.push(snode)        ### For all other algorithms like BFS,DFS,Astar and UCS it will not effect because depth limit will be infinity for them.
            
                
def tree_search(problem, frontier):
    frontier.push(Node(problem.initial))
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        successors = node.expand(problem)
        for snode in successors:
            frontier.push(snode)
    
def breadth_first_graph_search(problem):
    return graph_search(problem, FIFO())

### Depth first search will call graph search algorithm with specific depth limit.
def depth_first_graph_search(problem,depth= sys.maxsize):
    return graph_search(problem, LIFO(),depth)


def astar_graph_search(problem, f):
    return graph_search(problem, PriorityQueue(f))
def uniform_graph_search(problem, ucs):
    return graph_search(problem, PriorityQueue(ucs))


### depth limited search will call depth first search algorithm
def depth_limited_search(problem,depth):
    return depth_first_graph_search(problem,depth=depth)


### sys.maxsize is the maximum integer.
### Iterative_deep_graph_search algorithm will call depth limited search algorithm for different depths.
### Loop is the actually the depth.
### If no solution is found at specific depth then it will return null,null,null and try next depth.

def iterative_deep_graph_search(problem,depth= sys.maxsize):
    loop=0 
    depth = True   
    while depth:
        x,y,z = depth_limited_search(problem,loop)
        if x==None or y==None or z==None:
            print("\nNo Solution at depth ",loop)
        else:
            return x,y,z
        loop+=1
    

#%%
# priority functions for Priority Queues used in UCS and A*, resp., if you are unfamiliar with lambda calc.

def ucs(node):
    return node.path_cost

def f(node):
    return node.path_cost + h[node.state]

if __name__ == "__main__":
    ## Getting familiar with the Toy NavigationProblem
    #toyConnections = [('A', 'B', 5), ('A', 'C', 3), ('A', 'D', 1),('B', 'E', 3),('C', 'F', 3),('C', 'G', 3),('G', 'I', 1),('D', 'H', 1),('H', 'J', 1) ,('H', 'K', 1)]
    #h = {'S':7, 'A':1, 'B':2, 'C':6, 'G':0} ## simple heuristics according to slides
    
    #toy = NavigationProblem('A', 'J', toyConnections, directed=True)
    # Uniform cost:
    #sol, history,y = iterative_deep_graph_search(toy,depth = 3 )
    #print ("UCS Solution:", [(node.state, node.action) for node in sol.getPath()])
    #print ("exploration history:", [node.state for node in history])
    #exit(0)

#%%
    # Best first
    #sol, history = graph_search(toy, PriorityQueue(lambda node: h[node.state]))
    #print ("Greedy Search Solution:", [(node.state, node.action) for node in sol.getPath()])
    #print ("exploration history:", [node.state for node in history])
#%%
    # A*
    #sol, history = graph_search(toy, PriorityQueue(f))
    #print ("A* Solution:", [(node.state, node.action) for node in sol.getPath()])
    #print ("exploration history:", [node.state for node in history])

#%%
    ## Romania
    connections = [('A', 'S', 140), ('A', 'Z', 75), ('A', 'T', 118), ('C', 'P', 138), ('C', 'R', 146), ('C', 'D', 120), ('B', 'P', 101),
                   ('B', 'U', 85), ('B', 'G', 90), ('B', 'F', 211), ('E', 'H', 86), ('D', 'M', 75), ('F', 'S', 99), ('I', 'V', 92),
                   ('I', 'N', 87), ('H', 'U', 98), ('L', 'M', 70), ('L', 'T', 111), ('O', 'S', 151), ('O', 'Z', 71), ('P', 'R', 97), ('R', 'S', 80), ('U', 'V', 142)]
    
    locations =     {'A': (91, 492), 'C': (253, 288), 'B': (400, 327), 'E': (562, 293), 'D': (165, 299), 'G': (375, 270), 'F': (305, 449),
                     'I': (473, 506), 'H': (534, 350), 'M': (168, 339), 'L': (165, 379), 'O': (131, 571), 'N': (406, 537), 'P': (320, 368),
                     'S': (207, 457), 'R': (233, 410), 'U': (456, 350), 'T': (94, 410), 'V': (509, 444), 'Z': (108, 531)}

    
    romania = NavigationProblem('A', 'B', connections) ## for A*, you will need to also provide the locations
    #B is our goal state and A is the initial state
    
    



    import math
    allStates = list(locations.keys())
    historyLength={}
    solutionLength={}
    maxFrontierLength={}
    h={}

    
    ###Heuristic calculation for Romania problem
    ##Calulating distance between each node and the goal node and store that in "h" dictionary.
    for state in allStates:
        historyLength[state]=[]
        solutionLength[state]=[]
        maxFrontierLength[state]=[]
        h[state] = int(math.sqrt((locations[state][0]-locations['B'][0])**2 + (locations[state][1]-locations['B'][1])**2))
    
    for state in allStates:
        #Breadth first search
        solution, history,maximal = breadth_first_graph_search(NavigationProblem(state, 'B', connections))
        historyLength[state].append(len(history))
        solutionLength[state].append(len(solution.getPath()))
        maxFrontierLength[state].append(maximal)
        print ([(node.state, node.action) for node in solution.getPath()])
        print("\n")
        #Depth first search
        solution, history,maximal = depth_first_graph_search(NavigationProblem(state, 'B', connections))
        historyLength[state].append(len(history))
        solutionLength[state].append(len(solution.getPath()))
        maxFrontierLength[state].append(maximal)
        print ([(node.state, node.action) for node in solution.getPath()])
        print("\n")
        #Uniform Cost search
        solution, history,maximal = graph_search(NavigationProblem(state, 'B', connections), PriorityQueue(ucs))
        historyLength[state].append(len(history))
        solutionLength[state].append(len(solution.getPath()))
        maxFrontierLength[state].append(maximal)
        print ([(node.state, node.action) for node in solution.getPath()])
        print("\n")
        #A-star search
        solution, history,maximal = graph_search(NavigationProblem(state, 'B', connections) ,PriorityQueue(f))
        historyLength[state].append(len(history))
        solutionLength[state].append(len(solution.getPath()))
        maxFrontierLength[state].append(maximal)
        print ([(node.state, node.action) for node in solution.getPath()])
        print("\n")
        #Best First search
        
        solution, history,maximal = graph_search(NavigationProblem(state, 'B', connections) ,PriorityQueue(lambda node: h[node.state]))
        historyLength[state].append(len(history))
        solutionLength[state].append(len(solution.getPath()))
        maxFrontierLength[state].append(maximal)
        print ([(node.state, node.action) for node in solution.getPath()])
        print("\n")
        #Iterative Deepning search
        solution, history,maximal =iterative_deep_graph_search(NavigationProblem(state, 'B', connections))
        
        historyLength[state].append(len(history))
        solutionLength[state].append(len(solution.getPath()))
        maxFrontierLength[state].append(maximal)
        #print("Lenght of History", len(history))
        print ([(node.state, node.action) for node in solution.getPath()])
        print("\n")
    import pandas as pd
    import matplotlib.pyplot as plt
    labels = ["Breadth First", "Depth First", "Uniform Cost","A-star","Best First","Iterative Deepning"]
    '''
    sumB = 0
    sumD = 0
    sumU = 0
    sumA = 0
    sumBE = 0
    sumI = 0
    x=0
    
    for h in historyLength:
        sumB = sumB +historyLength[h][0]
        sumD = sumD +historyLength[h][1]
        sumU = sumU +historyLength[h][2]
        sumA = sumA +historyLength[h][3]
        sumBE = sumBE +historyLength[h][4]
        sumI = sumI +historyLength[h][5]
    print(sumB,sumD,sumU,sumA,sumBE,sumI)
    
    plt.bar(labels,[sumB,sumD,sumU,sumA,sumBE,sumI])
    plt.title('Cumulative Number of Max Frontiter for Algorithm')
    plt.show()
   
    exit(0)
    '''
    
    df = pd.DataFrame(solutionLength).T
    df.plot(kind="bar")
    plt.legend(labels)
    plt.title("Solution Length")
    plt.show()
   
    df = pd.DataFrame(historyLength).T
    df.plot(kind="bar")
    plt.legend(labels)
    plt.title("History")
    plt.show()
    
    df = pd.DataFrame(maxFrontierLength).T
    df.plot(kind="bar")
    plt.legend(labels)
    plt.title("Frontier length")
    plt.show()
    
    


