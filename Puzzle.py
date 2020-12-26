import networkx as nx
import numpy as np
import bisect
import random
import math
from random import randint
class PuzzleState:
    def __init__(self, matrix=None, goal=False, init=False, size=3,mutate=0,missPlaced=0,manhattan=0,mpm=0,e=0,l=0):
        ## ** add code that generates puzzle states of varying difficulty
        self.size = size
        self.zeroPosition = 0
        self.missPlaced = missPlaced
        self.manhattan = manhattan
        self.mpm = missPlaced+manhattan
        self.e = e
        self.l = l+manhattan

        if not matrix is None and mutate==0: ## 0 represents empty spot
            self.matrix = matrix
            self.zeroPosition = np.where(self.matrix.flatten() == 0)[0][0]
        else: 
            ## defining init or goal state
            ## this is too hard for the benchmark, change this!
            ## *** Your code here ***
            permutation = np.array(range(size*size))
            if goal:
                self.matrix = permutation.reshape((size, size))
                self.zeroPosition = 1
                self.__repr__()
            if init:
                ### Here I will generate the initial state based on the number of permutaion.    
                self.matrix = matrix

                self.zeroPosition = np.where(matrix.flatten() == 0)[0][0]

                [r1,c1] = np.where(self.matrix == 0)
                m = matrix.copy()      
                dummyactions = [1,1,1,1]
                
                while mutate>0:
                    x = randint(0, 3)              ### Generate a random number that will decide in which direction we can move.                          
                    [r1,c1] = np.where(m == 0)

                    if r1==0:
                        dummyactions[0]=0
                    else:
                        dummyactions[0]=1
                    if c1==0:
                        dummyactions[2]=0
                    else: 
                        dummyactions[2]=1
                    if r1==size-1:
                        dummyactions[1]=0
                    else:
                        dummyactions[1]=1
                    if c1==size-1:
                        dummyactions[3]=0
                    else:
                        dummyactions[3]=1

                    if x==0 and dummyactions[0]:
                        m[r1,c1],m[r1-1,c1] =m[r1-1,c1],m[r1,c1] 
                        mutate-=1
           
                    if x==1 and dummyactions[1]:
                        m[r1,c1],m[r1+1,c1] =m[r1+1,c1],m[r1,c1] 
                        mutate-=1
 
                    if x==2 and dummyactions[2]:
                        m[r1,c1],m[r1,c1-1] =m[r1,c1-1],m[r1,c1] 
                        mutate-=1
                    if x==3 and dummyactions[3]:
                        m[r1,c1],m[r1,c1+1] =m[r1,c1+1],m[r1,c1]
                        mutate-=1
                self.matrix = m
                self.__repr__()        
                

   
    def successors(self):

        pass
    '''
    def solveable(self):
        if (self.size % 2 == 0):
            if(self.zeroPosition % 2 == 0):
                if self.calculateInversion() % 2 != 0:
                    return 1
            else:
                if self.calculateInversion() % 2 == 0:
                    return  1


        else:
            if (self.calculateInversion() % 2 == 0):
                return 1
        return 0

    def calculateInversion(self):
        inv_count=0
        for i in range(0,self.size):
            for j in range(i+1,self.size): 
                if (self.matrix[j][i] > 0 and self.matrix[j][i] > self.matrix[i][j]): 
                    inv_count=inv_count+1
        return inv_count
    '''

    def __repr__(self):
        print(self.matrix)
    def __hash__(self):
        ## return a hash code, if you have a matrix, you can uncomment this:
        return hash(tuple(self.matrix.flatten()))
    def __eq__(self, other):
        ## this function defines equality between objects, if you have a matrix, you can uncomment this:
        return np.alltrue(other.matrix == self.matrix)

class PuzzleProblem:
    def __init__(self, size=3,mutate=0,initial=None,goal=None): #size 3 means 3x3 field
        self.size = size
        if not goal is None:
            self.goal = PuzzleState(np.array(goal),goal=True, size=size)  ## goal state is unshuffled 
        else:
            self.goal = PuzzleState(goal=True, size=size)
        if not initial is None:
            self.initial = PuzzleState(np.array(initial),init=True, size=size)
        else:
            self.initial = PuzzleState(matrix = self.goal.matrix ,init=True, size=size,mutate=mutate)

            
        
          ## init state is shuffled 
        #  ## init state is shuffled 
        #  ## goal state is unshuffled 
    

    def missPlacedtiles(self,matrix):
        m = matrix.flatten()
        m2 = self.goal.matrix.flatten()
        missing = 0
        for loop in range(0,self.size*self.size):
            if m[loop] != m2[loop]:
                missing=missing+1
        return missing - 1
    def linearConflict(self,matrix):
        conflicts = 0
        m = matrix.copy()
        m = m.flatten()
        n = self.size
        in_col = [0] * (n*n)
        in_row = [0] * (n*n) 
        for y in range(0,self.size):
            for x in range(0,self.size):
                i = y * n + x

                bx = m[i] % n
                by = m[i] / n

                in_col[i] = (bx == x)
                in_row[i] = (by == y) 
        for y in range(0,self.size):
            for x in range(0,self.size):
                i = y * n + x
                if (m[i] == 0):
                    continue
                if in_col[i]:
                    for r in range(y,n):
                        j = r * n + x
                        if (m[j] == 0): 
                            continue
                        if in_col[j] and m[j] < m[i]:
                            conflicts+=1
                if in_row[i]:
                    for c in range(x,n):
                        j = y * n + c
                        if (m[j] == 0):
                            continue
                        if (in_row[j] and m[j] < m[i]):
                            conflicts+=1
 
        return 2 * conflicts
        
        
        


    def manhattantiles(self,matrix):
        m = matrix
        m2 = self.goal.matrix
        distance = 0
        for loop in range(1,self.size*self.size):
            [r1,c1] = np.where(m == loop)
            [r2,c2] = np.where(m2 == loop)
            distance = distance +abs(r2-r1)+abs(c2-c1)
        return distance
    def elucidean(self,matrix):
        m = matrix
        m2 = self.goal.matrix
        distance = 0
        for loop in range(1,self.size*self.size):
            [r1,c1] = np.where(m == loop)
            [r2,c2] = np.where(m2 == loop)
            distance = distance +math.sqrt(abs(r2-r1)+abs(c2-c1))

        return distance
        
    
    def successors(self, state):
        actions = []
        
        dummyactions = [1,1,1,1]
        #It means it can move in all directions.
        #[1,1,1,1]====Up,down,left,right
        m = state.matrix.copy()
        [r1,c1] = np.where(m == 0)   
        if r1==0:
            dummyactions[0]=0
        if c1==0:
            dummyactions[2]=0
        if r1==self.size-1:
            dummyactions[1]=0
        if c1==self.size-1:
            dummyactions[3]=0

        m = state.matrix.copy()


        if dummyactions[0]:
            m[r1,c1],m[r1-1,c1] =m[r1-1,c1],m[r1,c1] 
            actions.append(("Go Up", 1, PuzzleState(m, self.size,missPlaced=self.missPlacedtiles(m),manhattan=self.manhattantiles(m),e=self.elucidean(m),l=self.linearConflict(m))))

        m = state.matrix.copy()

        if dummyactions[1]:
            m[r1,c1],m[r1+1,c1] =m[r1+1,c1],m[r1,c1] 
            actions.append(("Go Down", 1, PuzzleState(m, self.size,missPlaced=self.missPlacedtiles(m),manhattan=self.manhattantiles(m),e=self.elucidean(m),l=self.linearConflict(m))))

        m = state.matrix.copy()
        if dummyactions[2]:
            m[r1,c1],m[r1,c1-1] =m[r1,c1-1],m[r1,c1] 
            actions.append(("Go Left", 1, PuzzleState(m, self.size,missPlaced=self.missPlacedtiles(m),manhattan=self.manhattantiles(m),e=self.elucidean(m),l=self.linearConflict(m))))

        m= state.matrix.copy()
        if dummyactions[3]:
            m[r1,c1],m[r1,c1+1] =m[r1,c1+1],m[r1,c1]
            actions.append(("Go Right", 1, PuzzleState(m, self.size,missPlaced=self.missPlacedtiles(m),manhattan=self.manhattantiles(m),e=self.elucidean(m),l=self.linearConflict(m))))


        return actions

    def goal_test(self, state):
        for i in range(self.size):
            for j in range(self.size):
                if (self.goal.matrix[i][j] != state.matrix[i][j]):
                    return 0
        return 1
        
class Node:
    def __init__(self, state=None, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost 
        self.action = action       
    def getPath(self):

        currentNode = self
        path = [self]
        while currentNode.parent:
            path.append(currentNode.parent)
            currentNode = currentNode.parent
        path.reverse() 
        return path
    def expand(self, problem):
        successors = problem.successors(self.state)
        return [Node(newState, self, action, self.path_cost+cost) for (action, cost, newState) in successors]
    def __gt__(self, other):
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
class LIFO:  ## fill out yourself! 
    def __init__(self):
        pass
    def push(self, item):
        pass
    def pop(self):
        pass
class PriorityQueue:
    def __init__(self, f):
        self.list = []
        self.f = f
    def push(self, item):
        priority = self.f(item)
        bisect.insort(self.list, (priority, random.random(), item))
    def pop(self):
        return self.list.pop(0)[-1]
        
def graph_search(problem, frontier):

    closed = set() ## can store hashable objects, thats why we need to define a hash code for states
    frontier.push(Node(problem.initial))
    explorationHistory = []
    while frontier:

        if not frontier.list:
            print("CLOSED",len(closed))
            print(len(explorationHistory))
            print("No solution")
            exit(0)
        node = frontier.pop()
        explorationHistory.append(node)
        if problem.goal_test(node.state): 
            print("CLOSED",len(closed))
            return node, explorationHistory
        if node.state not in closed:
            closed.add(node.state)
            successors = node.expand(problem)
            for snode in successors:
                frontier.push(snode)
                #print(snode.state.__repr__())
                





def ucs(node):
    return node.path_cost


def mpm(node):
    return node.path_cost +node.state.missPlaced + node.state.manhattan

def missPlaced(node):
    return node.path_cost +node.state.missPlaced

def manhattan(node):
    return node.path_cost + node.state.manhattan

def elucidean(node):
    return node.path_cost + node.state.e
def linearConflict(node):
    return node.path_cost+node.state.manhattan+node.state.l

if __name__ == "__main__":
    import time
    
    ### Manhattan,Hamming distance, manhattan plus hamming distance, Elucidean, Numberofinversion (Linear conflicts) + Manhattan distance 
    
    
#%% ### This cell contain code to test program by passing START and GOAL state. It will work for N by N problem.
    #t= time.time()
    
    #problemSize = 4
    #start = [(1,5,2,3),(4,6,10,7),(8,9,11,15),(0,12,13,14)]
    #goal = [(0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15)]
    
    #Puzzle = PuzzleProblem(problemSize,initial=start,goal=goal)
    #finalNode, history = graph_search(Puzzle, PriorityQueue(linearConflict))
    #numberOfNodes = 0

    #for node in finalNode.getPath():
    #    print (node.state.matrix,node.action,node.state.manhattan)
    #print(len(history))
    #print(time.time()-t)
    #exit(0)

#%%
    
    times={}
    historys={}
    steps={}
    labels = ["UCS","Manhattan","Missing tiles","Manhattan + misplaced tiles","Euclidean","Linear Conflict + Manhattan"]

    difficultyLevels=20
    problemSize = 4  # I have implement N by N problem. You have to specify problemSize.
    #m is the mutation rate (Difficulty Level) to be produced in goal state to form the initial state.
    
    for loop in range(0,difficultyLevels):
        times[loop]=[]
        historys[loop]=[]
        steps[loop]=[]
    
    for m in range(0,difficultyLevels):
        Puzzle = PuzzleProblem(problemSize,m) 
        
        print("UCS")
        t= time.time()
        finalNode, history = graph_search(Puzzle, PriorityQueue(ucs))
        s=0
        for node in finalNode.getPath():
            s+=1
            #print (node.state.matrix,node.action,node.state.manhattan)
        times[m].append(time.time()-t)
        historys[m].append(len(history))
        steps[m].append(s)

        print("Manhattan")
        t= time.time()        
        finalNode, history = graph_search(Puzzle, PriorityQueue(manhattan))
        s=0
        for node in finalNode.getPath():
            s+=1
            #print (node.state.matrix,node.action,node.state.manhattan)
        times[m].append(time.time()-t)
        historys[m].append(len(history))
        steps[m].append(s)


        print("Hamming Distance")
        t= time.time()
        finalNode, history = graph_search(Puzzle, PriorityQueue(missPlaced))
        s=0
        for node in finalNode.getPath():
            s+=1
            #print (node.state.matrix,node.action,node.state.manhattan)
        times[m].append(time.time()-t)
        historys[m].append(len(history))
        steps[m].append(s)
    
        print("Manhattan plus misplaced tiles")
        t= time.time()
        finalNode, history = graph_search(Puzzle, PriorityQueue(mpm))
        s=0
        for node in finalNode.getPath():
            s+=1
            #print (node.state.matrix,node.action,node.state.manhattan)
        times[m].append(time.time()-t)
        historys[m].append(len(history))
        steps[m].append(s)

        print("Elucidean")
        t= time.time()
        finalNode, history = graph_search(Puzzle, PriorityQueue(elucidean))
        s=0
        for node in finalNode.getPath():
            s+=1
            #print (node.state.matrix,node.action,node.state.manhattan)
        times[m].append(time.time()-t)
        historys[m].append(len(history))
        steps[m].append(s)

        print("Linear Conflict plus Manhattan Distance")
        t= time.time()
        finalNode, history = graph_search(Puzzle, PriorityQueue(linearConflict))
        s=0
        for node in finalNode.getPath():
            s+=1
            #print (node.state.matrix,node.action,node.state.manhattan)
        times[m].append(time.time()-t)
        historys[m].append(len(history))
        steps[m].append(s)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(times).T
    df.plot(kind="bar")
    plt.legend(labels)
    plt.yscale("log")
    plt.show()
   
    df = pd.DataFrame(historys).T
    df.plot(kind="bar")
    plt.legend(labels)
    plt.yscale("log")
    plt.show()
    
    df = pd.DataFrame(steps).T
    df.plot(kind="bar")
    plt.legend(labels)
    plt.yscale("log")
    plt.show()








    exit(0)






    
    #for x in range(0,20):
        #Puzzle = PuzzleProblem(3)
        #Puzzle.goal.__repr__()
        #print(Puzzle.initial.zeroPosition)
        #finalNode, history = graph_search(Puzzle, PriorityQueue(manhattan))
        #for node in finalNode.getPath():
        #    print (node.state.matrix,node.action,node.state.manhattan)
        #    print ("Explored States:", len(history))
        #print(time.time()-t)
    

