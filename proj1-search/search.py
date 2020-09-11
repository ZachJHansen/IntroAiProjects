# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions

n = Directions.NORTH
s = Directions.SOUTH
e = Directions.EAST
w = Directions.WEST

class Q:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

    def Print(self):
        print("\nPrinting Fringe")
        for l in self.list:
            print(l)
        print("\n") 
    
    # Return true if state occurs as the last state in an enqueued sequence
    # Otherwise return false
    def find(self, state):
        for sequence in self.list:
            cap = sequence[-1][0]
            if state == cap:
                return True
        return False

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# Establishes the priority of a sequence (lowest cost has best priority)
def priority(sequence):
    end_node = sequence[-1]
    cost = end_node[2]
    return cost

# Establishes the priority of a sequence 
def manhattan_priority(sequence):
    end_node = sequence[-1]
    state, cost = end_node[0], end_node[2]
    f = cost + util.manhattanDistance(state, problem.goal)

def retrace(sequence, successor):
    for tupleware in sequence:
        if tupleware[0] == successor[0]:                        # Has the successor state been visited in this sequence?
#            print("Retrace!")
            return True
    return False

# Parameters: Successor tuple, current node tuple, strategy
# Action: If bfs or dfs returns the successor tuple unchanged, otherwise adds current cost to successor cost
# Returns: tuple (state, action, cost) where cost is cost of path up to state
def update_cost(successor, current, strategy):
    if (strategy == "dfs" or strategy == "bfs"):
        return successor
    else:
        temp = [successor[0], successor[1], current[2] + successor[2]]
        return (tuple(temp))

# Returns: tuple (state, action, cost, visited) where cost is cost of path up to state
def update_corner_cost(successor, current, strategy):
    if (strategy == "dfs" or strategy == "bfs"):
        return successor
    else:
        temp = [successor[0], successor[1], current[2] + successor[2], successor[-1]]
        return (tuple(temp))

def graph_search_wrapper(problem, strategy): 
    problem_type = type(problem).__name__
    if (problem_type == "CornersProblem"):
        path = graph_search(problem, strategy)
    else:
        path = graph_search(problem, strategy)
    
    #print("PATH", path)
    if (path is None):
        print("No path to goal. Exiting...")
        exit(0)
    else:
        action_path = []
        for tup in path:
            if tup[1] is not None:
                action_path.append(tup[1])
        #print("Path has length:", len(action_path))
        return action_path

def graph_search(problem, strategy):
    problem_type = type(problem).__name__
    if (strategy == "dfs"):
        fringe = util.Stack()
    elif (strategy == "bfs"):
        fringe = util.Queue()
    elif (strategy == "ucs" ):
        fringe = util.PriorityQueueWithFunction(priority)
    elif (strategy == "astar"):
        fringe = util.PriorityQueueWithFunction(astar_priority)
    else:
        exit(1)
    closed_set = set()                                          # Set of nodes that have been expanded
    fringe.push([(problem.getStartState(), None, 1)])    # Fringe of sequences (Lists) of (State, Action, Cost) SAC tuples yet to be explored
    while True:
        #fringe.Print()
        #print(closed_set)
        if fringe.isEmpty(): 
            return None                                         # No more candidates to search (empty frontier/fringe)
        sequence = fringe.pop()                                 # Choose a path to explore from the fringe
        node = sequence[-1]                                     # The node SAC to expand is the last in the sequence
        state = node[0]
        #return sequence
        if (problem_type == "CornersProblem"):
            golar = problem.isGoalState(sequence)
            #if state in [(1,1), (1,6), (6,6), (6,1)]:
            #    closed_set = set()
        else:
            golar = problem.isGoalState(state)
        if golar:
            return sequence
        if state not in closed_set:                             # State has not been previously visited
            closed_set.add(state)                               # Add state to set of explored states
            successors = problem.getSuccessors(state)           # All possible next states
            for child in successors:                
                if child not in closed_set and not retrace(sequence, child): #and not fringe.find(child)):   # State has not been previously visited
                    temp = sequence + [update_cost(child, node, strategy)]
                    fringe.push(temp)                           # Update the cost of s, add s to next, re-add to data structure    

def wrapper(path):
    if (path is None):
        print("No path to goal. Exiting...")
        exit(0)
    else:
        action_path = []
        for tup in path:
            if tup[1] is not None:
                action_path.append(tup[1])
        #print("Path has length:", len(action_path))
        return action_path

def depthFirstSearch(problem):
    return graph_search_wrapper(problem, "dfs")

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    strategy = "bfs"
    if type(problem).__name__ == 'CornersProblem':
        root = [problem.getStartState(), None, 0, 0]            # c0c1c2c3 ranges from 0 to 15 (0000 = 0, 0100 = 2, etc.)
        corner_count = problem.CornerCheck(root)
        root[-1] = corner_count
        fringe = Q()
        fringe.push([tuple(root)])                              # Queue of sequences of (state, action, cost, visitation count) tuples (nodes)
        closed_set = set()                                      # Set of (state, visitation count) pairs
        while True:
            if fringe.isEmpty():
                return None
            sequence = fringe.pop()
            node = sequence[-1]                                 # Get the last node in the path/sequence
            svp = (node[0], node[-1])                           # (state, visitations) pair svp
            closed_set.add(svp)
            successors = problem.getSuccessors(svp)             # All possible next nodes (state, action, cost, visitation) pairs
            for child in successors:   
                child_svp = (child[0], child[-1])               # Only need (state, visitations)            
                if child_svp not in closed_set: #and not fringe.find(child):    # State has not been previously visited
                    temp = sequence + [update_corner_cost(child, node, strategy)]
                    if problem.isGoalState(sequence):
                        return wrapper(sequence)
                    fringe.push(temp)  

    else:
        return graph_search_wrapper(problem, "bfs")

def uniformCostSearch(problem):
    return graph_search_wrapper(problem, "ucs")

def corner_astar(problem, heuristic):
    root = [problem.getStartState(), None, 0, 0]            # c0c1c2c3 ranges from 0 to 15 (0000 = 0, 0100 = 2, etc.)
    visit_count = problem.CornerCheck(root)
    root[-1] = visit_count
    fringe = util.PriorityQueue()
    fringe.push([tuple(root)], 0)                           # Queue of sequences of (state, action, cost, visitation count) tuples (nodes)
    closed_set = set()                                      # Set of (state, visitation count) pairs
    while True:
        if fringe.isEmpty():
            return None
        sequence = fringe.pop()
        if problem.isGoalState(sequence):
            return wrapper(sequence)
        node = sequence[-1]                                 # Get the last node in the path/sequence
        svp = (node[0], node[-1])                           # (state, visitations) pair svp
        closed_set.add(svp)
        successors = problem.getSuccessors(svp)             # All possible next nodes (state, action, cost, visitation) pairs
        for child in successors:   
            child_svp = (child[0], child[-1])               # Only need (state, visitations)            
            if child_svp not in closed_set: #and not fringe.find(child):    # State has not been previously visited
                temp = sequence + [update_corner_cost(child, node, "astar")]
                f_n = temp[-1][2] + heuristic(child_svp, problem)
                fringe.push(temp, f_n) 

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    if type(problem).__name__ == 'CornersProblem':
        return corner_astar(problem, heuristic)
    else:
        #Define Fringe
        Fringe = util.PriorityQueue()
        #Place Holder for State Inspected already
        done_state = []
        #Path that's returned in the end
        path = []
        #Complete Path to the Current State from S
        com_path = util.PriorityQueue()
        #Push the Start State to the Priority Queue
        Fringe.push(problem.getStartState(),0)
        #Define the current state
        current_state = Fringe.pop()
        
        #Check if we have reached the goal state, if not iterate
        while not problem.isGoalState(current_state):
            #If the current state has not been expanded before, add to done list
            if current_state not in done_state:
                done_state.append(current_state)
                #Get the successors of the current node
                successor_nodes = problem.getSuccessors(current_state)
                
                #Check the childern
                for child,direction,cost in successor_nodes:
                    if child not in done_state:
                        Fringe.push(child,problem.getCostOfActions(path+ [direction]) + heuristic(child,problem))
                        com_path.push(path+ [direction],problem.getCostOfActions(path+ [direction]) + heuristic(child,problem))
                    
            #Backup to the previous node        
            current_state = Fringe.pop()
            #Uh oh! Wrong direction, so remove it
            path = com_path.pop()
        return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch