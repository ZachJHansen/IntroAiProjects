# multiAgents.py
# --------------
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


from util import manhattanDistance as DrManhattan
from game import Directions
from math import tanh
import random, util
import statistics
#from searchAgents import mazeDistance

from game import Agent

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


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
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


# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
#import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    return 0 # Default to trivial solution

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    return 0

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(bfs(prob))


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    # Returns the cumulative distance between pacman and remaining food
    # Randomness introduced to remove thrashing
    def total_food_dist(self, pacman, food_grid):
        total = 0
        master = []
        c = 1
        for col in food_grid:
            indices = [(indx, c) for indx in range(len(col)-1, 1, -1) if col[indx] == True]
            master = master + indices
            c += 1
        for pellet in master:
            total += DrManhattan(pacman, pellet) #+ random.randrange(1,3)
        if total == 0:
            return 1
        else:
            return 1/total

    # Returns distance to nearest food (is this working right?)
    def nearest_food(self, pacman, food_grid, food_count):
        master = []
        c = 1
        for col in food_grid:
            indices = [(indx, c) for indx in range(len(col)-1, 1, -1) if col[indx] == True]
            master = master + indices
            c += 1
        nearest = None
        nearest = (float('inf'), float('inf'))
        dist = float('inf')
        for food in master:
            candidate = DrManhattan(pacman, food)
            if candidate < dist:
                nearest = food
                dist = candidate
        #print(master)
        #print("Pacman", pacman, "nearest", nearest, "dist", dist)
        del master
        if dist == 0:
            return 1
        else:
            return 1/dist

    def ghost_eval(self, pacman, ghost, scared_time=0):
        tan = tanh(DrManhattan(pacman, ghost.getPosition()) - 1.75)
        if scared_time - 3 > DrManhattan(pacman, ghost.getPosition()):  # Bust the ghost!
            return 1/tan
        else:
            return tan              # Fly, you fools!


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    # An evaluation function is a weighted linear function of features used to estimate utility of a state
    # Possible Pacman features: ghost proximity, food proximity, food count
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print(successorGameState)
        #print(newPos)
        #print(newFood)
        #for g in newGhostStates:
        #    print(str(g))

        fc = successorGameState.getNumFood()
        if fc != 0:
            fc = 1/fc
            #nf = self.nearest_food(newPos, newFood, fc)
        else:
            fc = 1
            #fc, nf = 0, 0
        if currentGameState.getPacmanPosition() == newPos:
            p = 1
        else:
            p = 0
        #print("nf", nf, "fc", fc)
        g1 = newGhostStates[0]
        weights = {"ghost_prox1" : 0.8, 
                    #"ghost_prox2": 0.3, 
                    "food_total": 0.9, 
                    "food_prox": 0.2,
                    "food_count": 0.9, 
                    "paralysis": -1, 
                    "score": 0.1,
                    "win": 10,
                    "lose": -10}
        features = {
            "ghost_prox1" : self.ghost_eval(newPos, g1, newScaredTimes[0]), 
            #"ghost_prox2" : tanh(DrManhattan(newPos, g2.getPosition()) - 1.75),
            "food_total": self.total_food_dist(newPos, newFood),
            #"food_prox": self.nearest_food(newPos, newFood, fc), 
            "food_count": fc,
            "paralysis": p,
            "score": successorGameState.getScore(),
            "win": successorGameState.isWin(),
            "lose": successorGameState.isLose()
        }
        
        if (len(newGhostStates)) > 1:
            g2 = newGhostStates[1]
            features["ghost_prox2"] = self.ghost_eval(newPos, g2, newScaredTimes[1])
            weights["ghost_prox2"] = 0.8
        #weights = {"food_prox": 1}
        #features = {"food_prox": nf}
        ws = lambda key : weights[key] * features[key]
        expected_value = sum([ws(key) for key in features])
        #print(nf, " | ", expected_value)
        return expected_value

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        max_depth = (self.depth * num_agents) - 1           # Ply * agents per ply (0 indexed)
        current_depth = 0
        best_action, best_score = None, float('-inf')
        for action in gameState.getLegalActions(self.index):
            score = self.value(gameState.generateSuccessor(self.index, action), num_agents, current_depth+1, max_depth)
            if score > best_score:
                best_action, best_score = action, score
        return best_action

    def max_value(self, state, num_agents, current_depth, max_depth):
        val = float('-inf')
        agentIndex = current_depth % num_agents
        actions = state.getLegalActions(agentIndex)
        for a in actions:
            successor = state.generateSuccessor(agentIndex, a)
            val = max(val, self.value(successor, num_agents, current_depth+1, max_depth))
        return val

    def min_value(self, state, num_agents, current_depth, max_depth):
        val = float('inf')
        agentIndex = current_depth % num_agents
        actions = state.getLegalActions(agentIndex)
        for a in actions:
            successor = state.generateSuccessor(agentIndex, a)
            val = min(val, self.value(successor, num_agents, current_depth+1, max_depth))
        return val

    # Returns true if the state is terminal (win or loss) or a leaf node at depth D
    def terminal(self, state, current_depth, max_depth):
        if current_depth > max_depth or (state.isWin() or state.isLose()):
            return True
        else:
            return False

    # Input: Game State, number of agents, current tree depth (0 indexed), max tree depth (NOT the ply, which is supplied via command line)
    # Output: Recursively calculated minimax value for the provided state
    def value(self, state, num_agents, current_depth, max_depth):
        if self.terminal(state, current_depth, max_depth):
            return self.evaluationFunction(state)
        if current_depth % num_agents == 0:                                 # Is the next agent min or max?
            return self.max_value(state, num_agents, current_depth, max_depth)  # Max agent returns max of its childrens' values
        else:
            return self.min_value(state, num_agents, current_depth, max_depth)  # Min agent returns min of its childrens' values

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        max_depth = (self.depth * num_agents) - 1           # Ply * agents per ply (0 indexed)
        current_depth = 0
        best_action, best_score = None, float('-inf')
        for action in gameState.getLegalActions(self.index):
            score = self.value(gameState.generateSuccessor(self.index, action), num_agents, current_depth+1, max_depth)
            if score > best_score:
                best_action, best_score = action, score
        return best_action

    def max_value(self, state, num_agents, current_depth, max_depth):
        val = float('-inf')
        agentIndex = current_depth % num_agents
        actions = state.getLegalActions(agentIndex)
        for a in actions:
            successor = state.generateSuccessor(agentIndex, a)
            val = max(val, self.value(successor, num_agents, current_depth+1, max_depth))
        return val

    def exp_value(self, state, num_agents, current_depth, max_depth):
        val = float('inf')
        tempSum = 0;
        tempXiPi = 0;
        agentIndex = current_depth % num_agents
        actions = state.getLegalActions(agentIndex)
        for a in actions:
            successor = state.generateSuccessor(agentIndex, a)
            #calcualte the multipication between the random event(Xi)and it's probability (Pi)
            tempXiPi = (self.value(successor, num_agents, current_depth+1, max_depth))/len(state.getLegalActions(agentIndex))
            #Find the total of XiPi since E(X)= sum(XiPi)
            tempSum += tempXiPi;
        val = tempSum;
        return val

    # Returns true if the state is terminal (win or loss) or a leaf node at depth D
    def terminal(self, state, current_depth, max_depth):
        if current_depth > max_depth or (state.isWin() or state.isLose()):
            return True
        else:
            return False

    # Input: Game State, number of agents, current tree depth (0 indexed), max tree depth (NOT the ply, which is supplied via command line)
    # Output: Recursively calculated minimax value for the provided state
    def value(self, state, num_agents, current_depth, max_depth):
        if self.terminal(state, current_depth, max_depth):
            return self.evaluationFunction(state)
        if current_depth % num_agents == 0:                                 # Is the next agent min or max?
            return self.max_value(state, num_agents, current_depth, max_depth)  # Max agent returns max of its childrens' values
        else:
            return self.exp_value(state, num_agents, current_depth, max_depth)  # Min agent returns min of its childrens' values

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Inspiration came from Quiz 5, Question 3 for the evaluation function. 
    Since the problem statement allows to utilze project 1 files, we will use the mazeDistance from searchAgents.py.
    An evaluation function is a weighted linear function of features used to estimate utility of a state
    Possible Pacman features: ghost proximity, food proximity, food count
    """
    "*** YOUR CODE HERE ***"
    
    #Get Pacman's new position
    pacManPosition = currentGameState.getPacmanPosition()
    #Define the Distance Measurement
    distanceMeasure = mazeDistance
     #Get the latest score
    currentScore = currentGameState.getScore()
    #A list of food coordinates
    foodPositionList = currentGameState.getFood().asList()
    #Create a list of food distance measurements.
    closeFoodList = []
    #If food is still present in the game
    if foodPositionList:
        for food in foodPositionList:
            #get the maze distance for each food item
            closeFoodList.append(distanceMeasure(pacManPosition, food,currentGameState))
            #locate the closest food item so far
            minFoodDistance = min(closeFoodList)
    else:
        #assign a random float value to the distance.
        minFoodDistance = random.uniform(0, 1)
    #Expand the utility by awarding more points for the min. distance between pacman and food.
    currentEvaluation = currentScore + (1 / minFoodDistance)
    return currentEvaluation

# Abbreviation
better = betterEvaluationFunction
