import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    # def recursiveDFS(node):
    #     if(problem.isGoalState(node.state)):
    #         return node.state
    #     successors = problem.getSuccessors(node.state)
    #     if (successors==[]):
    #         return False
    #     for successor in successors:
    #         path_cost = node.path_cost+successor[2]
    #         depth = node.depth+1
    #         node1 = recursiveDFS(Node(successor[0], successor[1], path_cost, node, depth))
    #         if (node1 is not False):
    #             return node1
    #     return False
    # value = recursiveDFS(Node(problem.getStartState(), None, 0, None, 0))
    # if value is not False:
    #     return value
    # else:
    #     return False

    #####
    stack=util.Stack()
    stack.push(Node(problem.getStartState(), None, 0, None, 0))
    while(not stack.isEmpty()):
        node = stack.pop()
        if(problem.isGoalState(node.state)):
            return node.state
        successors = problem.getSuccessors(node.state)
        for successor in successors:
            path_cost = node.path_cost+successor[2]
            depth = node.depth+1
            stack.push(Node(successor[0], successor[1], path_cost, node, depth))
    
    return False


######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    return util.points2distance(((problem.G.node[state]['x'], 0, 0),(problem.G.node[state]['y'], 0, 0)), ((problem.G.node[problem.end_node]['x'], 0, 0),(problem.G.node[problem.end_node]['y'], 0, 0)))

def AStar_search(problem, heuristic=nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """

    node=Node(problem.start_node, None, 0, None, 0)
    frontier = util.PriorityQueue()
    frontier.push(node, 0)
    explored = set([])
    while True:
        if frontier.isEmpty():
            return False
        node = frontier.pop()
        if (problem.isGoalState(node.state)):
            path = []
            n=node
            while True:
                path = [n.state]+path
                if n.state == problem.start_node:
                    break
                n = n.parent_node
            return path
        if node.state in explored:
            continue
        explored.add(node.state)
        successors = problem.getSuccessors(node.state)
        for successor in successors:
            if(successor[0] not in explored):
                cnode=Node(successor[0], successor[1], node.path_cost+successor[2], node, node.depth+1)
                frontier.push(cnode , cnode.path_cost+ heuristic(cnode.state, problem))