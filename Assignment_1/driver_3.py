import argparse
import collections
import copy
import math
import resource
import time


def manhattan_dist(idx, pos, n):
    """
    Calculate the manhattan distance of tile idx at position pos
    """

    row_dist = abs(pos // n - idx // n)
    col_dist = abs(pos % n - idx % n)
    return row_dist + col_dist


class StackedFrontier(object):
    def __init__(self, state):
        self.map = {state.str_config: state}
        self.order = [state.str_config]

    def push(self, state):
        self.map[state.str_config] = state
        self.order.append(state.str_config)

    def pop(self):
        state = self.map[self.order.pop()]
        del self.map[state.str_config]
        return state

    def empty(self):
        return len(self.map) == 0

    def clear(self):
        self.map = {}
        self.order = []

    def __contains__(self, state):
        return state.str_config in self.map


class QueueFrontier(object):
    def __init__(self, state):
        self.map = {state.str_config: state}
        self.order = collections.deque([state.str_config])

    def append(self, state):
        self.map[state.str_config] = state
        self.order.append(state.str_config)

    def pop(self):
        state = self.map[self.order.popleft()]
        del self.map[state.str_config]
        return state

    def empty(self):
        return len(self.map) == 0

    def clear(self):
        self.map = {}
        self.order = []

    def __contains__(self, state):
        return state.str_config in self.map


class PuzzleState(object):
    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        # if n * n != len(config) or n < 2:
        #     raise Exception("the length of config is not correct!")
        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.config = config
        self.str_config = str(config)[1:-1]
        self.children = []
        self.depth = parent.depth + 1 if parent else 0

        self.blank_pos = self.config.index(0)
        self.blank_row = self.blank_pos // self.n
        self.blank_col = self.blank_pos % self.n

    def manhattan_distance(self):
        dist = 0
        for pos, idx in enumerate(self.config):
            if idx != 0:  # Skip blank
                dist += manhattan_dist(idx, pos, self.n)
        return dist


    def display(self):
        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print(line)

    def move_left(self):
        if self.blank_col == 0:
            return None
        else:
            target = self.blank_pos - 1
            new_config = list(self.config)
            new_config[self.blank_pos], new_config[target] = new_config[target], new_config[self.blank_pos]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None
        else:
            target = self.blank_pos + 1
            new_config = list(self.config)
            new_config[self.blank_pos], new_config[target] = new_config[target], new_config[self.blank_pos]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None
        else:
            target = self.blank_pos - self.n
            new_config = list(self.config)
            new_config[self.blank_pos], new_config[target] = new_config[target], new_config[self.blank_pos]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None
        else:
            target = self.blank_pos + self.n
            new_config = list(self.config)
            new_config[self.blank_pos], new_config[target] = new_config[target], new_config[self.blank_pos]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self, reverse=False):
        # add child nodes in order of UDLR
        if len(self.children) == 0:
            tmp_children = [
                self.move_up(),
                self.move_down(),
                self.move_left(),
                self.move_right()
            ]
            self.children = [x for x in tmp_children if x is not None]
            if reverse:
                self.children = list(reversed(self.children))
        return self.children


def bfs_search(initial_state):
    """
    BFS search
    """
    goal = tuple(range(len(initial_state.config)))
    frontier = QueueFrontier(initial_state)
    explored = []
    success = None
    max_search_depth = 0

    while not frontier.empty():
        state = frontier.pop()
        if state.depth > max_search_depth:
            max_search_depth = state.depth
        explored.append(state.str_config)

        if test_goal(state, goal):
            frontier.clear()
            success = state
        else:
            for child in state.expand():
                if child.str_config not in explored \
                        and child not in frontier:
                    if child.depth > max_search_depth:
                        max_search_depth = child.depth
                    frontier.append(child)

    return dict(end_state=success, n_expanded=len(explored)-1, max_search_depth=max_search_depth)


def dfs_search(initial_state):
    """
    DFS search
    """
    goal = tuple(range(len(initial_state.config)))
    frontier = StackedFrontier(initial_state)
    explored = set([])
    success = None
    max_search_depth = 0

    while not frontier.empty():
        state = frontier.pop()
        explored.add(state.str_config)

        if test_goal(state, goal):
            frontier.clear()
            success = state
        else:
            for child in state.expand(reverse=True):
                if child.str_config not in explored \
                        and child not in frontier:
                    if child.depth > max_search_depth:
                        max_search_depth = child.depth
                    frontier.push(child)

    return dict(end_state=success, n_expanded=len(explored)-1, max_search_depth=max_search_depth)


def A_star_search(initial_state):
    """A * search"""

def test_goal(puzzle_state, goal):
    """test the state is the goal state or not"""
    return puzzle_state.config == goal


def rewind(state):
    tmp_state = copy.copy(state)
    path = []
    while tmp_state.parent is not None:
        path.insert(0, tmp_state.action)
        tmp_state = copy.copy(tmp_state.parent)
    return path


# Function that Writes to output.txt
def write_output(result, start_time):
    end_state = result['end_state']
    path = rewind(end_state)

    end_time = time.time()
    r = resource.getrusage(resource.RUSAGE_SELF)

    output_str = f"""path_to_goal: {path}
cost_of_path: {end_state.cost}
nodes_expanded: {result['n_expanded']}
search_depth: {end_state.depth}
max_search_depth: {result['max_search_depth']}
running_time: {end_time - start_time}
max_ram_usage: {r.ru_maxrss/1024/1024}
"""

    with open("output.txt", "w") as fh:
        fh.write(output_str)


def process(strategy, init_state, start_time):
    start_state = tuple(map(int, init_state))
    size = int(math.sqrt(len(start_state)))
    hard_state = PuzzleState(start_state, size)

    if strategy == "bfs":
        result = bfs_search(hard_state)
    elif strategy == "dfs":
        result = dfs_search(hard_state)
    elif strategy == "ast":
        result = A_star_search(hard_state)
    else:
        raise NotImplementedError(f"Unknown search strategy: {sm}")

    if result['end_state'] is not None:
        write_output(result, start_time)
    else:
        raise Exception("Unable to find solution")


if __name__ == '__main__':
    m_start_time = time.time()
    parser = argparse.ArgumentParser(description='Solve 8-puzzle.')
    parser.add_argument('sm', metavar='SM', help='Search model', choices=['bfs', 'dfs', 'ast'])
    parser.add_argument('start_state', metavar='INIT', help='Starting state')
    args = parser.parse_args()

    sm = args.sm
    m_start_state = args.start_state.split(",")
    process(sm, m_start_state, m_start_time)
