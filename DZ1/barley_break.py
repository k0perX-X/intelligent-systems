import numpy as np
import numba
from typing import Dict, List, Callable, Any
from enum import Enum
from random import choice, randrange


class Direction(Enum):  # 0 ← 1 ↑ 2 → 3 ↓
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    def __repr__(self):
        return str(self).replace("Direction.", "")

    def __str__(self):
        return super().__str__().replace("Direction.", "")


Tree = Dict[Direction, 'Branch']

path_branches = []

class Branch:
    def __init__(self, matrix: np.matrix, directions: List[Direction], tree: Tree = None):
        if tree is None:
            tree = dict()
        self.matrix = matrix
        self.directions = directions
        self.tree = tree

    def __str__(self):
        return str(self.matrix)
        # return str(self.matrix) + "\n" + str(self.tree)

    def __repr__(self):
        return str(self)


def selection_branch_func(l: List[Branch]) -> Branch:
    return l.pop(randrange(0, len(l)))
    # return l.pop(0)
    # return l.pop(len(l) - 1)


class BranchQueue:
    def __init__(self, func: Callable[[np.matrix], float], selection_func: Callable[[List[Branch]], Branch]):
        self.matrices: Dict[float, List[Branch]] = {}
        self.func = func
        self.selection_func = selection_func

    def put(self, branch: Branch):
        f = self.func(branch.matrix)
        if f in self.matrices:
            self.matrices[f].append(branch)
        else:
            self.matrices[f] = [branch]

    def get(self) -> tuple[Branch, float]:

        key = min(self.matrices.keys())
        if len(self.matrices[key]) == 1:
            el = self.matrices[key][0]
            del self.matrices[key]
            return el, key
        else:
            return self.selection_func(self.matrices[key]), key


@numba.njit()
def func(matrix: np.matrix) -> float:
    # height, weight = matrix.shape
    f: int = matrix.size - 1
    reshaped = matrix.reshape(-1)
    for i in range(1, reshaped.size):
        if reshaped[i - 1] == i:
            f -= 1
    print(f)
    return f


@numba.njit()
def condition_for_exiting(result_of_func: float, step: int) -> tuple[bool, Any]:
    if step > 1000000:
        return True, 1000000
    else:
        return True if result_of_func == 0 else False, step


@numba.njit(cache=True)
def _step_left(matrix: np.matrix, index: np.ndarray, new_matrix=True) -> np.matrix:
    if new_matrix:
        new_matrix = matrix.copy()
    else:
        new_matrix = matrix
    new_matrix[index[0], index[1]] = new_matrix[index[0], index[1] - 1]
    new_matrix[index[0], index[1] - 1] = 0
    return new_matrix


@numba.njit(cache=True)
def _step_up(matrix: np.matrix, index: np.ndarray, new_matrix=True) -> np.matrix:
    if new_matrix:
        new_matrix = matrix.copy()
    else:
        new_matrix = matrix
    new_matrix[index[0], index[1]] = new_matrix[index[0] - 1, index[1]]
    new_matrix[index[0] - 1, index[1]] = 0
    return new_matrix


@numba.njit(cache=True)
def _step_right(matrix: np.matrix, index: np.ndarray, new_matrix=True) -> np.matrix:
    if new_matrix:
        new_matrix = matrix.copy()
    else:
        new_matrix = matrix
    new_matrix[index[0], index[1]] = new_matrix[index[0], index[1] + 1]
    new_matrix[index[0], index[1] + 1] = 0
    return new_matrix


@numba.njit(cache=True)
def _step_down(matrix: np.matrix, index: np.ndarray, new_matrix=True) -> np.matrix:
    if new_matrix:
        new_matrix = matrix.copy()
    else:
        new_matrix = matrix
    new_matrix[index[0], index[1]] = new_matrix[index[0] + 1, index[1]]
    new_matrix[index[0] + 1, index[1]] = 0
    return new_matrix


def get_tree(matrix: np.matrix) -> tuple[Branch, int, Branch]:
    height, weight = matrix.shape
    root: Branch = Branch(matrix, [None])
    step_number = 0
    positions: Dict[bytes, Branch] = {matrix.tobytes(): root}
    branches = BranchQueue(func, selection_branch_func)
    branches.put(root)
    # pool = Pool(4)
    while True:
        step_number += 1
        branch, f = branches.get()
        path_branches.append(branch)
        # print(branch)
        cond = condition_for_exiting(f, step_number)
        if cond[0]:
            step_number = cond[1]
            break
        matrix, prev_directions, tree = branch.matrix, branch.directions, branch.tree
        index = np.argwhere(matrix == 0).reshape(-1)
        if prev_directions[-1] != Direction.RIGHT and index[1] != 0:  # Left
            new_matrix = _step_left(matrix, index)
            new_matrix_tobytes = new_matrix.tobytes()
            if new_matrix_tobytes not in positions:
                new_branch = Branch(new_matrix, prev_directions + [Direction.LEFT])
                positions[new_matrix_tobytes] = new_branch
                branches.put(new_branch)
            tree[Direction.LEFT] = positions[new_matrix_tobytes]
        if prev_directions[-1] != Direction.DOWN and index[0] != 0:  # Up
            new_matrix = _step_up(matrix, index)
            new_matrix_tobytes = new_matrix.tobytes()
            if new_matrix_tobytes not in positions:
                new_branch = Branch(new_matrix, prev_directions + [Direction.UP])
                positions[new_matrix_tobytes] = new_branch
                branches.put(new_branch)
            tree[Direction.UP] = positions[new_matrix_tobytes]
        if prev_directions[-1] != Direction.LEFT and index[1] != weight - 1:  # Right
            new_matrix = _step_right(matrix, index)
            new_matrix_tobytes = new_matrix.tobytes()
            if new_matrix_tobytes not in positions:
                new_branch = Branch(new_matrix, prev_directions + [Direction.RIGHT])
                positions[new_matrix_tobytes] = new_branch
                branches.put(new_branch)
            tree[Direction.RIGHT] = positions[new_matrix_tobytes]
        if prev_directions[-1] != Direction.UP and index[0] != height - 1:  # Down
            new_matrix = _step_down(matrix, index)
            new_matrix_tobytes = new_matrix.tobytes()
            if new_matrix_tobytes not in positions:
                new_branch = Branch(new_matrix, prev_directions + [Direction.DOWN])
                positions[new_matrix_tobytes] = new_branch
                branches.put(new_branch)
            tree[Direction.DOWN] = positions[new_matrix_tobytes]
    return root, step_number, branch


def shuffle(matrix: np.matrix):
    height, weight = matrix.shape
    for i in range(1000):
        index = np.argwhere(matrix == 0).reshape(-1)
        while True:
            direction = choice(list(Direction))
            if direction == Direction.RIGHT and index[1] != 0:  # Left
                _step_left(matrix, index, new_matrix=False)
                break
            elif direction == Direction.DOWN and index[0] != 0:  # Up
                _step_up(matrix, index, new_matrix=False)
                break
            elif direction == Direction.LEFT and index[1] != weight - 1:  # Right
                _step_right(matrix, index, new_matrix=False)
                break
            elif direction == Direction.UP and index[0] != height - 1:  # Down
                _step_down(matrix, index, new_matrix=False)
                break


if __name__ == '__main__':
    h, w = int(input('height: ')), int(input('weight: '))
    first = np.matrix(np.matrix(list(range(1, h * w)) + [0]).reshape(h, w))
    shuffle(first)
    print(first)
    # first = np.matrix("2, 4, 3; 1, 8, 5; 7, 0, 6")
    r, steps, b = get_tree(first)
    print(b.directions[1:])
    print(len(b.directions) - 1)
    print(steps)
