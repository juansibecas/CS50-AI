"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    turns_played = 0
    for row in board:
        for square in row:
            if square != EMPTY:
                turns_played += 1
    if turns_played % 2 == 0:  # odd turns return X, even turns return O
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = []
    for i, row in enumerate(board):
        for j, square in enumerate(row):
            if square == EMPTY:
                actions.append((i, j))

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    row = action[0]
    column = action[1]
    if board[row][column] != EMPTY:
        raise ValueError("square already taken")
    modified_board = copy.deepcopy(board)
    modified_board[row][column] = player(board)  # func player(board) returns X or O
    return modified_board



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    x_counter = [0, 0]                  # counter[0] is horizontal and counter[1] is vertical
    o_counter = [0, 0]

    for i in range(3):                  # checks for horizontal or vertical wins
        for j in range(3):
            horizontal = board[i][j]
            vertical = board[j][i]
            if horizontal == X:
                x_counter[0] += 1
            elif horizontal == O:
                o_counter[0] += 1
            if vertical == X:
                x_counter[1] += 1
            elif vertical == O:
                o_counter[1] += 1
        if x_counter[0] == 3 or x_counter[1] == 3:
            return X
        elif o_counter[0] == 3 or o_counter[1] == 3:
            return O
        x_counter = [0, 0]
        o_counter = [0, 0]

    for i in range(3):          # checks for diagonal wins
        diag1 = board[i][i]     # counter[0] is diag1 and counter[1] is diag2
        diag2 = board[i][-i-1]
        if diag1 == X:
            x_counter[0] += 1
        elif diag1 == O:
            o_counter[0] += 1
        if diag2 == X:
            x_counter[1] += 1
        elif diag2 == O:
            o_counter[1] += 1

    if x_counter[0] == 3 or x_counter[1] == 3:
        return X
    elif o_counter[0] == 3 or o_counter[1] == 3:
        return O
    else:
        return None



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    else:
        for row in board:
            if EMPTY in row:
                return False
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_is = winner(board)
    if winner_is == X:
        return 1
    elif winner_is == O:
        return -1
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    utilities = []
    next_boards = []
    possible_actions = actions(board)
    for action in possible_actions:
        next_boards.append(Board(board, action))

    for next_board in next_boards:
        next_board.get_next_boards()
        next_board.get_utility()
        utilities.append(next_board.utility)

    if player(board) == X:  # selects the action corresponding to the board with the highest or lowest utility
        action_to_take = next_boards[utilities.index(max(utilities))].action_taken
    else:
        action_to_take = next_boards[utilities.index(min(utilities))].action_taken

    return action_to_take


class Board:

    def __init__(self, prev_board, action):
        self.board = result(prev_board, action)
        self.action_taken = action

    def get_next_boards(self):
        self.possible_actions = actions(self.board)
        self.next_boards = []
        for action in self.possible_actions:
            self.next_boards.append(Board(self.board, action))

    def get_utility(self):
        if terminal(self.board):                # if board is terminal we can just call utility(board)
            self.utility = utility(self.board)

        else:
            for i, next_board in enumerate(self.next_boards):  # if board is not terminal, its utility is the highest
                next_board.get_next_boards()                   # or lowest of its successors' utilities, depending on if
                next_board.get_utility()                       # its X's or O's turn
                if player(self.board) == X:
                    if i == 0:
                        value = -10
                    if next_board.utility > value:
                        value = next_board.utility
                    if value == 1:                              # since 1 is the best possible result for X,
                        break                                   # theres no need to calculate the rest
                else:
                    if i == 0:
                        value = 10
                    if next_board.utility < value:
                        value = next_board.utility
                    if value == -1:
                        break
            self.utility = value