import random


class SOSBot:
    def __init__(self):
        pass

    def choose_move(self, board):
        size = len(board)
        # First, check for any move that immediately creates SOS for bot
        for r in range(size):
            for c in range(size):
                if board[r][c] == ' ':
                    for letter in ['S', 'O']:
                        board[r][c] = letter
                        if self._would_score(board, r, c):
                            board[r][c] = ' '  # undo
                            return (r, c), letter
                        board[r][c] = ' '  # undo

        # check if opponent can score next turn and block it
        for r in range(size):
            for c in range(size):
                if board[r][c] == ' ':
                    for letter in ['S', 'O']:
                        board[r][c] = letter
                        if self._would_score(board, r, c):
                            board[r][c] = ' '  # undo
                            # block by placing opposite letter
                            return (r, c), 'O' if letter == 'S' else 'S'
                        board[r][c] = ' '  # undo

        # Otherwise, pick random empty cell
        empty_cells = [(r, c) for r in range(size)
                       for c in range(size) if board[r][c] == ' ']
        move = random.choice(empty_cells)
        letter = random.choice(['S', 'O'])
        return move, letter

    def _would_score(self, board, r, c):
        size = len(board)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            # Check backward and forward to see if SOS is formed
            for delta in [-2, -1, 0]:
                positions = []
                for i in range(3):
                    nr = r + (delta + i) * dr
                    nc = c + (delta + i) * dc
                    if 0 <= nr < size and 0 <= nc < size:
                        positions.append(board[nr][nc])
                    else:
                        break
                if len(positions) == 3 and positions[0] == 'S' and positions[1] == 'O' and positions[2] == 'S':
                    return True
        return False
