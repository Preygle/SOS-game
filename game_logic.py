import numpy as np

class SOSGame:
    def __init__(self, board_size=8, wrap_around=True):
        """
        Initializes the board, scores, current player, and other game variables.
        """
        self.board_size = board_size
        self.wrap_around = wrap_around
        self.reset()

    def reset(self):
        """
        Resets the game to its initial state.
        Returns the initial state of the board.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 0
        self.scores = {0: 0, 1: 0}
        self.sos_patterns = set()
        return self.get_state()

    def get_state(self):
        """
        Returns the current state of the board in a format suitable for the neural network.
        This should be an 8x8x3 NumPy array with one-hot encoding:
        - Channel 0: 1 for 'S', 0 otherwise.
        - Channel 1: 1 for 'O', 0 otherwise.
        - Channel 2: 1 for empty cells, 0 otherwise.
        """
        state = np.zeros((self.board_size, self.board_size, 3), dtype=int)
        state[:, :, 0] = (self.board == 1)
        state[:, :, 1] = (self.board == 2)
        state[:, :, 2] = (self.board == 0)
        return np.transpose(state, (2, 0, 1)) # Transpose to get shape (3, 8, 8)

    def get_valid_actions(self):
        """
        Returns a list of all possible valid actions from the current state.
        An action is an integer from 0 to 127:
        - 0-63: Place 'S' in cells 0-63.
        - 64-127: Place 'O' in cells 64-127.
        """
        valid_actions = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == 0:
                    cell_index = r * self.board_size + c
                    valid_actions.append(cell_index)      # Action for 'S'
                    valid_actions.append(64 + cell_index) # Action for 'O'
        return valid_actions

    def step(self, action):
        """
        Takes an action, updates the game state, and returns the results.
        """
        if action < 0 or action >= 128:
            # Invalid action outside the defined range
            return self.get_state(), -10, True, {"error": "Invalid action"}

        piece = 1 if action < 64 else 2
        cell_index = action % 64
        row, col = divmod(cell_index, self.board_size)

        if self.board[row, col] != 0:
            # Punish for trying to play on a non-empty cell
            return self.get_state(), -10, True, {"error": "Cell not empty"}

        self.board[row, col] = piece
        
        new_sos_count = self._check_sos(row, col)
        
        reward = new_sos_count if new_sos_count > 0 else -0.1 # Small penalty for non-scoring moves

        if new_sos_count > 0:
            self.scores[self.current_player] += new_sos_count
        else:
            self.current_player = 1 - self.current_player

        done = np.all(self.board != 0)
        
        info = {}
        if done:
            info['winner'] = np.argmax(list(self.scores.values())) if self.scores[0] != self.scores[1] else -1 # -1 for draw

        return self.get_state(), reward, done, info

    def _check_sos(self, r, c):
        """
        A helper function to check for "SOS" formations involving the newly placed piece.
        Supports optional wrap-around.
        Returns the number of new SOS patterns formed.
        """
        new_patterns = 0
        piece = self.board[r, c]
        
        # Helper for coordinates
        def get_coords(r, c):
            if self.wrap_around:
                return r % self.board_size, c % self.board_size
            else:
                return r, c
                
        def is_valid(r, c):
            if self.wrap_around: return True # Always valid if wrapping
            return 0 <= r < self.board_size and 0 <= c < self.board_size
        
        # Case 1: The new piece is 'O' in the middle of an S-O-S
        if piece == 2: # 'O'
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                r1, c1 = get_coords(r - dr, c - dc)
                r2, c2 = get_coords(r + dr, c + dc)
                
                if is_valid(r1, c1) and is_valid(r2, c2):
                    if self.board[r1, c1] == 1 and self.board[r2, c2] == 1:
                        pattern = tuple(sorted(((r1, c1), (r, c), (r2, c2))))
                        if pattern not in self.sos_patterns:
                            self.sos_patterns.add(pattern)
                            new_patterns += 1

        # Case 2: The new piece is 'S' at one end of an S-O-S
        elif piece == 1: # 'S'
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]:
                # Check for S(new)-O-S
                ro, co = get_coords(r + dr, c + dc)
                rs, cs = get_coords(r + 2 * dr, c + 2 * dc)
                
                if is_valid(ro, co) and is_valid(rs, cs):
                    if self.board[ro, co] == 2 and self.board[rs, cs] == 1:
                        pattern = tuple(sorted(((r, c), (ro, co), (rs, cs))))
                        if pattern not in self.sos_patterns:
                            self.sos_patterns.add(pattern)
                            new_patterns += 1

        return new_patterns
