import random
import time

class SOSBot:
    def __init__(self, wrap_around=True):
        self.max_depth = 2
        self.wrap_around = wrap_around

    def choose_move(self, board):
        # "Smart Greedy" - Depth 2 Minimax
        # Goal: Maximize My Score - Opponent's Potential Score
        
        move, letter = self.best_move_minimax(board)
        return move, letter

    def best_move_minimax(self, board):
        size = len(board)
        best_score = -float('inf')
        candidates = []
        
        # 1. Get all legal moves
        moves = [(r, c) for r in range(size) for c in range(size) if board[r][c] == ' ']
        random.shuffle(moves) # Randomize to avoid predictable deterministic play
        
        for r, c in moves:
            for letter in ['S', 'O']:
                # Simulate My Move
                board[r][c] = letter
                points_gained = self._count_sos(board, r, c)
                
                if points_gained > 0:
                    # If we score, we KEEP TURN.
                    # Simple heuristic: Scoring is almost always good.
                    score = 100 + points_gained
                    
                    # Optional: We could recurse here to see if this leads to a chain.
                    # But for "Greedy Plus", just taking the points is usually right.
                else:
                    # Turn passes to opponent.
                    # Minimize their best response.
                    opponent_max_score = 0
                    
                    # Scan opponent's moves
                    # Optimization: We only need to check moves in the vicinity? 
                    # No, strict safety requires checking all.
                    # On 8x8, remaining moves ~60. 2 letters = 120 checks.
                    # Total complexity: 120 * 120 = 14,400 checks. Very fast.
                    
                    for or_r in range(size):
                        for or_c in range(size):
                            if board[or_r][or_c] == ' ':
                                for op_letter in ['S', 'O']:
                                    board[or_r][or_c] = op_letter
                                    op_pts = self._count_sos(board, or_r, or_c)
                                    if op_pts > opponent_max_score:
                                        opponent_max_score = op_pts
                                    board[or_r][or_c] = ' ' # Undo op
                                    
                                    # Optimization: If opponent can score, this branch is bad.
                                    # Unless we found a move where opponent scores 0, we can't prune much logic here
                                    # without full alpha-beta. But full search is fast enough.
                    
                    score = -opponent_max_score * 10
                    
                    # Secondary Tie-Breaker: Create Threats (Setups)
                    # If opponent can't score (score == 0), prefer moves that Create Potential for us.
                    if score == 0:
                        # Reduced weight for clustering (0.01) to be "more open"
                        score += self._evaluate_potential(board, r, c) * 0.01
                        # Add randomness to avoid repetitive patterns in equivalent positions
                        score += random.uniform(0, 0.05)

                board[r][c] = ' ' # Undo my move
                
                if score > best_score:
                    best_score = score
                    candidates = [((r, c), letter)]
                elif score == best_score:
                    candidates.append(((r, c), letter))
        
        if not candidates:
            return (0,0), 'S' # Should not happen unless board full
            
        return random.choice(candidates)

    def _count_sos(self, board, r, c):
        # Returns number of SOS formed by placing piece at r, c
        size = len(board)
        count = 0
        piece = board[r][c]
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            # If 'S', check 'S-O-S' centered on neighbor
            if piece == 'S':
                 # Check side 1: (r+dr, c+dc) is O, (r+2dr, c+2dc) is S
                 if self._check_pt(board, r+dr, c+dc, 'O') and self._check_pt(board, r+2*dr, c+2*dc, 'S'):
                     count += 1
                 # Check side 2: (r-dr, c-dc) is O, (r-2dr, c-2dc) is S
                 if self._check_pt(board, r-dr, c-dc, 'O') and self._check_pt(board, r-2*dr, c-2*dc, 'S'):
                     count += 1
                     
            # If 'O', check 'S-O-S' centered on self
            elif piece == 'O':
                if self._check_pt(board, r-dr, c-dc, 'S') and self._check_pt(board, r+dr, c+dc, 'S'):
                    count += 1
                    
        return count
        
    def _check_pt(self, board, r, c, target):
        size = len(board)
        
        if self.wrap_around:
            r = r % size
            c = c % size
            return board[r][c] == target
        else:
            if 0 <= r < size and 0 <= c < size:
                return board[r][c] == target
            return False
        
    def _evaluate_potential(self, board, r, c):
        # Heuristic: Value moves that adjacent to existing letters (clustering)
        # Random moves in corners are usually useless.
        score = 0
        size = len(board)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < size and 0 <= nc < size:
                     if board[nr][nc] != ' ':
                         score += 1
        return score
