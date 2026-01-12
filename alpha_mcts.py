import math
import numpy as np
import torch
import copy
from game_logic import SOSGame

# Wrapper for SOSGame to match MCTS interface
class GameWrapper:
    WRAP_AROUND = True # Default, can be modified by main app
    
    def __init__(self, state=None):
        self.game = SOSGame(board_size=8, wrap_around=GameWrapper.WRAP_AROUND)
        if state is not None:
            self.set_state(state)
            
    def set_state(self, state_dict):
        # state_dict should contain board, scores, current_player
        self.game.board = np.copy(state_dict['board'])
        self.game.scores = state_dict['scores'].copy()
        self.game.current_player = state_dict['current_player']
        self.game.sos_patterns = set(tuple(p) for p in state_dict['sos_patterns'])
        
    def get_state(self):
        return {
            'board': np.copy(self.game.board),
            'scores': self.game.scores.copy(),
            'current_player': self.game.current_player,
            'sos_patterns': list(self.game.sos_patterns)
        }
        
    def step(self, action):
        _, _, done, _ = self.game.step(action)
        return done
        
    def is_finished(self):
        return np.all(self.game.board != 0)
        
    def get_winner(self):
        s0 = self.game.scores[0]
        s1 = self.game.scores[1]
        
        if s0 > s1: return 1
        elif s1 > s0: return 2
        else: return 0 
        
    def get_valid_moves(self, state=None):
        if state:
            board = state['board']
            moves = []
            for r in range(8):
                for c in range(8):
                    if board[r, c] == 0:
                        idx = r * 8 + c
                        moves.append(idx)      # S
                        moves.append(64 + idx) # O
            return moves
        else:
            return self.game.get_valid_actions()
            
    @property
    def current_player(self):
        return self.game.current_player # 0 or 1
        
    @staticmethod
    def get_player_from_state(state):
        return state['current_player']
        
    @staticmethod
    def encode_state(state):
        board = state['board']
        # Expanded to 6 channels:
        # 0: S pieces
        # 1: O pieces
        # 2: Empty cells
        # 3: Current Player (0 or 1 plane)
        # 4: Score P0 (Normalized)
        # 5: Score P1 (Normalized)
        
        tensor = np.zeros((6, 8, 8), dtype=np.float32)
        tensor[0] = (board == 1)
        tensor[1] = (board == 2)
        tensor[2] = (board == 0)
        tensor[3] = state['current_player'] 
        
        # Normalize scores (max theoretical score is ~64ish? Let's say 40 is high)
        s0 = state['scores'][0] / 40.0
        s1 = state['scores'][1] / 40.0
        
        tensor[4] = s0
        tensor[5] = s1
        
        return torch.tensor(tensor)

class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None, prior=0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}  # action -> MCTSNode
        
        self.n_visits = 0
        self.q_value = 0  # Average value
        self.w_sum = 0    # Total value sum
        self.prior = prior
        
    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                # Note: We don't create child state yet to save memory, 
                # we'll create it when we traverse
                self.children[action] = MCTSNode(None, parent=self, action_taken=action, prior=prob)
                
    def is_expanded(self):
        return len(self.children) > 0

    def best_child(self, c_puct=1.0):
        # UCB formula
        # Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        sqrt_n = math.sqrt(self.n_visits)
        
        for action, child in self.children.items():
            score = child.q_value + c_puct * child.prior * sqrt_n / (1 + child.n_visits)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

class AlphaMCTS:
    def __init__(self, model, game_cls, args):
        self.model = model
        self.game_cls = game_cls
        self.args = args # dictionary with 'num_simulations', 'c_puct', 'device'
        
    @torch.no_grad()
    def search(self, root_state):
        root = MCTSNode(root_state, prior=0)
        
        # Initial expansion
        policy, _ = self.compute_policy_value(root_state)
        valid_moves = self.game_cls.get_valid_moves(root_state)
        
        # Mask invalid moves
        policy_masked = {a: policy[a] for a in valid_moves}
        sum_probs = sum(policy_masked.values())
        if sum_probs > 0:
            policy_masked = {k: v/sum_probs for k, v in policy_masked.items()} # Normalize
        else:
             # If all probs are 0 (shouldn't happen often), uniform dist
            policy_masked = {a: 1.0/len(valid_moves) for a in valid_moves}

        root.expand(policy_masked.items())
        
        for _ in range(self.args['num_simulations']):
            node = root
            game = self.game_cls(state=node.state) # Clone game state
            
            # 1. Select
            while node.is_expanded():
                action, node = node.best_child(self.args['c_puct'])
                # If node.state is None, it means we just entered a node that hasn't been stepped in simulation yet.
                # However, for efficiency, usually we store state in node only when expanding? 
                # Actually, simpler: Re-simulate game steps as we descend.
                game.step(action)
            
            # Node is now a leaf (unexpanded).
            # If state not stored, use the game's current state
            leaf_state = game.get_state()
            node.state = leaf_state 
            
            # 2. Expand & Evaluate
            value = 0
            if not game.is_finished():
                policy, value = self.compute_policy_value(leaf_state)
                valid_moves = game.get_valid_moves()
                
                # Mask and Normalize
                policy_masked = {a: policy[a] for a in valid_moves}
                sum_probs = sum(policy_masked.values())
                if sum_probs > 0:
                    policy_masked = {k: v/sum_probs for k, v in policy_masked.items()}
                else:
                    policy_masked = {a: 1.0/len(valid_moves) for a in valid_moves}
                    
                node.expand(policy_masked.items())
            else:
                # Game over
                # Value is usually framed from perspective of current player.
                # If game is done, we know the winner.
                winner = game.get_winner()
                # If winner is the player who Just moved to get here? 
                # Careful with SOS turn keeping.
                # Simplification: Value is always [1 if current_player wins, -1 if loses]
                # But 'current_player' might have changed or not.
                # 'value' from Net is for 'node.state' current_player.
                
                # If game is Over at 'leaf_state', who won?
                if winner == 0: # Draw
                    value = 0
                else:
                    # if winner is 1 (P1) and current_player at leaf_state is 0 (P1), value = 1.
                    # if winner is 1 (P1) and current_player at leaf_state is 1 (P2), value = -1.
                    cp = game.current_player
                    if (winner == 1 and cp == 0) or (winner == 2 and cp == 1):
                         value = 1
                    else:
                         value = -1

            # 3. Backpropagate
            # We need to backpropagate 'value'. 
            # Crucially: Value is always "Likelihood of Current Player winning".
            # When we go UP the tree, if the parent's current_player is DIFFERENT, we flip value.
            # If the parent's current_player is SAME (Bonus turn), we keep value.
            
            while node is not None:
                node.n_visits += 1
                node.w_sum += value
                node.q_value = node.w_sum / node.n_visits
                
                if node.parent:
                    # Check if player changed between parent and current node
                    # This requires Game logic to know if player switched.
                    # We can infer from states or store 'player' in node.
                    # Let's store 'player_turn' in Node.
                    pass 
                
                    # Actually, standard AlphaZero negation: value = -value
                    # ONLY if players switch. In SOS, player might NOT switch.
                    # We need to handle this.
                    if node.parent.state is None: 
                        # Root parent logic handled differently? No, root has state.
                        pass
                    
                    # To do this correctly without re-simulating everything:
                    # Store 'player_who_made_move' in the node?
                    # Or 'player_at_this_node'.
                    pass
                
                # Simple Hack for standard Turn-based (Switch every time): value = -value
                # SOS Version:
                # If node.player == node.parent.player:
                #    value = value (Same player, so if I win, you win)
                # Else:
                #    value = -value (Switch, so if I win, you lose)
                
                # To do this, we need 'player' index in MCTSNode.
                
                old_node = node
                node = node.parent
                if node:
                     # We need to know who is playing at 'node' vs 'old_node'
                     # We can decode from state or store it.
                     # Let's assume state has player info.
                     pass
                     # Ideally we need a game.get_player(state) function.
                     
                     # FOR NOW: Let's assume strict alternation for simplicity in first pass?
                     # NO, SOS relies on bonus turns. 
                     
                     # FIX: Store player index in MCTSNode
                     current_player_at_child = self.game_cls.get_player_from_state(old_node.state)
                     current_player_at_parent = self.game_cls.get_player_from_state(node.state)
                     
                     if current_player_at_child != current_player_at_parent:
                         value = -value
            
    def compute_policy_value(self, state):
        # Prepare input
        encoded = self.game_cls.encode_state(state).to(self.args['device'])
        encoded = encoded.unsqueeze(0) # Batch dim
        
        policy_logits, value = self.model(encoded)
        
        policy = torch.softmax(policy_logits, dim=1).cpu().detach().numpy()[0]
        value = value.item()
        return policy, value
        
    def get_action_prob(self, root_state, temperature=1.0):
        # Create root if not exists (search should be called before this)
        # But here we assume search has populated the counts.
        # Rerun search for safety? No, external loop calls search().
        
        # We need the root node corresponding to root_state. 
        # Since we create a new tree every move in this simple version:
        
        # This function assumes 'search' was just called and we have a 'root' object lying around? 
        # Refactor: 'search' returns the prob distribution?
        pass

    # Refactored public method
    def get_probs(self, state, temperature=1.0):
        root = MCTSNode(state)
        
        # We need to know the player to backprop correctly.
        # Let's update expand/backprop logic inside Search to be self-contained or use a robust method.
        # See simplified loop above.
        
        for _ in range(self.args['num_simulations']):
            node = root
            game = self.game_cls(state=node.state) # Copy
            
            path = [node]
            
            # Select
            while node.is_expanded():
                action, node = node.best_child(self.args['c_puct'])
                game.step(action)
                path.append(node)
                
            leaf_state = game.get_state()
            node.state = leaf_state
            
            # Evaluate
            value = 0
            if not game.is_finished():
                policy, v = self.compute_policy_value(leaf_state)
                valid_moves = game.get_valid_moves()
                
                # Valid Mask
                policy_masked = {a: policy[a] for a in valid_moves}
                sum_p = sum(policy_masked.values())
                if sum_p > 0:
                    for a in policy_masked: policy_masked[a] /= sum_p
                else:
                    policy_masked = {a: 1.0/len(valid_moves) for a in valid_moves}
                
                node.expand(policy_masked.items())
                value = v
            else:
                winner = game.get_winner()
                p_turn = game.current_player
                # If p_turn is 0 (P1) and winner is 1 (P1) -> +1
                if winner == -1: # Draw
                    value = 0
                else: 
                     # winner is 1 or 2. p_turn is 0 or 1.
                     # mappings: p_turn 0 = P1 (winner 1). p_turn 1 = P2 (winner 2).
                     win_val = 1 if winner == (p_turn + 1) else -1
                     value = win_val

            # Backprop
            # Path is [Root, Child1, Child2, ..., Leaf]
            # Value is relative to Leaf's player.
            
            # We iterate backwards
            for i in range(len(path)-1, -1, -1):
                node = path[i]
                node.n_visits += 1
                node.w_sum += value
                node.q_value = node.w_sum / node.n_visits
                
                # Before moving to parent, check if we need to flip value for parent's perspective.
                if i > 0:
                    parent = path[i-1]
                    # Check player change
                    # We need helper to get player from state.
                    # We can cache it in node.
                    curr_p = self.game_cls.get_player_from_state(node.state)
                    parent_p = self.game_cls.get_player_from_state(parent.state)
                    
                    if curr_p != parent_p:
                        value = -value
        
        # Calculate return probs
        counts = {a: child.n_visits for a, child in root.children.items()}
        actions = list(counts.keys())
        visits = list(counts.values())
        
        if temperature == 0:
            best_a = actions[np.argmax(visits)]
            probs = {a: (1.0 if a == best_a else 0.0) for a in actions}
        else:
            # v^(1/t)
            visits = [v ** (1.0/temperature) for v in visits]
            sum_v = sum(visits)
            probs = {actions[i]: visits[i]/sum_v for i in range(len(actions))}
            
        # Return array of size 128
        full_probs = np.zeros(128)
        for a, p in probs.items():
            full_probs[a] = p
            
        return full_probs

