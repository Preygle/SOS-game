import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
import json
import time
import argparse
from collections import deque

from models import AlphaZeroResNet
from alpha_mcts import AlphaMCTS, GameWrapper
# from game_logic import SOSGame  # Now handled in GameWrapper inside alpha_mcts


# Training Args
ARGS = {
    'num_res_blocks': 6,
    'num_channels': 128,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'epochs': 1, # Epochs per self-play batch
    'num_simulations': 200, # MCTS Sims per move (Increase for better play)
    'c_puct': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'self_play_games': 50, # Games per iteration
    'iterations': 1000, # Total training loops
    'buffer_size': 5000,
    'checkpoint_dir': 'checkpoints',
    'log_file': 'training_log.csv',
    'replay_dir': 'replays'
}

def train():
    parser = argparse.ArgumentParser(description='AlphaZero Training')
    parser.add_argument('--scratch', action='store_true', help='Restart training from scratch')
    args = parser.parse_args()

    if not os.path.exists(ARGS['checkpoint_dir']): os.makedirs(ARGS['checkpoint_dir'])
    if not os.path.exists(ARGS['replay_dir']): os.makedirs(ARGS['replay_dir'])
    
    # Handle Scratch: clear checkpoints
    if args.scratch:
        print("Scrubbing checkpoints for fresh start...")
        for f in os.listdir(ARGS['checkpoint_dir']):
            if f.startswith("model_") and f.endswith(".pth") or f == 'best.pth':
                os.remove(os.path.join(ARGS['checkpoint_dir'], f))
    
    # Init Model
    # Input channels: 6 (S, O, Empty, Player, Score0, Score1)
    nnet = AlphaZeroResNet(8, ARGS['num_res_blocks'], ARGS['num_channels'], input_channels=6).to(ARGS['device'])
    optimizer = optim.Adam(nnet.parameters(), lr=ARGS['lr'], weight_decay=ARGS['weight_decay'])
    
    # Check for existing checkpoint
    start_iter = 0
    if not args.scratch:
        checkpoint_files = [f for f in os.listdir(ARGS['checkpoint_dir']) if f.startswith('model_') and f.endswith('.pth')]
        if checkpoint_files:
            # Sort by iteration number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            start_iter = int(latest_checkpoint.split('_')[1].split('.')[0])
            print(f"Resuming from checkpoint: {latest_checkpoint} (Next Iter {start_iter + 1})")
            checkpoint_path = os.path.join(ARGS['checkpoint_dir'], latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=ARGS['device'])
            
            # Support both direct state_dict (legacy) and full dict (new)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                nnet.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                nnet.load_state_dict(checkpoint) # Legacy fallback
    
    # Init MCTS
    mcts = AlphaMCTS(nnet, GameWrapper, ARGS)
    
    # Replay Buffer
    replay_buffer = deque(maxlen=ARGS['buffer_size'])
    
    # Logging - Append mode if resuming
    mode = 'a' if start_iter > 0 else 'w'
    with open(ARGS['log_file'], mode) as f:
        if mode == 'w':
             f.write('Iteration,PolicyLoss,ValueLoss,TotalLoss,BestWinner,AvgMoves\n')

    print(f"Starting training on {ARGS['device']}...")
    

    for i in range(start_iter, ARGS['iterations']):
        print(f"--- Iteration {i+1}/{ARGS['iterations']} ---")
        
        # 1. Self Play
        iteration_examples = []
        nnet.eval()
        
        total_moves = 0
        
        for g in range(ARGS['self_play_games']):
            game = GameWrapper()
            game_history = [] # (state_encoded, pi, current_player)
            replay_moves = [] # For saving replay
            
            while not game.is_finished():
                state = game.get_state()
                temp = 1 if len(game_history) < 10 else 0
                
                # Get MCTS Probs
                probs = mcts.get_probs(state, temperature=temp)
                
                # Store
                sym = GameWrapper.encode_state(state)
                game_history.append([sym, probs, game.current_player])
                
                # Pick Action
                if temp == 0:
                    action = np.argmax(probs)
                else:
                    action = np.random.choice(len(probs), p=probs)
                    
                replay_moves.append(int(action))
                game.step(action)
                
            total_moves += len(game_history)
            
            # Game Over
            winner = game.get_winner() # 0, 1, 2
            
            # Assign rewards
            for hist in game_history:
                p = hist[2]
                if winner == 0:
                    v = 0
                elif winner == (p + 1):
                    v = 1
                else:
                    v = -1
                
                hist[2] = v # Replace player with value
                iteration_examples.append(hist)
            
            # Save Best Replay (Arbitrary: Last game of iteration)
            if g == ARGS['self_play_games'] - 1:
                replay_data = {
                    'iteration': i,
                    'winner': winner,
                    'moves': replay_moves,
                    'final_scores': game.game.scores
                }
                with open(os.path.join(ARGS['replay_dir'], f'replay_{i}.json'), 'w') as f:
                    json.dump(replay_data, f)

        replay_buffer.extend(iteration_examples)
        
        # 2. Train
        nnet.train()
        pi_losses = []
        v_losses = []
        
        if len(replay_buffer) > ARGS['batch_size']:
             # Train for K epochs
             for _ in range(ARGS['epochs'] * (len(iteration_examples) // ARGS['batch_size'] + 1)):
                 batch = random.sample(replay_buffer, ARGS['batch_size'])
                 states, pis, vs = zip(*batch)
                 
                 states = torch.stack(states).to(ARGS['device'])
                 pis = torch.tensor(np.array(pis), dtype=torch.float32).to(ARGS['device'])
                 vs = torch.tensor(np.array(vs), dtype=torch.float32).to(ARGS['device'])
                 
                 out_pi, out_v = nnet(states)
                 
                 # Loss
                 l_pi = -torch.mean(torch.sum(pis * F.log_softmax(out_pi, dim=1), dim=1))
                 
                 # V: MSE
                 l_v = F.mse_loss(out_v.view(-1), vs)
                 
                 total_loss = l_pi + l_v
                 
                 optimizer.zero_grad()
                 total_loss.backward()
                 optimizer.step()
                 
                 pi_losses.append(l_pi.item())
                 v_losses.append(l_v.item())

        # Log
        avg_pi = np.mean(pi_losses) if pi_losses else 0
        avg_v = np.mean(v_losses) if v_losses else 0
        avg_moves = total_moves / ARGS['self_play_games']
        
        print(f"Loss PI: {avg_pi:.4f}, Loss V: {avg_v:.4f}, Avg Moves: {avg_moves:.1f}")
        
        with open(ARGS['log_file'], 'a') as f:
            f.write(f"{i},{avg_pi},{avg_v},{avg_pi+avg_v},0,{avg_moves}\n")
            
        # Checkpoint (Save every iteration to assume resume capability)
        if True: # (i+1) % 1 == 0
            checkpoint_state = {
                'iteration': i + 1,
                'model_state_dict': nnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint_state, os.path.join(ARGS['checkpoint_dir'], f'model_{i+1}.pth'))
            
    # Save Final
    final_state = {
        'iteration': ARGS['iterations'],
        'model_state_dict': nnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(final_state, os.path.join(ARGS['checkpoint_dir'], 'best.pth'))
    print("Training Complete.")

if __name__ == "__main__":
    train()
