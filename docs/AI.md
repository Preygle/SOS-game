# AlphaZero AI Architecture

This project implements a state-of-the-art AI agent based on the **AlphaZero** algorithm, capable of mastering the SOS game through self-play reinforcement learning.

## Neural Network Architecture
The core "brain" of the AI is a **Deep Residual Network (ResNet)** designed to predict optimal moves and evaluate board positions.

### input representation
The game state is encoded into an `8x8x6` tensor:
*   **Channel 0**: 'S' pieces (Boolean mask)
*   **Channel 1**: 'O' pieces (Boolean mask)
*   **Channel 2**: Empty cells (Boolean mask)
*   **Channel 3**: Current Player turn (0 or 1 plane)
*   **Channel 4**: Player 1 Score (Normalized)
*   **Channel 5**: Player 2 Score (Normalized)

### Network Layers
*   **Convolutional Input Block**: Initial feature extraction.
*   **Residual Tower**: **6 Residual Blocks** with **128 Filters** each, enabling deep learning of complex spatial patterns.
*   **Policy Head**: Outputs a probability distribution over all possible moves (128 actions: 64 cells * 2 symbols).
*   **Value Head**: Outputs a scalar value [-1, 1] estimating the probability of winning from the current state.

## Monte Carlo Tree Search (MCTS)
The AI uses MCTS to look ahead and simulate future game states.
*   **Simulations per Move**: **50** — tuned for Colab GPU feasibility while preserving strategic depth.
*   **Exploration Constant (c_puct)**: **1.5** — slightly elevated to compensate for fewer simulations with broader exploration.
*   **Selection**: Uses Upper Confidence Bound (UCB) guided by the Neural Network's prior probabilities.

## Training Configuration

> **Design philosophy**: The model architecture (`num_res_blocks`, `num_channels`) is kept at full Grandmaster capacity — these parameters define *capability ceiling*, not training time. Training time is dominated by `num_simulations` (how many MCTS rollouts per move). Cutting that from 200 → 50 gives a **4× speedup** while the trained model retains full strategic depth.

| Parameter | Value | Effect |
| :--- | :--- | :--- |
| **num_res_blocks** | 6 | Full depth — holds complex multi-move strategies |
| **num_channels** | 128 | Full bandwidth — captures subtle board patterns |
| **iterations** | 300 | Main learning loop (300 × 25 = 7,500 total games) |
| **self_play_games** | 25 | Games generated per iteration |
| **num_simulations** | 50 | MCTS rollouts per move — primary speed lever |
| **c_puct** | 1.5 | Exploration constant (higher = broader search) |
| **batch_size** | 128 | Samples per gradient step — stable GPU utilisation |
| **epochs** | 4 | Training passes per collected batch |
| **buffer_size** | 10,000 | Replay buffer — retains more learning history |
| **lr** | 2e-3 | Learning rate with Adam optimiser |
| **Est. Colab T4 time** | ~4–6 h | With CUDA, resumable via checkpoints |


## Neural Network Block Diagram

![AlphaZero Architecture](screenshots/AlphaGo.png)

## Diagram Annotations & Key Highlights

For your block diagram, consider highlighting these unique aspects of the AlphaZero implementation for SOS:

1.  **Dual Action Output (Policy Head)**:
    *   Unlike Go (Move only) or Chess (Move piece), SOS requires choosing **Piece Type ('S' or 'O')** per cell.
    *   **Annotation**: "Output Vector Size 128: [0-63] = Place 'S', [64-127] = Place 'O'".

2.  **Score-Aware Input (Input Block)**:
    *   The game state is not just board positions; the current score difference is critical for strategy.
    *   **Annotation**: "Channels 4-5 encode Normalized Player Scores".

3.  **Turn-Sensitive Value**:
    *   Due to bonus turns, the value prediction is strictly for the **Current Player**, which may not alternate every turn.

## Performance
Even with limited training, the AlphaZero AI demonstrates strategic depth, capable of:
*   Blocking opponent SOS attempts.
*   Setting up multi-point combinations.
*   Utilizing Orbit Mode wrapping to create unexpected connections.
*   Adapting to both Standard and Wrapped board topologies.
