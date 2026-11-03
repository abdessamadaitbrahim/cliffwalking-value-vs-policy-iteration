# Value Iteration vs Policy Iteration on CliffWalking

Lab project (Deep Learning module, Big Data & AI track) at ENSA Tetouan, Abdelmalek Essaadi University, academic year 2024-2025.

**Author:** Abdessamad Ait Brahim
**Supervisor:** Anas Belcaid

This project implements two dynamic programming algorithms, **Value Iteration** and **Policy Iteration**, on the `CliffWalking-v1` environment of [Gymnasium](https://gymnasium.farama.org/), and compares their convergence, efficiency and solution quality.

## The environment

A 4x12 grid. The agent starts at the bottom-left corner and must reach the bottom-right corner without falling off the cliff.

| Property | Value |
|---|---|
| States | 48 (4 x 12) |
| Actions | 4 (0: up, 1: right, 2: down, 3: left) |
| Start | state 36 (bottom-left) |
| Goal | state 47 (bottom-right) |
| Cliff | states 37 to 46 |
| Dynamics | deterministic |

Rewards: `-1` for every step, `-100` for falling off the cliff (the agent goes back to the start).

## Algorithms

Both algorithms use the known transition model of the environment (`env.unwrapped.P`) with a discount factor `gamma = 0.9` and a convergence threshold `theta = 1e-6`.

- **Value Iteration:** repeatedly applies the Bellman optimality update `V(s) <- max_a sum P(s'|s,a) [R(s,a,s') + gamma V(s')]` (synchronous updates, up to 5000 iterations) until the largest change is below `theta`, then extracts the greedy policy.
- **Policy Iteration:** alternates policy evaluation and greedy policy improvement until the policy no longer changes (up to 100 iterations). It starts from the "always go right" policy.

## Results

![Comparison](results/comparison_cliffwalking.png)

| | Value Iteration | Policy Iteration |
|---|---|---|
| Success rate | 100% | 100% |
| Mean reward | -13.00 | -13.00 |
| Mean episode length | 13 steps | 13 steps |
| Iterations | ~15 | ~15 |
| Execution time | ~0.0094 s | ~0.3259 s |

Both algorithms reach the same optimal policy. Value Iteration was about 35 times faster in the lab run, because each update is a single simple step, while Policy Iteration has to fully evaluate every intermediate policy. Exact timings depend on the machine and on the implementation.

Optimal policy found by both algorithms:

```
[→] [→] [→] [→] [→] [→] [→] [→] [→] [→] [→] [↓]
[→] [→] [→] [→] [→] [→] [→] [→] [→] [→] [→] [↓]
[→] [→] [→] [→] [→] [→] [→] [→] [→] [→] [→] [↓]
[S] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [G]
```

**Optimal strategy (13 steps):** go up 1 cell, walk 11 cells to the right along the row just above the cliff, then go down 1 cell into the goal. It is the shortest possible path, so the total reward is -13.

### Optimal trajectory animations

| Value Iteration | Policy Iteration |
|---|---|
| ![Value Iteration](results/value_iteration.gif) | ![Policy Iteration](results/policy_iteration.gif) |

## Project structure

```
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- notebooks/
|   `-- TP_ai_gym_CliffWalking.ipynb    # Full implementation and experiments
|-- results/
|   |-- comparison_cliffwalking.png     # Comparison of the two algorithms
|   |-- value_iteration.gif             # Optimal trajectory (Value Iteration)
|   `-- policy_iteration.gif            # Optimal trajectory (Policy Iteration)
`-- docs/
    `-- TP_CliffWalking_rapport.pdf     # Lab report (in French)
```

## Notebook content

`notebooks/TP_ai_gym_CliffWalking.ipynb` contains, in order:

1. Imports and environment creation
2. Value Iteration implementation
3. Policy Evaluation and Policy Iteration implementation
4. Execution and timing of both algorithms
5. Policy evaluation by simulation (100 episodes)
6. Text visualization of the policies
7. Comparison plots
8. GIF generation of the optimal trajectories

## How to run

```bash
python -m venv .venv
source .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/TP_ai_gym_CliffWalking.ipynb
```

Run all cells in order. The notebook saves the comparison plot and the GIFs in its working directory.

## Conclusion

Both algorithms produce exactly the same optimal 13-step path with a 100% success rate. In this small deterministic environment, Value Iteration is the more efficient choice.

## Documents

- [Lab report (PDF, French)](docs/TP_CliffWalking_rapport.pdf)
