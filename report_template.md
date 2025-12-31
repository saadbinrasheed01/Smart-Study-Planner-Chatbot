# Computational Intelligence Project Report

## Title
Smart Study Planner AI Agent using Genetic Algorithm (GA)

## Abstract
This project presents an AI agent that generates a weekly study timetable based on user-defined subjects, priorities, and constraints. The problem is formulated as a constrained optimization task. A Genetic Algorithm (GA) searches the space of possible schedules and maximizes a fitness function that rewards meeting target hours per subject while penalizing constraint violations (studying on off-days, exceeding daily limits, or too many consecutive hours of one subject). A Streamlit demo provides an interactive interface and visualizations.

## 1. Introduction
### 1.1 Problem Statement
Students struggle to distribute study time across multiple subjects while respecting daily limits and personal constraints. Manual planning is often inconsistent.

### 1.2 Objectives
- Collect subjects + priorities + constraints
- Use GA (Computational Intelligence) to optimize a weekly plan
- Provide an interactive demo and explainable outputs

### 1.3 Scope
Weekly planning with 1-hour slots. Can be extended to exam deadlines, 30-minute slots, and personalized analytics.

## 2. Methodology
### 2.1 Encoding (Chromosome)
- Week has 7 days; each day has slots_per_day slots.
- Chromosome: integer matrix A (7 x slots_per_day)
  - A[d, t] = subject index (0..S-1) or -1 for empty slot

### 2.2 Target Allocation
Targets are computed in two steps:
1) Allocate each subject's minimum weekly hours
2) Distribute remaining hours based on priority weights

### 2.3 Fitness Function
Fitness = Reward - Penalties

Reward:
- Minimize total deviation from targets
  reward = - sum_i |achieved_i - target_i|

Penalties:
- Off-day penalty if any study on off-days
- Daily limit penalty if day study hours exceed max
- Consecutive penalty if same subject exceeds max consecutive slots
- Imbalance penalty to discourage packing study into few days

### 2.4 GA Operators
- Selection: Tournament
- Crossover: Day-level uniform crossover
- Mutation: swap/change/toggle slots
- Elitism: top-k individuals copied to next generation

### 2.5 Architecture
1) UI collects inputs
2) GA computes targets and runs optimization
3) Best chromosome is converted to timetable
4) Charts show allocation and convergence

## 3. Implementation
- Python, Streamlit, NumPy, Pandas, Matplotlib

## 4. Results & Evaluation
### 4.1 Baseline
Compare GA vs random schedule generator.

### 4.2 Metrics
- Target deviation: sum |achieved - target|
- Constraint violations count
- Best fitness and convergence curve

### 4.3 Results (fill after running)
- GA reduced deviation by __% compared to random.
- Violations reduced to zero for off-days and daily limit.

## 5. Conclusion
GA successfully generates a feasible schedule and demonstrates evolutionary optimization and constraint handling.

## 6. Future Work
- Add exam deadlines (urgency)
- Multi-week planning
- LLM explanation module

## References
(Add any references you used here.)
