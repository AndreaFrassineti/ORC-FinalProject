# FINAL PROJECT -- Frassineti/PaisBianco
# Learning an Approximate Control-Invariant Set for MPC

## Overview
This project explores the concept of learning an approximate control-invariant set to be used as a **terminal constraint** in a **Model Predictive Control (MPC)** formulation. The goal is to ensure recursive feasibility in an MPC problem by leveraging a learned **N-step backward reachable set**.

## Problem Definition
We aim to determine whether an initial state $x_{init}$ belongs to the N-step backward reachable set of $S$, where:
- $S$ is the set of equilibrium states:
  $$S = \{ x \in X : \exists u \in U, x = f(x,u) \}$$
  For a **robot manipulator**, this typically corresponds to **zero-velocity states**:
  $$S = \{ (q, \dot{q}) : q_{min} \leq q \leq q_{max}, \dot{q} = 0 \}$$

To check if a state $x_{init}$ belongs to the backward reachable set, we solve the following **Optimal Control Problem (OCP)**:

$\min_{X,U} \space \space \space 1$

**subject to:**
- $x_{i+1} = f(x_i, u_i), \quad i = 0, \dots, N-1$
- $x_{i+1} \in X, \quad u_i \in U, \quad i = 0, \dots, N-1$
- $x_0 = x_{\text{init}}$
- $x_N \in S$



If the OCP has a solution, $x_{init}$ belongs to the set; otherwise, it does not.

## Data Generation & Neural Network Training
To approximate the backward reachable set:
1. **Generate a dataset** by sampling random $x_{init}$ values and solving the OCP.
2. **Label each state** as 0 (outside the set) or 1 (inside the set).
3. **Train a neural network** to classify states as inside or outside the set.

## MPC
MPC is an optimization-based control strategy that computes control inputs by solving a constrained optimization problem over a finite horizon at each time step. It ensures that the system respects constraints while optimizing a performance criterion.

In this project, we incorporate the learned neural network into the MPC **terminal constraint** to improve recursive feasibility. This ensures that the system remains within a feasible region where a valid control sequence exists.

## Implementation 
- **System:** Single pendulum → Double pendulum
- **Cost function:** Encourages reaching a joint configuration near limits while **minimally penalizing** velocities and torques.
- **Constraints:** Enforced using the learned invariant set.

## Results
All the result obtained are explained in the report.

