3D Galaxy Collision Simulation (N-body RK4)
Overview

This project simulates the gravitational interaction and collision of two disk galaxies using an N-body model.

Each galaxy consists of particles distributed in a rotating disk around a central massive core.

The system evolves under mutual gravitational interaction using a fourth-order Runge-Kutta integrator.

Physical Model

Self-gravitating particles

Two fixed galactic cores

Softened gravitational potential

Circular initial velocity profile

Velocity dispersion

Numerical Integration

Fourth-order Runge-Kutta (RK4)

Features

3D visualization

Two interacting galaxies

Real-time gravitational dynamics

Progress bar during rendering

Dependencies
numpy
matplotlib
tqdm


Install:

pip install -r requirements.txt


Run:

python main.py

Author

Esther Menéndez
Computational Physics & Simulation

## Simulation Preview

![Galaxy Simulation](galaxy_simulation.gif)

