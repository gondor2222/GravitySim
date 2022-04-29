# GravitySim: Basic N-body simulation in Unity #

## About ##

GravitySim is a particle n-body simulation that simulates the mutual gravitational attraction between particles.
For more information on n-body simulations and the n-body problem see https://en.wikipedia.org/wiki/N-body_simulation
This specific implementation is capable of calculating using both the naive O(N^2) method (direct pair calculations),
or the O(N\*log(N)) Barnes-Hut method (summation over quad/octtree), both of which are calculated in parallel on the
CPU and GPU.

Currently the simulation is run in 2D, with 3D planned once the bugs are ironed out.

## Overview of main algorithm ##

The main Monobehaviour handles GPU-CPU synchronization as well as particle updates and changes to render parameters.
Every frame, a thread checks if a simulation frame is already being calculated. If one is, this thread returns and yields.

If no simulation thread is being calculated, the thread first checks if any particles have changed in mass (due to a collision).
The ones that have have their internal parameters such as radius, and temperature updated.

The thread then begins the main simulation frame, seting up input and output buffers to the compute shader as well as
data structures used to aid the computation (no extra structures for the naive method, but Barnes-Hut must construct the
quadtree).

Solution data consists of accelerations in all simulated dimensions, up to four identifiers for particles identified as being within
collision range, and some debug information.

This thread then dispatches a group of the lowest-indexed particles to be solved on the GPU. While waiting, the thread solves
backwards from the highest-indexed particles. Whenever the GPU responds, its results are accumulated into
the same solution data structure as the CPU's and the next group is immediately deployed. The calculations are complete when
the cpu's current target particle becomes less than the maximum particle already solved by the GPU.

Once the calculations are complete, the cpu steps through all the particles, updating their velocities using the computed accelerations.
It then steps through all the particles again, joining particles to their center of mass reference frame if they were identified as having
collided. Finally, it steps through all the particles a last time to update their positions using their velocities.

Several key positions in the CPU code check the time that has passed since the last frame every n loops, with the thread yielding
if too much time has passed. This transfers control temporarily back to the other threads handling renderering, mouse inputs, etc. and also
guarantees the CPU is able to respond to requests to close or pause the program.

If the number of living particles falls below half the size of the particle array, the array is reduced to half its size to reduce memory
consumption and slightly speed up the O(N) loops over it.

## Package Structure ##

Only the Assets folder is related to the simulation itself. The other folders contain assets used by Unity to set up and run
the project.

### Materials ###
The `Materials` folder contains textures and materials used to display various star surface textures
This folder is currently a WIP and the simulation uses the yellow dwarf texture for all objects

### Scenes ###

There is currently only one scene: The main scene, which does all simulation and visualizes the results

### Scripts ###

Contains CPU code to be executed by the simulation.
See documentation in the scripts themselves for more information. The Main and MainBarnesHut scripts are the Monobehaviours
which handle the main simulation logic, for the naive and Barnes-Hut methods respectively.

### Shaders ###

#### StarSurfaceShader ####

Translates a grayscale temperature variation texture and a base temperature parameter into the corresponding texture colored
according to black-body emission temperature.

#### UpdatePhysics ####

Contains two GPU kernels: one for simulating particle accelerations according to the naive method, and another for simulating
according to the Barnes-Hut method.

## Setup instructions ##

Clone the repository to a folder of your choice and then import the folder as a project in Unity

## Usage ##

-- WIP --