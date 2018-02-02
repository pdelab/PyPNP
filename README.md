# PyPNP

## Overview

PyPNP is a solver for simulating charge-transport systems with an arbitrary number of charge-carrying species.
These systems are modeled by the Poisson-Nernst-Planck (PNP) equations with the possibility of coupling to the Navier-Stokes (NS) equation to simulate electrokinetic phenomena.

The solver uses continuous piecewise linear finite elements for the PNP system, and div-conforming (discontinuous) finite elements for the NS system.
The nonlinearities of the system are handled by implementing a monolithic Newton approach of the entire PNP (and NS) system.
The linearized systems are stabilized by an edge-averaged finite element (EAFE) approximation.

This package is based on the [FEniCS](https://fenicsproject.org/) package for discretizing the differential equations.
This permits the use of any compatible linear algebra solver for solving the resulting linear system of equations.


## Getting started

To ensure that the git hooks are properly being used (and creating a docker image for MacOS), run:
```
  ./scripts/init
```

When contributing, ensure code styling is consistent by running `./scripts/lint`.
This script is automatically run whenever a commit is made.
Use `git commit ... --no-verify` to bypass linting, although this practice is strongly discouraged.

### MacOS

The FEniCS distribution uses [Docker](https://www.docker.com/) for simplicity.
Install Docker by following [these instructions](https://docs.docker.com/docker-for-mac/install/).
To run the project in a docker container, run
```
  ./scripts/start
```
The Docker container will open in a shared directory to the repository's root so that any updates to files in the repository are available from within the Docker container.

To exit the Docker container, simply run the `exit` command.

### Linux systems

If FEniCS is not already installed or there are compatibility issues, follow the steps for MacOS to run FEniCS in a Docker container.
