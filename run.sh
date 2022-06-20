#!/bin/bash
#SBATCH -J ex   # Job name
#SBATCH -o mass.out     # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1        # Total number of nodes requested
#SBATCH -n 1         # Total number of mpi tasks requested
#SBATCH -t 1-20:00:00      # Run time (hh:mm:ss) -  1 day-20 hours
# Launch MPI-based executable
prun ipython helloworld.py      # want to run this python code
echo "Done"