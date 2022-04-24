#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# install level-zero
$GITHUB_WORKSPACE/dev/install-level-zero-ubuntu.sh

sudo apt install  level-zero

#$GITHUB_WORKSPACE/dev/install-build-level-zero-deps-ubuntun.sh
$GITHUB_WORKSPACE/dev/install-level-zero-ubuntu.sh

# Install dependencies for building
$GITHUB_WORKSPACE/dev/install-build-deps-ubuntu.sh

# Setup building envs
source /opt/intel/oneapi/setvars.sh
#source /home/runner/work/level-zero/setvars.sh

<<<<<<< HEAD
cd  $GITHUB_WORKSPACE/dev/tools/list-compute-devices/
./build.sh
./run.sh
=======
$GITHUB_WORKSPACE/dev/tools/check-gpu-cpu/build.sh
$GITHUB_WORKSPACE/dev/tools/check-gpu-cpu/run.sh
>>>>>>> update

# Prepare lib resources
cd $GITHUB_WORKSPACE/mllib-da
../dev/prepare-build-deps-gpu.sh
./build.sh -p CPU_GPU_PROFILE

unset LD_LIBRARY_PATH
<<<<<<< HEAD
./test.sh -p CPU_GPU_PROFILE -q -d host
=======
./test.sh -p CPU_GPU_PROFILE -t cpu -q
>>>>>>> update ci
