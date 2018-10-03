PSE-MPI
================

PSE-MPI is an MPI usage anomaly detection tool that employs partial symbolic execution.
It is built on top of the KLEE symbolic virtual machine.

## Building

LLVM 3.6 and MPICH are required.

PSE-MPI can only be built with CMake.  Following is an example of the `cmake` command.

```
cmake -DENABLE_SOLVER_Z3=ON -DCMAKE_PREFIX_PATH=/path/to/z3 \
      -DENABLE_UNIT_TESTS=OFF -DENABLE_SYSTEM_TESTS=OFF \
      -DENABLE_DOCS=OFF -DENABLE_DOXYGEN=OFF \
      -DUSE_CMAKE_FIND_PACKAGE_LLVM=OFF -DLLVM_CONFIG_BINARY=/path/to/llvm-config \
      -DUSE_CXX11=ON -DMPI_HEADER_PATH=/path/to/mpich/include \
      /path/to/pse-mpi
```
For more information, please refer to [the building instructions of KLEE](http://klee.github.io/).

## Usage

1. Compile the target MPI program into a single bitcode file `target.bc`.
2. Run the `klee` command.
    ```
    klee -np <nprocs> -entry-function <function_name> target.bc
    ```
    To find more options, please run `klee -help`.
