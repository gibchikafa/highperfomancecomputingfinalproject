# Distributed Parallel Matrix-Matrix Multiply with Fox Algorithm
This project is a parallel implementation of the Fox Algorithm using MPI in C. They are two files: fox.c and fox_optimized. The fox.c uses a serial implementation of matrix multiply. In the fox_optimized.c we optimize the matrix-matrix multiply loops to utilize temporal and spatial locality.

#Compiling Code
The expiriment was conducetd on KTH Beskow supercomputer. To compile the fox.c with the gcc compiler on  Beskow use command: cc fox.c -o fox.out. On Linux computer use mpicc fox.c -o fox.out. To compile the fox_optimized.c on Beskow use command: mpicc -DCLS=$(getconf LEVEL1 DCACHE LINESIZE) fox_optimized.c -o fox_optimized.out. The -DCLS=$(getconf LEVEL1_DCACHE_LINESIZE) allows us to get the L1 cache size.

#Input matrices and running the code
In our implementation we assume that the number of processes p is a perfect square i.e 4,9, 16, 25 e.t.c. Let q = √p. If A dimension is m × k, and matrix B dimension is k × n, the sizes of the matrices should be such that q divides m, k and n so that we are able to create compatible partitions and the blocks of the partition are of the same size. By having all the partitions of the same size we have equal load on all the processes in the grid. For simplicity we will assume that both A and B are square matrices of the same size. To run the code on Beskow use the following command: srun -n <numOfProcess> ./fox.out <matrixSize>.
