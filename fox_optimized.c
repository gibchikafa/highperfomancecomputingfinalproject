#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define SM (CLS / sizeof (double))

int ProcNum = 0;  //Number of available processes
int ProcRank = 0; //Rank of current process
int GridSize;     //Size of virtual processor grid

int GridCoords[2]; //Coordinates of current processor in grid
MPI_Comm GridComm; // Grid communicator
MPI_Comm ColComm;  //Column communicator
MPI_Comm RowComm;  //Row communicator

/// Function for simple initialization of matrix elements
void DummyDataInitialization(double *pAMatrix, double *pBMatrix, int Size)
{
    int i, j; // Loop variables
    for (i = 0; i < Size; i++)
        for (j = 0; j < Size; j++)
        {
            pAMatrix[i * Size + j] = 1.0;
            pBMatrix[i * Size + j] = 1.0;
        }
}

// Function for creating a zero matrix
void ZeroMatrix(double *m, int n)
{
    int row, col;
    for (row = 0; row < n; row++)
        for (col = 0; col < n; col++)
            m[row * n + col] = 0.0;
}

// Function for formatted matrix output
void PrintMatrix(double *pMatrix, int RowCount, int ColCount)
{
    int i, j; // Loop variables
    for (i = 0; i < RowCount; i++)
    {
        for (j = 0; j < ColCount; j++)
            printf("%7.4f ", pMatrix[i * ColCount + j]);
        printf("\n");
    }
}

// Function for matrix multiplication
void SerialResultCalculation(double *pAMatrix, double *pBMatrix,
                             double *pCMatrix, int Size)
{
    for (int i0 = 0; i0 < Size; i0 += SM)
    {
        int imax = i0 + SM > Size ? Size : i0 + SM;
        for (int j0 = 0; j0 < Size; j0 += SM)
        {
            int jmax = j0 + SM > Size ? Size : j0 + SM;
            for (int k0 = 0; k0 < Size; k0 += SM)
            {
                int kmax = k0 + SM > Size ? Size : k0 + SM;
                for (int j1 = j0; j1 < jmax; ++j1)
                {
                    int sj = Size * j1;
                    for (int i1 = i0; i1 < imax; ++i1)
                    {
                        int mi = Size * i1;
                        int ki = Size * i1;
                        int kij = ki + j1;

                        for (int k1 = k0; k1 < kmax; ++k1)
                        {
                            pCMatrix[kij] += pAMatrix[mi + k1] * pBMatrix[sj + k1];
                        }
                    }
                }
            }
        }
    }
}

// Function for block multiplication
void BlockMultiplication(double *pAblock, double *pBblock, double *pCblock, int Size)
{
    SerialResultCalculation(pAblock, pBblock, pCblock, Size);
}

// Function for creating the two-dimensional grid communicator
// and communicators for each row and each column of the grid
void CreateGridCommunicators()
{
    int DimSize[2];  // Number of processes in each dimension of the grid
    int Periodic[2]; // =1, if the grid dimension should be periodic
    int Subdims[2];  // =1, if the grid dimension should be fixed

    DimSize[0] = GridSize;
    DimSize[1] = GridSize;
    Periodic[0] = 0;
    Periodic[1] = 0;

    // Creation of the Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);
    // Determination of the cartesian coordinates for every process
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);
    // Creating communicators for rows
    Subdims[0] = 0; // Dimensionality fixing
    Subdims[1] = 1; // The presence of the given dimension in the subgrid
    MPI_Cart_sub(GridComm, Subdims, &RowComm);
    // Creating communicators for columns
    Subdims[0] = 1;
    Subdims[1] = 0;
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}

// Function for checkerboard matrix decomposition
void CheckerboardMatrixScatter(double *pMatrix, double *pMatrixBlock,
                               int Size, int BlockSize)
{
    double *MatrixRow = (double *)malloc(BlockSize * Size * sizeof(double));
    if (GridCoords[1] == 0)
    {
        MPI_Scatter(pMatrix, BlockSize * Size, MPI_DOUBLE, MatrixRow,
                    BlockSize * Size, MPI_DOUBLE, 0, ColComm);
    }
    for (int i = 0; i < BlockSize; i++)
    {
        MPI_Scatter(&MatrixRow[i * Size], BlockSize, MPI_DOUBLE,
                    &(pMatrixBlock[i * BlockSize]), BlockSize, MPI_DOUBLE, 0, RowComm);
    }
    free(MatrixRow);
}

// Data distribution among the processes
void DataDistribution(double *pAMatrix, double *pBMatrix, double *pMatrixAblock, double *pBblock, int Size, int BlockSize)
{
    // Scatter the matrix among the processes of the first grid column
    CheckerboardMatrixScatter(pAMatrix, pMatrixAblock, Size, BlockSize);
    CheckerboardMatrixScatter(pBMatrix, pBblock, Size, BlockSize);
}

// Function for gathering the result matrix
void ResultCollection(double *pCMatrix, double *pCblock, int Size,
                      int BlockSize)
{
    double *pResultRow = (double *)malloc(Size * BlockSize * sizeof(double));
    for (int i = 0; i < BlockSize; i++)
    {
        MPI_Gather(&pCblock[i * BlockSize], BlockSize, MPI_DOUBLE,
                   &pResultRow[i * Size], BlockSize, MPI_DOUBLE, 0, RowComm);
    }
    if (GridCoords[1] == 0)
    {
        MPI_Gather(pResultRow, BlockSize * Size, MPI_DOUBLE, pCMatrix,
                   BlockSize * Size, MPI_DOUBLE, 0, ColComm);
    }
    free(pResultRow);
}

// Broadcasting blocks of the matrix A to process grid rows
void ABlockCommunication(int iter, double *pAblock, double *pMatrixAblock,
                         int BlockSize)
{
    // Defining the leading process of the process grid row
    int Pivot = (GridCoords[0] + iter) % GridSize;
    // Copying the transmitted block in a separate memory buffer
    if (GridCoords[1] == Pivot)
    {
        for (int i = 0; i < BlockSize * BlockSize; i++)
            pAblock[i] = pMatrixAblock[i];
    }
    // Block broadcasting
    MPI_Bcast(pAblock, BlockSize * BlockSize, MPI_DOUBLE, Pivot, RowComm);
}

// Function for cyclic shifting the blocks of the matrix B
void BblockCommunication(double *pBblock, int BlockSize)
{
    MPI_Status Status;
    int NextProc = GridCoords[0] + 1;
    if (GridCoords[0] == GridSize - 1)
        NextProc = 0;
    int PrevProc = GridCoords[0] - 1;
    if (GridCoords[0] == 0)
        PrevProc = GridSize - 1;
    MPI_Sendrecv_replace(pBblock, BlockSize * BlockSize, MPI_DOUBLE,
                         NextProc, 0, PrevProc, 0, ColComm, &Status);
}

// Function for parallel execution of the Fox method
void ParallelResultCalculation(double *pAblock, double *pMatrixAblock,
                               double *pBblock, double *pCblock, int BlockSize)
{
    for (int iter = 0; iter < GridSize; iter++)
    {
        // Sending blocks of matrix A to the process grid rows
        ABlockCommunication(iter, pAblock, pMatrixAblock, BlockSize);
        // Block multiplication
        BlockMultiplication(pAblock, pBblock, pCblock, BlockSize);
        // Cyclic shift of blocks of matrix B in process grid columns
        BblockCommunication(pBblock, BlockSize);
    }
}

// Test printing of the matrix block
void TestBlocks(double *pBlock, int BlockSize, char str[])
{
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0)
    {
        printf("%s \n", str);
    }
    for (int i = 0; i < ProcNum; i++)
    {
        if (ProcRank == i)
        {
            printf("ProcRank = %d \n", ProcRank);
            PrintMatrix(pBlock, BlockSize, BlockSize);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Function for testing the matrix multiplication result
void TestResult(double *pAMatrix, double *pBMatrix, double *pCMatrix,
                int Size)
{
    double *pSerialResult;
    // Result matrix of serial multiplication
    double Accuracy = 1.e-6;
    // Comparison accuracy
    int equal = 0;
    // =1, if the matrices are not equal
    int i;
    // Loop variable
    if (ProcRank == 0)
    {
        pSerialResult = (double *)malloc(Size * Size * sizeof(double));
        for (i = 0; i < Size * Size; i++)
        {
            pSerialResult[i] = 0;
        }
        BlockMultiplication(pAMatrix, pBMatrix, pSerialResult, Size);
        for (i = 0; i < Size * Size; i++)
        {
            if (fabs(pSerialResult[i] - pCMatrix[i]) >= Accuracy)
                equal = 1;
        }
        if (equal == 1)
            printf("The results of serial and parallel algorithms are NOT"
                   "identical. Check your code.\n");
        else
            printf("The results of serial and parallel algorithms are "
                   "identical. \n");
    }
}

// Function for computational process termination
void ProcessTermination(double *pAMatrix, double *pBMatrix,
                        double *pCMatrix, double *pAblock, double *pBblock, double *pCblock,
                        double *pMatrixAblock)
{
    if (ProcRank == 0)
    {
        free(pAMatrix);
        free(pBMatrix);
        free(pCMatrix);
    }

    free(pAblock);
    free(pBblock);
    free(pCblock);
    free(pMatrixAblock);
}

// Function for memory allocation and data initialization
void ProcessInitialization(double **pAMatrix, double **pBMatrix,
                           double **pCMatrix, double **pAblock, double **pBblock, double **pCblock,
                           double **pTemporaryAblock, int *Size, int BlockSize)
{
    //MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //int bSize = *BlockSize
    int pABlockSize = BlockSize * BlockSize;
    *pAblock = (double *)malloc(pABlockSize * sizeof(double));
    *pBblock = (double *)malloc(pABlockSize * sizeof(double));
    *pCblock = (double *)malloc(pABlockSize * sizeof(double));
    *pTemporaryAblock = (double *)malloc(pABlockSize * sizeof(double));

    ZeroMatrix(*pCblock, BlockSize);

    int pMatrixSize = *Size * *Size;

    if (ProcRank == 0)
    {
        *pAMatrix = (double *)malloc(pMatrixSize * sizeof(double));
        *pBMatrix = (double *)malloc(pMatrixSize * sizeof(double));
        *pCMatrix = (double *)malloc(pMatrixSize * sizeof(double));
        DummyDataInitialization(*pAMatrix, *pBMatrix, *Size);
        //RandomDataInitialization(pAMatrix, pBMatrix, Size);
    }
}

int main(int argc, char *argv[])
{
    double *pAMatrix; // First argument of matrix multiplication
    double *pBMatrix; // Second argument of matrix multiplication
    double *pCMatrix; // Result matrix

    int Size = atoi(argv[1]); // Size of matrices

    int BlockSize; // Sizes of matrix blocks

    double *pAblock; // Initial block of matrix A

    double *pBblock; // Initial block of matrix B

    double *pCblock; // Block of result matrix C

    double *pMatrixAblock;
    double Start, Finish, Duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    MPI_Barrier(MPI_COMM_WORLD);

    GridSize = sqrt((double)ProcNum);

    if (ProcNum != GridSize * GridSize)
    {
        if (ProcRank == 0)
        {
            printf("Number of processes must be a perfect square \n");
        }
    }
    else if (Size % GridSize != 0 && Size != 0)
    {
        if (ProcRank == 0)
        {
            printf("Size of matrices must be divisible by the grid size!\n");
        }
    }
    else
    {
        BlockSize = Size / GridSize;
        if (ProcRank == 0)
        {
            //printf("Number of processes is %d \n", ProcNum);
            //printf("Matrix Size is %d \n", Size);
            //printf("GridSize is %d \n", GridSize);
            //printf("Parallel matrix multiplication program\n");
        }

    
        // Creating the cartesian grid, row and column communcators
        CreateGridCommunicators();

        // Memory allocation and initialization of matrix elements
        ProcessInitialization(&pAMatrix, &pBMatrix, &pCMatrix, &pAblock, &pBblock, &pCblock, &pMatrixAblock, &Size, BlockSize);

        Start = MPI_Wtime();

        DataDistribution(pAMatrix, pBMatrix, pMatrixAblock, pBblock, Size, BlockSize);

        //TestBlocks(pMatrixAblock, BlockSize, "Initial blocks of matrix A");
        //TestBlocks(pBblock, BlockSize, "Initial blocks of matrix B");

        // Execution of the Fox method
        ParallelResultCalculation(pAblock, pMatrixAblock, pBblock, pCblock, BlockSize);

        // Gathering the result matrix
        ResultCollection(pCMatrix, pCblock, Size, BlockSize);
        Finish = MPI_Wtime();
        Duration = Finish - Start;
        //TestResult(pAMatrix, pBMatrix, pCMatrix, Size);

        // Process Termination
        ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock, pCblock, pMatrixAblock);
    }

    if (ProcRank == 0)
    {
        //printf("SM is %zu\n", SM);
        printf("The time is %f\n", Duration);
    }

    MPI_Finalize();

    return 0;
}
