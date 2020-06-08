#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function for simple initialization of matrix elements
void DummyDataInitialization(double *pAMatrix, double *pBMatrix, int Size)
{
    int i, j; // Loop variables
    for (i = 0; i < Size; i++)
        for (j = 0; j < Size; j++)
        {
            pAMatrix[i * Size + j] = 1;
            pBMatrix[i * Size + j] = 1;
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

// Function for random initialization of matrix elements
void RandomDataInitialization(double *pAMatrix, double *pBMatrix,
                              int Size)
{
    int i, j; // Loop variables
    srand(23);
    for (i = 0; i < Size; i++)
        for (j = 0; j < Size; j++)
        {
            pAMatrix[i * Size + j] = rand() / 1000.0;
            pBMatrix[i * Size + j] = rand() / 1000.0;
        }
}

// Function for memory allocation and initialization of matrix elements
void ProcessInitialization(double **pAMatrix, double **pBMatrix,
                           double **pCMatrix, int Size)
{
    // Memory allocation
    *pAMatrix = (double *)malloc(Size * Size * sizeof(double));
    *pBMatrix = (double *)malloc(Size * Size * sizeof(double));
    *pCMatrix = (double *)malloc(Size * Size * sizeof(double));
    // Initialization of matrix elements
    DummyDataInitialization(*pAMatrix, *pBMatrix, Size);

    ZeroMatrix(*pCMatrix, Size);

}

// Function for formatted matrix output
void PrintMatrix(double *pMatrix, int RowCount, int ColCount)
{
    int i, j; // Loop variables
    for (i = 0; i < RowCount; i++)
    {
        for (j = 0; j < ColCount; j++)
            printf("%7.4f ", pMatrix[i * RowCount + j]);
        printf("\n");
    }
}

// Function for matrix multiplication
void SerialResultCalculation(double *pAMatrix, double *pBMatrix,
                             double *pCMatrix, int Size)
{
    int i, j, k; // Loop variables
    for (i = 0; i < Size; i++)
    {
        for (j = 0; j < Size; j++)
            for (k = 0; k < Size; k++)
                pCMatrix[i * Size + j] += pAMatrix[i * Size + k] * pBMatrix[k * Size + j];
    }
}


// Function for computational process termination
void ProcessTermination(double *pAMatrix, double *pBMatrix,
                        double *pCMatrix)
{
    free(pAMatrix);
    free(pBMatrix);
    free(pCMatrix);
}

void main()
{
    double *pAMatrix; //First argument of matrix multiplication
    double *pBMatrix; //Second argument of matrix multiplication
    double *pCMatrix; //Result matrix
    int Size; //Size of matrices
    
    time_t start, finish;
    double duration;
           
    printf("Serial matrix multiplication program\n");

    // Setting the size of matrices
    do
    {
        printf("\nEnter the size of matrices: ");
        scanf("%d", &Size);
        printf("\nChosen matrices' size = %d\n", Size);
        if (Size <= 0)
            printf("\nSize of objects must be greater than 0!\n");
    } while (Size <= 0);

    // Memory allocation and initialization of matrix elements
    ProcessInitialization(&pAMatrix, &pBMatrix, &pCMatrix, Size);
    // Matrix output
    printf("Initial A Matrix \n");
    PrintMatrix(pAMatrix, Size, Size);
    printf("Initial B Matrix \n");
    PrintMatrix(pBMatrix, Size, Size);

    // Matrix multiplication
    start = clock();
    SerialResultCalculation(pAMatrix, pBMatrix, pCMatrix, Size);
    finish = clock();

    duration = (finish - start) / (double)CLOCKS_PER_SEC;
    // Printing the result matrix
    printf("\n Result Matrix: \n");
    PrintMatrix(pCMatrix, Size, Size);
    // Printing the time spent by matrix multiplication
    printf("\n Time of execution: %f\n", duration);
    // Computational process termination
    ProcessTermination(pAMatrix, pBMatrix, pCMatrix);
}