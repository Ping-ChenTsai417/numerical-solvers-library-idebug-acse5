# Matrix linear solvers

The classes implement multiple methods to solve the linear system **Ax = b**, where **A** is a positive definite matrix, whereas **x** and **b** are vectors.

#### Solvers for *dense* matrix implemented:

1.  Jacobi method
2.  Gauss-Siedel method
3.  Gaussian elimination method
4.  LU decomposition method
5.  Inverse method
6.  Cholesky decomposition method
7.  Conjugate gradient method
8.  Gauss-Jordan elimination method
9.  Cramer's Rule

#### Solvers for *sparse* matrix implemented:

1.  Jacobi method
2.  Gauss-Siedel method
3.  Conjugate gradient method
4.  Cholesky method

## Documentation

A short report on the assignment can be read here [Assignment_Report.pdf](Assignment_Report.pdf).

## Installation guide

To use the solver, simply download and load the following files ([Matrix.h](Matrix.h), [Matrix.cpp](Matrix.cpp), [CSRMatrix.h](CSRMatrix.h), [CSRMatrix.cpp](CSRMatrix.cpp) and [Main_test_solver.cpp](Main_test_solver.cpp)) into a new project in Visual Studio to compile and run them.

The [Main_test_solver.cpp](Main_test_solver.cpp) contains the main method which are seperated into blocks of codes that can solve the linear system for a randomly-generated or a user-defined input, test the linear solvers and run timing tests. 

## User instructions

### (a) For dense solvers (3 x 3 matrix)

Example usage:

```c++
#include"Matrix.h"
#include"Matrix.cpp"

using namespace std;

void main()
{
    int rows = 3;
    int cols = 3;

    // fill in values for mat_A_dense and vect_b
    double A[9] = { 4,-1,1,-1,2,-1,1,-1,4 };
    double b[3] = { 1,2,1 };

    // create pointers to objects of type Matrix
    auto* mat_A_dense = new Matrix<double>(rows, cols, A);
    auto* vect_b = new Matrix<double>(rows, 1, b);
    // vect_out will be initialised to zero in the solver method
    auto* vect_output = new Matrix<double>(rows, 1, true);

    // solving mat_A_dense * vect_output = vect_b using dense Gauss-Siedel method
    mat_A_dense->solver(*vect_b, *vect_output, Gauss_Siedel);

    // print out solution to the screen
    vect_output->printValues();

    // deallocate the memory
    delete mat_A_dense;
    delete vect_b;
    delete vect_output;
}
```

Output:

```c++
Printing values
0.5  1.5  0.5
```

To use a different dense matrix solver, the user has to change the third input argument (which is an `enum` type) according to as follows:

1.  Jacobi method - `Jacobi`
2.  Gauss-Siedel method - `Gauss_Siedel`
3.  Gaussian elimination method - `Gaussian`
4.  LU decomposition method - `LU`
5.  Inverse method - `Inverse`
6.  Cholesky decomposition method - `Cholesky`
7.  Conjugate gradient method - `Conjugate_Gradient`
8.  Gauss-Jordan elimination method - `Gauss_Jordan`
9.  Cramer's Rule - `Cramers`

### (b) For sparse solvers (3 x 3 matrix)

Example usage:

```c++
#include"Matrix.h"
#include"Matrix.cpp"
#include"CSRMatrix.h"
#include"CSRMatrix.cpp"

using namespace std;

void main()
{
    int rows = 3;
    int cols = 3;
    int nnzs = 5;

    // fill in values for mat_A_sparse and vect_b
    double A[5] = { 3,4,1,1,2 };
    int row_position[4] = { 0,1,3,5 };
    int col_index[5] = { 0,1,2,1,2 };
    double b[3] = { 9,5,3 };

    // create pointers to objects of type CSRMatrix and Matrix
    auto* mat_A_sparse = new CSRMatrix<double>(rows, cols, nnzs, A, row_position, col_index);
    auto* vect_b = new Matrix<double>(rows, 1, b);
    // vect_out will be initialised to zero in the solver method
    auto* vect_output = new Matrix<double>(rows, 1, true);

    // solving mat_A_sparse * vect_output = vect_b using sparse Gauss-Siedel method
    mat_A_sparse->solver(*vect_b, *vect_output, Gauss_Siedel_CSR);

    // print out solution to the screen
    vect_output->printValues();

    // deallocate the memory
    delete mat_A_sparse;
    delete vect_b;
    delete vect_output;
} 
```

Output:

```c++
Printing values
3  1  1
```

To use a different sparse matrix solver, the user has to change the third input argument (which is an `enum` type) according to as follows:

1.  Jacobi method - `Jacobi_CSR`
2.  Gauss-Siedel method - `Gauss_Siedel_CSR`
3.  Conjugate gradient method - `Conjugate_Gradient_CSR`
4.  Cholesky method - `Cholesky_CSR`

A more detailed example on how to use the solver can be found in the [Main_test_solver.cpp](Main_test_solver.cpp) file. 


## Testing

To test the linear solvers, additionally download and load the [Test.h](Test.h) and [Test.cpp](Test.cpp) files. The class `Test` will test all the solver methods on multiple sizes of diagonally-dominant matrices, **A** and vector, **b** that are randomly generated. The tests are done by comparing the original vector **b**, to the one obtained by multiplying the matrix, **A** with the output solution, **x** for each method. 

To run the test, uncomment the following blocks in the [Main_test_solver.cpp](Main_test_solver.cpp) file.
 
 (a) **Test for dense solvers**
 
```c++
    Test<double> General_Solver;
    std::cout << "Test for Dense solvers:\n";
    General_Solver.Run_Test(verb_0, General_Solver_Test, All_Dense);
```

(b) **Test for sparse solvers**

```c++
    Test<double> Sparse_Solver;
    std::cout << "Test for Sparse solvers:\n";
    Sparse_Solver.Run_Test(verb_0, CSR_Solver_Test, All_Sparse);
```

The first input argument is an `enum` type which determines the amount of information printed on the screen when the tests are run. `verb_0` will only display whether the methods are solving the system correctly. Inputting `verb_1` will additionally show the value of error (sum of the absolute difference between corresponding elements of vectors **b**) for each method.

## Note: BandedMatrix

`BandedMatrix` class aims to permutate a sparse matrix into a banded matrix. The class contains the Reverse Cuthill-Mckee algorithm (`void Cuthill_Mckee(vector<T> degree_array)`) but still with a few bugs. Although the class is not yet ready for the user to  completely utilise it, the user may still be able to:

1.	Compute the degree of nodes for a sparse matrix
2.	Reverse the order of a permutation array
3.	Permutate a sparse matrix into a banded matrix using the correct permutation array, R

To check the example output, uncomment the following lines in the [Main_test_solver.cpp](Main_test_solver.cpp) file.
    
```c++
    //Banded Class
    int nnzs2 = 22;
    int rows2 = 8;
    int cols2 = 8;
    double mat_val2[22]{ 1, 3, 3, 4, 6, 6, 4, 8, 1, 1, 2, 3, 1, 5, 6, 7, 2, 2, 1, 6, 2, 4 };
    double IA2[9]{ 0, 2, 6, 9, 11, 14, 17, 19, 22 };
    double JA2[22]{ 0, 4, 1, 2, 5, 7, 1, 2, 4, 3, 6, 0, 2, 4, 1, 5, 7, 3, 6, 1, 5, 7 };

    auto* banded_mat = new BandedMatrix<double>(rows2, cols2, nnzs2, true);
    for (int i = 0; i < nnzs2; i++)
    {
        banded_mat->values[i] = mat_val2[i];
        banded_mat->col_index[i] = JA2[i];
    }
    for (int i = 0; i < rows2 + 1; i++)
    {
        banded_mat->row_position[i] = IA2[i];
    }
    cout << endl << "============================Print Original Sparse Matrix============================" << endl;
    banded_mat->printBandedMatrix();
    
    // Do not run the following line , since Cuthill_Mckee() has bugs inside.
    // banded_mat->Run_Reverse_Cuthill_Mckee();
    
    // However, other functions works well

    // Get the degree from the adjacency matrix
    vector<double> degree = banded_mat->get_degree();
    cout << endl << "Print node degree: " ;
    for (auto i : degree)
        cout << i << ' ';
    cout << endl;

    // Assume Cuthill_Mckee() works perfectly, then R is computed correcltly
    // See BandedMatrix.h for initialising vector R.   ----->    vector<double> R{ 0, 4, 2, 1, 5, 7, 3, 6 };
    banded_mat->reverse_R();
    banded_mat->permutate_to_Band();

    delete banded_mat;
```

