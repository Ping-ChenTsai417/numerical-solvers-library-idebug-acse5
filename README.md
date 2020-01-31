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

A short report on the assignment can be found here [Assignment_Report.pdf](Assignment_Report.pdf).

## Installation guide

To use the solver, simply download and load the following files ([Matrix.h](Matrix.h), [Matrix.cpp](Matrix.cpp), [CSRMatrix.h](CSRMatrix.h), [CSRMatrix.cpp](CSRMatrix.cpp) and [Main_test_solver.cpp](Main_test_solver.cpp)) into a new project in Visual Studio to compile and run them.

## User instructions

Basic example usage:

 (a) **For dense solvers** (3 x 3 matrix)
 
```c++
#include"Matrix.h"
#include"Matrix.cpp"

using namespace std;

void main()
{
    int rows = 3;
    int cols = 3;
    
    // create pointers to objects of type Matrix
    auto* mat_A_dense = new Matrix<double>(rows, cols, true);
    auto* vect_b = new Matrix<double>(rows, 1, true);
    auto* vect_output = new Matrix<double>(rows, 1, true);
    
    // fill in values for mat_A_dense and vect_b....
    // vect_out will be initialised to zero in the solver method
       
    // solving mat_A_dense * vect_output = vect_b using Gauss-Siedel method
    mat_A_dense->solver(*vect_b, *vect_output, Gauss_Siedel);
    
    // print out solution to the screen
    vect_output->printValues();
    
    // deallocate the memory
    delete mat_A_dense;
    delete vect_b;
    delete vect_output;
}
```

To use a different solver method, the user needs to change the third input argument (which is an `enum` type) according to as follows:

1.  Jacobi method - `Jacobi`
2.  Gauss-Siedel method - `Gauss_Siedel`
3.  Gaussian elimination method - `Gaussian`
4.  LU decomposition method - `LU`
5.  Inverse method - `Inverse`
6.  Cholesky decomposition method - `Cholesky`
7.  Conjugate gradient method - `Conjugate_Gradient`
8.  Gauss-Jordan elimination method - `Gauss_Jordan`
9.  Cramer's Rule - `Cramers`

(b) **For sparse solvers** (3 x 3 matrix)

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
    
    // create pointers to objects of type Matrix and CSRMatrixs
    auto* mat_A_sparse = new CSRMatrix<double>(rows, cols, true);
    auto* vect_b = new Matrix<double>(rows, 1, true);
    auto* vect_output = new Matrix<double>(rows, 1, true);
    
    // fill in values for mat_A_sparse and vect_b....
    // vect_out will be initialised to zero in the solver method

    // solving mat_A_sparse * vect_output = vect_b using Gauss-Siedel method
    mat_A_sparse->solver(*vect_b, *vect_output, Gauss_Siedel_CSR);
    
    // print out solution to the screen
    vect_output->printValues();
    
    // deallocate the memory
    delete mat_A_sparse;
    delete vect_b;
    delete vect_output;
}   
```

To use a different solver method, the user needs to change the third input arguments (which is an `enum` type) according to as follows:

1.  Jacobi method - `Jacobi_CSR`
2.  Gauss-Siedel method - `Gauss_Siedel_CSR`
3.  Conjugate gradient method - `Conjugate_Gradient_CSR`
4.  Cholesky method - `Cholesky_CSR`

A more detailed example on how to use the solver can be found in the second block of the [Main_test_solver.cpp](Main_test_solver.cpp) file. 


## Testing

To test the linear solvers, additionally download and load the [Test.h](Test.h) and [Test.cpp](Test.cpp) files. The class `Test` will test all the solver methods on multiple sizes of diagonally-dominant matrices, **A** and vector **b** that are randomly generated. The tests are done by comparing the original vector **b**, to the one obtained by multiplying the matrix, **A** with the output solution, **x** for each method. 

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

The first input argument is an `enum` type which dicates the amount of information printed on the screen when the tests are run. `verb_0` will displays the minimum amount of information which is whether the method is solving the system correctly. Additionally, `verb_1` will also shows the discrepancy between the actual **b** and the **b** obtained from multiplying  **A** with calculated vector **x** for each solver method.
