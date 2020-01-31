# Matrix linear solvers

The classes implement multiple methods to solve the linear system **Ax = b**, where **A** is a positive definite matrix, whereas **x** and **b** are vectors.

#### Solvers for *dense* matrix implemented:

1.  Jacobi method
2.  Gauss-Seidel method
3.  Gaussian elimination method
4.  LU decomposition method
5.  Inverse method
6.  Cholesky decomposition method
7.  Conjugate gradient method
8.  Gauss-Jordan elimination method
9.  Cramer's Rule

#### Solvers for *sparse* matrix implemented:

1.  Jacobi method
2.  Gauss-Seidel method
3.  Conjugate gradient method
4.  Cholesky method

## Documentation

A short report on the assignment can be found here [Assignment_Report.pdf](Assignment_Report.pdf).

## Installation guide

To use the solver, simply download and load the following files ([Matrix.h](Matrix.h), [Matrix.cpp](Matrix.cpp), [CSRMatrix.h](CSRMatrix.h), [CSRMatrix.cpp](CSRMatrix.cpp) and [Main_test_solver.cpp](Main_test_solver.cpp)) into a new project in Visual Studio to compile and run them.

## User instructions

```c++
Add codes here
```

## Testing

To test the solver, additionally download and load the [Test.h](Test.h) and [Test.cpp](Test.cpp) files that contain the class `Test` which will test the solvers for both dense and sparse matrix. To run the test, uncomment the following part in the [Main_test_solver.cpp](Main_test_solver.cpp) file.

```c++
Add codes here
```

