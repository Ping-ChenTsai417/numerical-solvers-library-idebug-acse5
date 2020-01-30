#pragma once
#include "Matrix.h"

template <class T>
class CSRMatrix: public Matrix<T>
{
public:

   // Member variables
   int* row_position = nullptr;
   int* col_index = nullptr;
   int nnzs = -1; // number of non-zero values in the matrix

   // Constructor where we want to preallocate memory ourselves
   CSRMatrix(int rows, int cols, int nnzs, bool preallocate);

   // Constructor where we already have allocated memory outside
   CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_position, int *col_index);

   // Destructor
   ~CSRMatrix();

   // Print out the values in our matrix
	virtual void printMatrix();

   // Multiply two sparse matrices
   void matMatMult(CSRMatrix<T>& mat_right, CSRMatrix<T>& output);

   // Multiply matrix and vector
   virtual void matVecMult(Matrix<T>& input, Matrix<T>& output);

   // Inserts a value to the sparse matrix, at the row i and column j
   void setvalues(T value, int i,int j);

   // if non-zero value is present returns an index in the array of values of the 
   // non-zero value at a given row and column, otherwise returns -1
   int Get_Value_Index(int row, int column);

   // Main solver method which calls different type of sparse solver methods
   // specify the type of solver by typing in an enum "type of solver"
   void solve(const Matrix<T>& vect_b, Matrix<T>& vect_output, int type_of_solver);
 
   // Different methods to solve Ax = b for sparse matrix
   void Jacobi_CSR_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);
   void Gauss_Seidel_CSR_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);
   void Conjugate_Gradient_CSR_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);
   void Cholesky_CSR_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);

   // Conversion between CSRMatrix and dense matrix
   void Convert_CSRMatrix_To_Matrix(Matrix<T>& M);
   void Convert_Matrix_To_CSR_Matrix(Matrix<T>& M);
   void Fast_CSR_Transpose(CSRMatrix<T>& Output);

};
template <class T>
bool check_error_CSR(CSRMatrix<T>& mat, Matrix<T>& vect, Matrix<T>& vect_output);
