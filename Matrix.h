#pragma once
#include <string>

template <class T>
class Matrix
{
public:

    // Member variables
    T* values = nullptr;
    int rows = -1;
    int cols = -1;

   // Constructor where we want to preallocate memory ourselves
   Matrix(int rows, int cols, bool preallocate);

   // Constructor where we already have allocated memory outside
   Matrix(int rows, int cols, T *values_ptr);
   
   // Destructor
   virtual ~Matrix();

   // Print out the values in our matrix
   void printValues();
   virtual void printMatrix();

   // Perform some operations with our matrix
   void matMatMult(Matrix<T>& mat_right, Matrix<T>& output);
   virtual void matVecMult(Matrix<T>& mat_right, Matrix<T>& output);
   void matVecMult(T* vect_in, T* vect_out);
   double innerProduct(Matrix<T>& vec1, Matrix<T>& vec2);

   // Calculate the matrix inverse
   void inverse(Matrix& inverse_mat);

   // Main solver method which calls different type of solver methods 
   virtual void solve(const Matrix<T>& vect_b,  Matrix<T>& vect_output, int type_of_solver);

   // Different methods of solvers to solve Ax = b
   void Jacobi_Solver(const Matrix<T>& vect_b,  Matrix<T>& vect_output);
   void Gauss_Siedel_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);
   void Gaussian_Solver(const Matrix<T>& vect_b,Matrix<T>& vect_output);
   void LU_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);
   void Inverse_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);
   void Cholesky_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);
   void Conjugate_Gradient_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);
   void Gauss_Jordan_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output);


   // Additional functions for the solvers
   void decompose_LU(Matrix* upper, Matrix* lower, Matrix* permut);
   void forward_substitution(Matrix* lower, T* vect_in, T* vect_out);
   void back_substitution(Matrix* upper, T* vect_in, T* vect_out);

   // Swap the rows within the matrix (mainly for partial pivoting in LU)
   void swap_rows(Matrix& matrix, int current_row, int max_row);

   // vector operation for conjugate gradient method
   void CG_CombineVector(T gradient, double alpha, const Matrix<T>& vect_b);

// We want our subclass to know about this
protected:
   bool preallocated = false;

   double tol = 0.000000000001; // tolerance for conjugate gradient
   const double near_zero = 0.00000001; // almost equivalent to 0, use it to avoid scalar divided by 0

// Private variables - there is no need for other classes 
// to know about these variables
private:
   int size_of_values = -1;   
};

//function to check convergence, needed for Gauss-Seidel method
template <class T>
bool check_error(Matrix<T>& mat, Matrix<T>& vect, Matrix<T>& vect_output);

enum solver_method_dense { Jacobi, Gauss_Siedel, Gaussian, LU, Inverse, Cholesky, Conjugate_Gradient, Gauss_Jordan, Last_Dense, All_Dense};
enum solver_method_csr   { Jacobi_CSR, Gauss_Siedel_CSR, Cholesky_CSR, Conjugate_Gradient_CSR, Last_Sparse, All_Sparse};
