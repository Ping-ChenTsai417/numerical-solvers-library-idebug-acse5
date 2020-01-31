#ifndef MY_MATRIX_TESTS

#define MY_MATRIX_TESTS

#include "Matrix.h"
#include "CSRMatrix.h"
#include <random>

template <class T>
class Test
{
public:
	void Generate_Sparse_CSR_Matrix(CSRMatrix<T>& M);
	void Generate_Dense_CSR_Matrix(CSRMatrix<T>& M);
	void Generate_Dense_Matrix(Matrix<T>& Matrix2);


	void Initialize_CSRMatrix(CSRMatrix<T>& M);
	void Initialize_Empty_Vector(Matrix<T>& Vector);
	void Generate_Random_Vector(Matrix<T>& Vector);
	double Compare_Two_Matrices(const Matrix<T>& Matrix_1, const Matrix<T>& Matrix_2);

	void Output_Matrix_To_File(Matrix<T>& M, std::string file_name);

	void Dense_Matrix_Solver_Test(int Solver_Type);
	void Sparse_Matrix_Solver_Test(int Solver_Type);

	bool Test_Dense_Solution(Matrix<T>& Vector_X, Matrix<T>& Vector_Actual_B, Matrix<T>& Matrix);
	bool Test_Sparse_Solution(Matrix<T>& Vector_X, Matrix<T>& Vector_Actual_B, CSRMatrix<T>& Matrix);
	void File_for_python(Matrix<T>& M, Matrix<T>& b, Matrix<T>& x, std::string file_name);

	void Solver_Timing_Test_All();
	double Test_Timing_Dense(Matrix<T>& Vector_Actual_B, Matrix<T>& M, int Solver);
	double Test_Timing_Sparse(Matrix<T>& Vector_Actual_B, CSRMatrix<T>& M, int Solver);
	void Time_Solvers_Dense();
	void Time_Solvers_Sparse();

	void Run_Test(int verbose, int test_index, int configuration);
	void Run_General_Solver(int configuration);
	void Run_Sparse_Solver(int configuration);

	
//Configuration
//Size of the array
	int rows = 5;
	int cols = 5;

//Diagonal random settings (random generation on diagonal in range DIAG_MIN to DIAG_MAX)
	T DIAG_MIN = 10;
	T DIAG_MAX = 20;

//Settings for non-diagonal elements (in order to guarantee matrix being positive definite it is good idea to have it being order of magnitude less than diagonal, and it has to be symmetrical)
	T GENERAL_MIN = 0;
	T GENERAL_MAX = 1;
	double percentage = 0.1;

	double accuracy = 0.1;
	int verbose = 0;
	bool Fatal = false;
	bool Fail = false;
	std::string Test_Number = "0";

};

enum test_enum { General_Solver_Test, CSR_Solver_Test, Timing_Test,Last_Test};
enum verbosity { verb_0,verb_1,verb_2,verb_3,verb_4 };
#endif // !MY_MATRIX_TESTS
