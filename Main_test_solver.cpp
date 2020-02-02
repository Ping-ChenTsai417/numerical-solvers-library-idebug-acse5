#include <iostream>
#include <stdlib.h>    
#include <time.h> 
#include <string>
#include "Matrix.h"
#include "Matrix.cpp"
#include "CSRMatrix.h"
#include "CSRMatrix.cpp"
#include "Test.h"
#include "Test.cpp"
#include "BandedMatrix.h"
#include "BandedMatrix.cpp"

int main()

{

	////////THIS PART OF CODE GENERATES 50x50 DENSE MATRIX M AND VECTOR B AND THEN SOLVES THE EQUATION M*x=b
///===============================================================================================================
	///We will create a random dense matrix
	Test<double> TestMatrix;
	//matrix size will be 50x50
	TestMatrix.rows = 50;
	TestMatrix.cols = 50;
	//the values on the diagonal will be between 20 and 40, all other values will be between 0 and 1
	//the matrix will be symmetric
	TestMatrix.DIAG_MAX = 40;
	TestMatrix.DIAG_MIN = 20;
	//create a matrix object
	auto* DenseMatrix = new Matrix   <double>(TestMatrix.rows, TestMatrix.cols, true);
	//use functon from tests library to fill DenseMatrix with random values, described above
	TestMatrix.Generate_Dense_Matrix(*DenseMatrix);
	//create vector B
	auto* Vector_B = new Matrix   <double>(TestMatrix.rows, 1, true);
	//use functon from tests library to fill Vector_B with random values
	TestMatrix.Generate_Random_Vector(*Vector_B);
	//create vector to write out the result
	auto* Vector_X = new Matrix   <double>(TestMatrix.rows, 1, true);
	//use functon from tests library to fill with zeros for now
	TestMatrix.Initialize_Empty_Vector(*Vector_X);
    
	//call the solver method on dense matrix to solve the equation M*x=b
	//here we use LU solver, there are many others avaiable,just change the last parameter
	//to one of: Jacobi, Gauss_Siedel, Gaussian, LU, Inverse, Cholesky, Conjugate_Gradient, Gauss_Jordan,Cramers
	DenseMatrix->solver(*Vector_B, *Vector_X, Gauss_Jordan);

	//now we will do some tests to check if the result is correct

	//create a new empty vector
	auto* Test_Vector = new Matrix   <double>(TestMatrix.rows, 1, true);

	//now multipy calculated x_vector and original matrix
	//it should equal  to original b vector (to a given tolerance)
	DenseMatrix->matVecMult(*Vector_X, *Test_Vector);

	std::cout << "Lets check what is the difference orignal B vector and calculated_B_vector: \n";
	double difference = TestMatrix.Compare_Two_Matrices(*Test_Vector, *Vector_B);
	
	delete DenseMatrix;
	delete Vector_B;
	delete Vector_X;
	delete Test_Vector;
///===============================================================================================================
	///////////Example usage of solver for used-defined matrices
	////////////////////////////////////////////////////////////
	//Uncomment block below to see example user usage
	/*
	int rows = 3;
	int cols = 3;
	//define matrix
	auto* mat = new Matrix<double>(rows, cols, true);

	//an example positive definite matrix
	double mat_val[9]{ 4,-1,1,-1,2,-1,1,-1,4 };
	//example of vector b
	double vec_val[3]{ 1,2,1 };

	//write out the values onto the matrix
	for (int i = 0; i < rows * cols; i++)
	{
		mat->values[i] = mat_val[i];
	}

	//define vecotr b
	auto* vector_b = new Matrix<double>(rows, 1, true);
	//define vectors to store output for two different solvers
	auto* vector_out = new Matrix<double>(rows, 1, true);
	auto* vector_out2 = new Matrix<double>(rows, 1, true);
	//write the values to vector_b
	//and fill vectors_output with zeros
	for (int j = 0; j < rows; j++)
	{
		vector_b->values[j] = vec_val[j];
		vector_out->values[j] = 0;
		vector_out2->values[j] = 0;
	}
	//call the solvers 
	mat->solver(*vector_b, *vector_out, Gauss_Siedel);
	mat->solver(*vector_b, *vector_out2, LU);
	//print the resulting vectors_output
	std::cout << "Now printing result values from Gauss-Seidel:" << std::endl;
	vector_out->printMatrix();
	std::cout << "\nNow printing result values from LU:" << std::endl;
	vector_out2->printMatrix();

	delete mat;
	delete vector_b;
	delete vector_out;
	delete vector_out2;
	*/
///===============================================================================================================
	///////////UNCOMMENT BLOCK OF CODE BELOW TO PERFORM TESTS ON DENSE SOLVERS
	/*
    Test<double> General_Solver;
    std::cout << "Test for Dense solvers:\n";
    General_Solver.Run_Test(verb_1, General_Solver_Test,All_Dense);
	*/
///===============================================================================================================
	/////////UNCOMMENT BLOCK OF CODE BELOW TO PERFORM TESTS ON SPARSE SOLVERS
	/*
    Test<double> Sparse_Solver;
	std::cout << "Test for Sparse solvers:\n";
    Sparse_Solver.Run_Test(verb_1, CSR_Solver_Test, All_Sparse);
	*/
///===============================================================================================================
	////////UNCOMMENT BLOCK OF CODE BELOW TO PERFORM TIMING TESTS ON SPARSE MATRICES
	////////RESULTS WILL BE WRITTEN TO THE FILE
	////////the matrices will be originally generated as SPARSE CSR matrices
	////////and then converted to normal sparse matrices for dense solvers
	//////tests will be performed on matrices size 10x10 to 200x200
	/*
	Test<double> Timing_Test_Sparse;
	Timing_Test_Sparse.Time_Solvers_Sparse();
	*/
///===============================================================================================================
	////////UNCOMMENT BLOCK OF CODE BELOW TO PERFORM TIMING TESTS ON DENSE MATRICES
	////////RESULTS WILL BE WRITTEN TO THE FILE
	///////the matrices will be originally generated as DENSE CSR matrices
	///////and then converted to normal dense matrices for dense solvers
	//////tests will be performed on matrices size 10x10 to 200x200
	/*
	Test<double> Test_Timing_Dense;
	Test_Timing_Dense.Time_Solvers_Dense();
	*/
///===============================================================================================================

//==========================================================================================================
	////////Banded Class Example Output
	////////PLEASE NOTE THAT THIS CLASS IS NOT FINISHED
	///////it was included here to show that the effort has been made to include banded matrix class in our library
	///////UNCOMMENT BLOCK OF CODE BELOW TO USE IT
	/*
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
	cout << endl<< "Example output for Banded Class:" << endl;
	cout << endl << "============================Print Original Sparse Matrix============================" << endl;
	banded_mat->printBandedMatrix();

	// Do not run the following line , since Cuthill_Mckee() has bugs inside.
	// banded_mat->Run_Reverse_Cuthill_Mckee();


	// However, other functions works well

	// Get the degree from the adjacency matrix
	vector<double> degree = banded_mat->get_degree();
	cout << endl << "Print node degree: ";
	for (auto i : degree)
		cout << i << ' ';
	cout << endl;


	// Assume Cuthill_Mckee() works perfectly, then R is computed correcltly
	// See BandedMatrix.h for initialising vector R.   ----->    vector<double> R{ 0, 4, 2, 1, 5, 7, 3, 6 };
	banded_mat->reverse_R();
	banded_mat->permutate_to_Band();

	delete banded_mat;
	*/
//===============================================================================================================

}
