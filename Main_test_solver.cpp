// Sparse_Matrix.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "CSRMatrix.h"
#include "CSRMatrix.cpp"
#include "Matrix.h"
#include "Matrix.cpp"
#include <stdlib.h>    
#include <time.h> 
#include <string>
#include "Test.h"
#include "Test.cpp"

int main()
{
	/////Check for memory leaks but on windows only
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    //srand(time(NULL));

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
	mat->solve(*vector_b, *vector_out, Gauss_Siedel);
	mat->solve(*vector_b, *vector_out2, LU);
	//print the resulting vectors_output
	std::cout << "Now printing result values from Gauss-Seidel:" << std::endl;
	vector_out->printMatrix();
	std::cout << "\nNow printing result values from LU:" << std::endl;
	vector_out2->printMatrix();

	delete mat;
	delete vector_b;
	delete vector_out;
	delete vector_out2;
	////////////////////////////////////////////////////////////
	*/
	///////////UNCOMMENT BLOCK OF CODE BELOW TO PERFORM TESTS ON DENSE SOLVERS
	
    Test<double> General_Solver;
    std::cout << "Test for Dense solvers:\n";
    General_Solver.Run_Test(verb_0, General_Solver_Test,All_Dense);
	

	/////////UNCOMMENT BLOCK OF CODE BELOW TO PERFORM TESTS ON SPARSE SOLVERS
    
    Test<double> Sparse_Solver;
	std::cout << "Test for Sparse solvers:\n";
    Sparse_Solver.Run_Test(verb_0, CSR_Solver_Test, All_Sparse);
	

	////////UNCOMMENT BLOCK OF CODE BELOW TO PERFORM TIMING TESTS ON SPARSE MATRICES
	////////RESULTS WILL BE WRITTEN TO THE FILE
	////////the matrices will be originally generated as SPARSE CSR matrices
	////////and then converted to normal sparse matrices for dense solvers
	/*
	Test<double> Timing_Test_Sparse;
	Timing_Test_Sparse.Time_Solvers_Sparse();
	*/

	////////UNCOMMENT BLOCK OF CODE BELOW TO PERFORM TIMING TESTS ON DENSE MATRICES
	////////RESULTS WILL BE WRITTEN TO THE FILE
	///////the matrices will be originally generated as DENSE CSR matrices
	///////and then converted to normal dense matrices for dense solvers
	/*
	Test<double> Test_Timing_Dense;
	Test_Timing_Dense.Time_Solvers_Dense();
	*/

	//stop but for windows only
    //system("PAUSE"); 
}
