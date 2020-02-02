#include "Test.h"
#include "Matrix.h"
#include "CSRMatrix.h"
#include <iostream>
#include <fstream>
#include <chrono>

template <class T>
void Test<T>::Generate_Sparse_CSR_Matrix(CSRMatrix<T>& M)
{
    //Generate random sparse matrix given the parameters
    this->Initialize_CSRMatrix(M);
    //Engine setup for random number generation
    std::random_device r;
    std::default_random_engine eng{ r() };
    std::uniform_real_distribution<double> diagonal(this->DIAG_MIN, this->DIAG_MAX);
    std::uniform_real_distribution<double> general(this->GENERAL_MIN, this->GENERAL_MAX);
    std::uniform_real_distribution<double> percentile_dist(0,1);

    //Temporary variables
    T value;
    double exist;

    //Generation of random placements and elements for the whole matrix
    for (int i = 0; i <this->rows; i++)
    {
        for (int j = 0; j < i; j++)
        {
            //Generate random variable which decides whether M[i][j] value is going to be set depending on user configuration
            exist = percentile_dist(eng);
            if (this->percentage >= exist)
            {
                //Generate random value to set, and set both M[i][j] and M[j][i]
                value = general(eng);
                M.setvalues(value, i, j);
                M.setvalues(value, j, i);
            }
        }
      
    }

    //Generation of diagonal elements
    for (int i = 0; i < M.rows; i++)
    {
        value = diagonal(eng);
        M.setvalues(value, i, i);
    }
}

template <class T>
void Test<T>::Generate_Dense_CSR_Matrix(CSRMatrix<T>& M)
{
    this->Initialize_CSRMatrix(M);
    //Engine setup for random number generation
    std::random_device r;
    std::default_random_engine eng{ r() };
    std::uniform_real_distribution<double> diagonal(this->DIAG_MIN, this->DIAG_MAX);
    std::uniform_real_distribution<double> general(this->GENERAL_MIN, this->GENERAL_MAX);

    //Temporary variables
    T value;

    //Generation of random placements and elements for the whole matrix
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = i + 1; j < M.cols; j++)
        {
            value = general(eng);
            M.setvalues(value, i, j);
            M.setvalues(value, j, i);
        }
    }

    //Generation of diagonal elements
    for (int i = 0; i < M.rows; i++)
    {
        value = diagonal(eng);
        M.setvalues(value, i, i);
    }
}

template <class T>
void Test<T>::Generate_Dense_Matrix(Matrix<T>& Matrix2)
{
    //Engine setup for random number generation
    std::random_device r;
    std::default_random_engine eng{ r() };
    std::uniform_real_distribution<double> diagonal(this->DIAG_MIN, this->DIAG_MAX);
    std::uniform_real_distribution<double> general(this->GENERAL_MIN, this->GENERAL_MAX);

    T value;
    //Generate random value for each of the matrix element
    for (int i = 0; i < Matrix2.rows; i++)
    {
        for (int j = i + 1; j < Matrix2.cols; j++)
        {
            //Generate value depending on user configuration
            value = general(eng);
            Matrix2.values[i * Matrix2.cols + j] = value;
            Matrix2.values[j * Matrix2.cols + i] = value;
        }
    }

    for (int i = 0; i < Matrix2.rows; i++)
    {
        //Generate values on the diagonal
        value = diagonal(eng);
        Matrix2.values[i * Matrix2.cols + i] = value;
    }

}

template <class T>
double Test<T>::Compare_Two_Matrices(const Matrix<T>& Matrix_1, const Matrix<T>& Matrix_2)
{
    //Compare the L1 norm between two vectors (or matrices)
    double Output = 0;
    for (int i = 0; i < Matrix_1.rows; i++)
    {
        for (int j = 0; j < Matrix_1.cols; j++)
        {
            Output = Output + abs(Matrix_1.values[i* Matrix_1.cols+j] - Matrix_2.values[i * Matrix_1.cols + j]);
        }

    }
    std::cout << "Difference between those 2 vectors/matrices is : " << Output << "\n";
    return Output;
    
}


template <class T>
void Test<T>::Initialize_CSRMatrix(CSRMatrix<T>& M)
{
    //Initialize the values of row_position of CSR matrix, which is required in order to use setvalues
    for (int i = 0; i < M.rows + 1; i++)
    {
        M.row_position[i] = 0;
    }
}

template <class T>
void Test<T>::Initialize_Empty_Vector(Matrix<T>& Vector)
{
    //Initialize the values of the vector to 0
    for (int i = 0; i < Vector.rows ; i++)
    {
        Vector.values[i] = 0;
    }
}

template <class T>
void Test<T>::Generate_Random_Vector(Matrix<T>& Vector)
{
    //Engine setup for random number generation
    std::random_device r;
    std::default_random_engine eng{ r() };
    std::uniform_real_distribution<double> general(this->GENERAL_MIN, this->GENERAL_MAX);

    T value;
    //For each element of the vector generate random value, configured by the user
    for (int i = 0; i < Vector.rows ; i++)
    {
        value = general(eng);
        Vector.values[i] = value;
    }
}
template<class T>
void Test<T>::File_for_python(Matrix<T>& M,Matrix<T> &b,Matrix<T> &x, std::string file_name)
{
	std::ofstream myfile;
    // Create a python style file in order to double check the result of the tests
	myfile.open(file_name + ".txt");
	myfile << "M=np.array([[";
    // Print an matrix 
	for (int i = 0; i < M.rows; i++)
	{
		for (int j = 0; j < M.cols; j++)
		{
			if (j != M.cols - 1)
			{
				myfile << M.values[i * M.cols + j] << ",";
			}
			else
			{
				myfile << M.values[i * M.cols + j];
			}
		}
		if (i != M.rows - 1)
		{

			myfile << "],\n[";
		};
	}myfile << "]])\n";
	myfile << "b=np.array([[";
    // Print an b vector
	for (int i = 0; i < b.rows; i++)
	{
		for (int j = 0; j < b.cols; j++)
		{
			if (j != b.cols - 1)
			{
				myfile << b.values[i * b.cols + j] << ",";
			}
			else
			{
				myfile << b.values[i * b.cols + j];
			}
		}
		if (i != M.rows - 1)
		{

			myfile << "],\n[";
		};
	}myfile << "]])\n";
	myfile << "x=np.array([[";
    //Print calculated vector
	for (int i = 0; i < x.rows; i++)
	{
		for (int j = 0; j < x.cols; j++)
		{
			if (j != x.cols - 1)
			{
				myfile << x.values[i * x.cols + j] << ",";
			}
			else
			{
				myfile << x.values[i * x.cols + j];
			}
		}
		if (i != x.rows - 1)
		{

			myfile << "],\n[";
		};
	}myfile << "]])\n";
	myfile << "b_new=M@x\nprint(b-b_new)";
	myfile.close();
	
}
template <class T>
void Test<T>::Output_Matrix_To_File(Matrix<T>& M,std::string file_name)
{
    //Print matrix to the file for the purpose of debugging
	std::ofstream myfile;
	myfile.open(file_name + ".txt");
	myfile << "[";
	for (int i = 0; i < M.rows; i++)
	{
		for (int j = 0; j < M.cols; j++)
		{
			if (j != M.cols - 1)
			{
				myfile << M.values[i * M.cols + j] << ",";
			}
			else
			{
				myfile << M.values[i * M.cols + j];
			}
		}
		if (i != M.rows - 1)
		{

			myfile << "],\n[";
		}
	}myfile << "]";
	myfile.close();
}

template <class T>
void Test<T>::Dense_Matrix_Solver_Test(int Solver_Type)
{
    //Test a dense matrix solver for a particular solver type
    //Generate a random matrix with parameters configured by test class variables
    auto* DenseMatrix = new Matrix   <T>(this->rows, this->cols, true);
    this->Generate_Dense_Matrix(*DenseMatrix);
    //Generate random vector
    auto* Vector_B = new Matrix   <T>(this->rows, 1, true);
    this->Generate_Random_Vector(*Vector_B);

    auto* Output_Vector = new Matrix   <T>(this->rows, 1, true);

    this->Initialize_Empty_Vector(*Output_Vector);

    //Run a solver on a matrix
    DenseMatrix->solver(*Vector_B, *Output_Vector, Solver_Type);

    //Check if the solver got a correct solution
    if (!this->Test_Dense_Solution(*Output_Vector, *Vector_B, *DenseMatrix))
    {
        std::cout << "Incorrect Solve" << "\n";
        this->Fail = true;
    }
    else
    {
        std::cout << "Correct Solve" << "\n";
    }
    //Depending of the level of debug required output some matrix and calculated vector
    if (this->verbose >= verb_4)
    {
        std::cout << "Random Matrix:" << "\n";
        DenseMatrix->printMatrix();
        std::cout << "Vector B:" << "\n";
        Vector_B->printValues();
        std::cout << "Calculated X:" << "\n";
        Output_Vector->printValues();
    }

    delete DenseMatrix;
    delete Vector_B;
    delete Output_Vector;

}

template <class T>
void Test<T>::Sparse_Matrix_Solver_Test(int Solver_Type)
{
    //Simillarly to test dense test, but this time generate a sparse matrix
    auto* Sparse_Matrix = new CSRMatrix <double>(this->rows, this->cols, 0, true);
    this->Initialize_CSRMatrix(*Sparse_Matrix);
    this->Generate_Sparse_CSR_Matrix(*Sparse_Matrix);
    //Generate random vector
    auto* Vector_B = new Matrix   <T>(this->rows, 1, true);
    this->Generate_Random_Vector(*Vector_B);

    auto* Output_Vector = new Matrix   <T>(this->rows, 1, true);

    this->Initialize_Empty_Vector(*Output_Vector);
    // Run a sparse solver on a generated matrix
    Sparse_Matrix->solver(*Vector_B, *Output_Vector, Solver_Type);

    //Check if the solution was correct
    if (!this->Test_Sparse_Solution(*Output_Vector, *Vector_B, *Sparse_Matrix))
    {
        std::cout << "Incorrect Solve" << "\n";
        this->Fail = true;
    }
    else
    {
        std::cout << "Correct Solve" << "\n";
    }
    //Depending of the level of debug required output some matrix and calculated vector
    if (this->verbose >= verb_4)
    {
        std::cout << "Random Matrix:" << "\n";
        Sparse_Matrix->printMatrix();
        std::cout << "Vector B:" << "\n";
        Vector_B->printValues();
        std::cout << "Calculated X:" << "\n";
        Output_Vector->printValues();
    }

    delete Sparse_Matrix;
    delete Vector_B;
    delete Output_Vector;

}

template<class T>
bool Test<T>::Test_Dense_Solution(Matrix<T>& Vector_X, Matrix<T>& Vector_Actual_B, Matrix<T>& M)
{
    //Compairs the value of actual b vector of equation A*x=b, with the value obtained by multiplying A*x' where x' is our solution
    bool Output = true;
    auto* Vector_Model_B = new Matrix<T>(this->rows, 1, true);
    this->Initialize_Empty_Vector(*Vector_Model_B);

    M.matVecMult(Vector_X, *Vector_Model_B);
    //Print information depending on the level of information required by the user
    if (this->verbose>=verb_2)
    {
        std::cout << "Model answer is " << "\n";
        Vector_Model_B->printValues();
        std::cout << "Absolute differences between actual and calculated solution" << "\n";
    }

	double abs_difference = 0;
    //Calculate the b_model vector by multiplying A and x and comapre it to the actual b vector
    for (int i = 0; i < this->rows; i++)
    {
		abs_difference += (abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i]));

        if (this->verbose >= verb_2)
        {
            std::cout << (abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i])) << " ";
        }

        if (abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i]) > this->accuracy)
        {
            Output = false;
            break;
        }

        if (abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i]) != abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i]))
        {
            Output = false;
            break;
        }

    }
	if (this->verbose == verb_1)
	{
		std::cout << "Absolute differences between actual and calculated solution summed:\n" << abs_difference << "\n";
	}
    if (this->verbose >= verb_2)
    {
        std::cout << "\n";
    }

    delete Vector_Model_B;
    return Output;
}

template<class T>
bool Test<T>::Test_Sparse_Solution(Matrix<T>& Vector_X, Matrix<T>& Vector_Actual_B, CSRMatrix<T>& M)
{
    //Compairs the value of actual b vector of equation A*x=b, with the value obtained by multiplying A*x' where x' is our solution
    bool Output = true;
    auto* Vector_Model_B = new Matrix<T>(this->rows, 1, true);
    this->Initialize_Empty_Vector(*Vector_Model_B);

    M.matVecMult(Vector_X, *Vector_Model_B);
	double abs_difference = 0;
    //Print information depending on the level of information required by the user
    if (this->verbose >= verb_2)
    {
        std::cout << "Absolute differences between actual and calculated solution" << "\n";
    }
    //Calculate the b_model vector by multiplying A and x and comapre it to the actual b vector
    for (int i = 0; i < this->rows; i++)
    {
		abs_difference += (abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i]));

        if (this->verbose >= verb_2)
        {
            std::cout << (abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i])) << " ";
        }
        if (abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i]) != abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i]))
        {
            Output = false;
            break;
        }

        if (abs(Vector_Actual_B.values[i] - Vector_Model_B->values[i]) > this->accuracy)
        {
            Output = false;
            break;
        }
    }
	if (this->verbose == verb_1)
	{
		std::cout << "Absolute differences between actual and calculated solution summed:\n" << abs_difference << "\n";
	}
    if (this->verbose >= verb_2)
    {
        std::cout << "\n";
    }
    delete Vector_Model_B;
    return Output;
}

template <class T>
void Test<T>::Solver_Timing_Test_All()
{
    bool Test_Sparse = true;
    //General timing test of all solvers (both sparse and dense)
    if (Test_Sparse)
    {
        this->DIAG_MIN = 100;
        this->DIAG_MAX = 200;
        this->rows = 500;
        this->cols = 500;
        this->percentage = 0.01;
    }
    else
    {
        this->DIAG_MIN = 100;
        this->DIAG_MAX = 200;
        this->rows = 200;
        this->cols = 200;
    }
    //Depending on the configuration either generate dense matrix and copy it to sparse or the other way around
    auto* DenseMatrix = new Matrix   <T>(this->rows, this->cols, true);
    auto* Sparse_Matrix = new CSRMatrix <T>(this->rows, this->cols, 0, true);
    if (Test_Sparse)
    {
        this->Generate_Sparse_CSR_Matrix(*Sparse_Matrix);
        Sparse_Matrix->Convert_CSRMatrix_To_Matrix(*DenseMatrix);
    }
    else
    {
        this->Generate_Dense_Matrix(*DenseMatrix);
        this->Initialize_CSRMatrix(*Sparse_Matrix);
        Sparse_Matrix->Convert_Matrix_To_CSR_Matrix(*DenseMatrix);
    }



    auto* Vector_B = new Matrix   <T>(this->rows, 1, true);
    this->Generate_Random_Vector(*Vector_B);

    auto* Output_Vector = new Matrix   <T>(this->rows, 1, true);
    //For each of the dense solvers run a test on input matrices
    for (int i = Jacobi; i != Last_Dense; i++)
    {
        this->Initialize_Empty_Vector(*Output_Vector);
        std::chrono::steady_clock::time_point Begin_Dense = std::chrono::steady_clock::now();
        DenseMatrix->solver(*Vector_B, *Output_Vector, i);
        std::chrono::steady_clock::time_point End_Dense = std::chrono::steady_clock::now();
        std::cout << "Time difference Dense Method "<<i <<" is "  << std::chrono::duration_cast<std::chrono::microseconds>(End_Dense - Begin_Dense).count() << "[us]" << std::endl;
        if (this->verbose >= 2)
        {
            Output_Vector->printValues();
        }
    }
    //For each of the sparse solvers run a test on input matrices
    for (int i = Jacobi_CSR; i != Last_Sparse; i++)
    {
        this->Initialize_Empty_Vector(*Output_Vector);
        std::chrono::steady_clock::time_point Begin_Sparse = std::chrono::steady_clock::now();
        Sparse_Matrix->solver(*Vector_B, *Output_Vector, i);
        std::chrono::steady_clock::time_point End_Sparse = std::chrono::steady_clock::now();
        std::cout << "Time difference Sparse Method " << i << " is " << std::chrono::duration_cast<std::chrono::microseconds>(End_Sparse - Begin_Sparse).count() << "[us]" << std::endl;
        if (this->verbose >= 2)
        {
            Output_Vector->printValues();
        }
    }

   delete Sparse_Matrix;
   delete DenseMatrix;
   delete Vector_B;
   delete Output_Vector;
}

template <class T>
void Test<T>::Time_Solvers_Dense()
{

    std::ofstream myfile;
    myfile.open("Dense_Test_100_200.csv");
    myfile << "Matrix Size" << "," << "Jacobi" << "," << "Gauss_Siedel" << "," << "Gaussian" << "," << "LU" << "," << "Inverse" << "," << "Cholesky" << "," << "Conjugate_Gradient" << "," << "Gauss_Jordan" <<  "," << "Cramers" << ","<<"Jacobi_CSR" << "," << "Gauss_Siedel_CSR" << "," << "Cholesky_CSR" << "," << "Conjugate_Gradient_CSR" << "\n";

    //Run a detailed test with lots of configurability (run it on the dense matrix).
    //It allows to compare the robustness and timing of different solver depending on size and type of matrix
    //Afterwards it outputs the results to the file
    for (int j = 1; j < 21; j++)
    {
        std::cout << j << "\n";
        // Generate the matrix and vary the paramters of interest
        this->verbose = 0;
        this->rows = 10*j;
        this->cols = 10*j;
        this->DIAG_MIN = 100;
        this->DIAG_MAX = 200;
        auto* DenseMatrix = new Matrix   <T>(this->rows, this->cols, true);
        auto* Sparse_Matrix = new CSRMatrix <T>(this->rows, this->cols, 0, true);
        auto* Vector_B = new Matrix   <T>(this->rows, 1, true);

        this->Generate_Random_Vector(*Vector_B);
        this->Generate_Dense_CSR_Matrix(*Sparse_Matrix);
        Sparse_Matrix->Convert_CSRMatrix_To_Matrix(*DenseMatrix);

        //Output relevant parameters to file
        myfile << 10*j << ",";

        double timing;
        //Run a test for each of the dense solvers and write the time taken to the file
        for (int i = Jacobi; i != Last_Dense; i++)
        {
           
                if (i == Cramers && j > 20)
                {
                    myfile  << ",";
                }
                else
                {
                    timing = this->Test_Timing_Dense(*Vector_B, *DenseMatrix, i);
                    myfile << timing << ",";
                }
            
        }
        //Run a test for each of the sparse solvers and write the time taken to the file
       for (int i = Jacobi_CSR; i != Last_Sparse; i++)
       {
       
          timing = this->Test_Timing_Sparse(*Vector_B, *Sparse_Matrix, i);
           myfile << timing << ",";
       
       }

        myfile << "\n";



        delete Vector_B;
        delete DenseMatrix;
        delete Sparse_Matrix;
    }

    myfile.close();

}

template <class T>
void Test<T>::Time_Solvers_Sparse()
{

    std::ofstream myfile;
    myfile.open("Sparse_Test_100_200.csv");

    myfile << "Matrix Size"<< "," << "Sparsity" << "," << "Jacobi" << "," << "Gauss_Siedel" << "," <<"Gaussian"<<","<< "LU" << "," <<"Inverse"<< "," << "Cholesky" << "," << "Conjugate_Gradient" << ","<<"Gauss_Jordan"<<"," << "Cramers"<< ","<<"Jacobi_CSR" << "," << "Gauss_Siedel_CSR" << "," << "Cholesky_CSR" << "," << "Conjugate_Gradient_CSR" << "\n";
    //Run a detailed test with lots of configurability (run it on the sparse matrix).
    //It allows to compare the robustness and timing of different solver depending on size and type of matrix
    //Afterwards it outputs the results to the file
    for (int j = 1; j < 21; j++)
    {

        int size = 10;
        //Configure all the required parameters of matrix
        std::cout << j << "\n";
		this->verbose = 0;

		this->rows = size * j;
		this->cols = size * j;

		this->DIAG_MIN = 100;
		this->DIAG_MAX = 200;
		this->percentage = 0.01;
		auto* DenseMatrix = new Matrix   <T>(this->rows, this->cols, true);
		auto* Sparse_Matrix = new CSRMatrix <T>(this->rows, this->cols, 0, true);
		auto* Vector_B = new Matrix   <T>(this->rows, 1, true);

		this->Generate_Random_Vector(*Vector_B);
		this->Generate_Sparse_CSR_Matrix(*Sparse_Matrix);
		Sparse_Matrix->Convert_CSRMatrix_To_Matrix(*DenseMatrix);

		myfile << size * j << ",";
		myfile << percentage << ",";

		double timing;

        //Run test on each of the dense solvers and write result to the file
        for (int i = Jacobi; i != Last_Dense; i++)
		{

			timing = this->Test_Timing_Dense(*Vector_B, *DenseMatrix, i);
			myfile << timing << ",";

		}
        //Run test on each of the sparse solvers and write result to the file
        for (int i = Jacobi_CSR; i != Last_Sparse; i++)
		{

			timing = this->Test_Timing_Sparse(*Vector_B, *Sparse_Matrix, i);
			myfile << timing << ",";
		}

        myfile << "\n";



        delete Vector_B;
        delete DenseMatrix;
        delete Sparse_Matrix;
    }

    myfile.close();

}

template <class T>
double Test<T>::Test_Timing_Sparse(Matrix<T>& Vector_B, CSRMatrix<T>& M, int Solver)
{
    //Run single sparse solver on particular matrix, and calculate time it took for the solver to complete
    auto* Output_Vector = new Matrix   <T>(this->rows, 1, true);
    this->Initialize_Empty_Vector(*Output_Vector);

    std::chrono::steady_clock::time_point Begin_Sparse = std::chrono::steady_clock::now();
    M.solver(Vector_B, *Output_Vector, Solver);
    std::chrono::steady_clock::time_point End_Sparse = std::chrono::steady_clock::now();
    //Check if solver calculated correct answer and only return actual time if answer is accurate 
    if (this->Test_Sparse_Solution(*Output_Vector, Vector_B, M))
    {
        delete Output_Vector;
        return std::chrono::duration_cast<std::chrono::microseconds>(End_Sparse - Begin_Sparse).count();
    }
    else
    {
        delete Output_Vector;
        return -1;
    }
}

template <class T>
double Test<T>::Test_Timing_Dense(Matrix<T>& Vector_B, Matrix<T>& M,int Solver)
{
    //Run single dense solver on particular matrix, and calculate time it took for the solver to complete
    auto* Output_Vector = new Matrix   <T>(this->rows, 1, true);
    this->Initialize_Empty_Vector(*Output_Vector);

    std::chrono::steady_clock::time_point Begin_Dense = std::chrono::steady_clock::now();
    M.solver(Vector_B, *Output_Vector, Solver);
    std::chrono::steady_clock::time_point End_Dense = std::chrono::steady_clock::now();
    //Check if solver calculated correct answer and only return actual time if answer is accurate 
    if (this->Test_Dense_Solution(*Output_Vector, Vector_B, M))
    {
        delete Output_Vector;
        return std::chrono::duration_cast<std::chrono::microseconds>(End_Dense - Begin_Dense).count();
    }
    else
    {
        delete Output_Vector;
        return -1;
    }
}

template <class T>
void Test<T>::Run_Test(int verbose, int test_index,int configuration)
{
    //General core rutine for running a test 
    std::cout << "###########################################################" << "\n";
    std::cout << "Running Test: " << test_index;
    std::cout << "\n" << "###########################################################" << "\n";
    this->verbose = verbose;
    // Run a test specified by the user
    switch (test_index) {
        case General_Solver_Test: 
            this->Run_General_Solver(configuration);
            break;
        case CSR_Solver_Test :
            this->Run_Sparse_Solver(configuration);
            break;
        case Timing_Test :
            this->Solver_Timing_Test_All();
            break;
        default: std::cout << "Give a valid test index" << "\n";
    }
    // Make it result of the test easily visible to the user
    if (this->Fail)
    {
        std::cout << "###########################################################" << "\n";
        std::cout << "Test Failed";
        std::cout << "\n" << "###########################################################" << "\n" << "\n";
    }

    if (!this->Fail)
    {
        std::cout << "###########################################################" << "\n";
        std::cout << "Test " << test_index << " Passed";
        std::cout << "\n" << "###########################################################" << "\n" << "\n";

    }

    
}

template <class T>
void Test<T>::Run_General_Solver(int configuration)
{
    //Run selection of tests on each of the solvers.
    //Quick test, which quickly lets you spot obvious problems with solvers
    this->DIAG_MIN = 40;
    this->DIAG_MAX = 70;
    for (int i = 10; i < 101; i = i + 10)
    {
        this->rows = i;
        this->cols = i;
        //It allows for running either single test or all the tests
        if (configuration == Gauss_Siedel || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config : Gauss Siedel" << "\n";
            }
            this->Dense_Matrix_Solver_Test(Gauss_Siedel);
            
        }
        if (configuration == Jacobi || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config : Jacobi" << "\n";
            }
            this->Dense_Matrix_Solver_Test(Jacobi);
        }
        if (configuration == LU || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config : LU" << "\n";
            }
            this->Dense_Matrix_Solver_Test(LU);
        }
        if ((configuration == Cholesky) || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config : Cholesky Decomposition" << "\n";
            }
            this->Dense_Matrix_Solver_Test(Cholesky);
        }   
        if (configuration == Gaussian || configuration == All_Dense)
        {
          if (this->verbose >= 1)
           {
               std::cout << "Running General Solver Test on config : Gaussian" << "\n";
           }
           this->Dense_Matrix_Solver_Test(Gaussian);
        }
        if (configuration == Inverse || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config : Inverse" << "\n";
            }
            this->Dense_Matrix_Solver_Test(Inverse);
        }
        if (configuration == Conjugate_Gradient || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config : Conjugate gradient" << "\n";
            }
            this->Dense_Matrix_Solver_Test(Conjugate_Gradient);
        }
        if (configuration == Gauss_Jordan || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config : Gauss Jordan" << "\n";
            }
            this->Dense_Matrix_Solver_Test(Gauss_Jordan);
        }
	    if (configuration == Cramers || configuration == All_Dense)
            {
		    if (this->verbose >= 1)
		    {
			    std::cout << "Running General Solver Test on config : Cramers" << "\n";
		    }
		    this->Dense_Matrix_Solver_Test(Cramers);
            }
    }
}

template <class T>
void Test<T>::Run_Sparse_Solver(int configuration)
{
    //Run selection of tests on each of the solvers.
    //Quick test, which quickly lets you spot obvious problems with solvers
    this->DIAG_MIN = 40;
    this->DIAG_MAX = 80;

    this->percentage = 0.01;
    for (int i = 10; i < 101; i = i + 10)
    {
        this->rows = i;
        this->cols = i;
        //It allows for running either single test or all the tests
        if (configuration == Jacobi_CSR || configuration == All_Sparse)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running Sparse Solver Test on config : Jacobi_CSR" << "\n";
            }
            this->Sparse_Matrix_Solver_Test(Jacobi_CSR);
        }
        if (configuration == Gauss_Siedel_CSR || configuration == All_Sparse)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running Sparse Solver Test on config : Gauss-Seidel_CSR" << "\n";
            }
            this->Sparse_Matrix_Solver_Test(Gauss_Siedel_CSR);
        }
        if (configuration == Cholesky_CSR || configuration == All_Sparse)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running Sparse Solver Test on config : Cholesky_CSR" << "\n";
            }
            this->Sparse_Matrix_Solver_Test(Cholesky_CSR);
        }
        if (configuration == Conjugate_Gradient_CSR || configuration == All_Sparse)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running Sparse Solver Test on config : Conjugate_Gradient_CSR" << "\n";
            }
           
            this->Sparse_Matrix_Solver_Test(Conjugate_Gradient_CSR);
        }

    }
    
}


