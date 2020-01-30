#include "Test.h"
#include "Matrix.h"
#include "CSRMatrix.h"
#include <iostream>
#include <fstream>
#include <chrono>

template <class T>
void Test<T>::Generate_Sparse_CSR_Matrix(CSRMatrix<T>& M)
{
    this->Initialize_CSRMatrix(M);
    //Engine setup for random number generation
    std::random_device r;
    std::default_random_engine eng{ r() };
    std::uniform_real_distribution<double> diagonal(this->DIAG_MIN, this->DIAG_MAX);
    std::uniform_real_distribution<double> general(this->GENERAL_MIN, this->GENERAL_MAX);

    //Temporary variables
    T value;
    int row;
    int col;

    //Generation of random placements and elements for the whole matrix
    for (int i = 0; i < M.rows * M.cols * this->percentage / 2; i++)
    {
        //std::cout << "bla";
        row = rand() % M.rows;
        col = rand() % M.cols;
        value = general(eng);
        M.setvalues(value, row, col);
        M.setvalues(value, col, row);
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

    for (int i = 0; i < Matrix2.rows; i++)
    {
        for (int j = i + 1; j < Matrix2.cols; j++)
        {
            value = general(eng);
            Matrix2.values[i * Matrix2.cols + j] = value;
            Matrix2.values[j * Matrix2.cols + i] = value;
        }
    }

    for (int i = 0; i < Matrix2.rows; i++)
    {
        value = diagonal(eng);
        Matrix2.values[i * Matrix2.cols + i] = value;
    }


	//std::cout << "Generated matrix is: " << std::endl;
	//Matrix2.printMatrix();
}

template <class T>
double Test<T>::Compare_Two_Matrices(const Matrix<T>& Matrix_1, const Matrix<T>& Matrix_2)
{
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
    for (int i = 0; i < M.rows + 1; i++)
    {
        M.row_position[i] = 0;
    }
}

template <class T>
void Test<T>::Initialize_Empty_Vector(Matrix<T>& Vector)
{
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

    for (int i = 0; i < Vector.rows ; i++)
    {
        value = general(eng);
        Vector.values[i] = value;
    }
}

template <class T>
void Test<T>::Output_Matrix_To_File(Matrix<T>& M,std::string file_name)
{
    std::ofstream myfile;
    myfile.open(file_name+".csv");
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            myfile << M.values[i * M.cols + j] << ",";
        }
        myfile << "\n";
    }
    myfile.close();
}

template <class T>
void Test<T>::Dense_Matrix_Solver_Test(int Solver_Type)
{
    auto* DenseMatrix = new Matrix   <T>(this->rows, this->cols, true);
    this->Generate_Dense_Matrix(*DenseMatrix);

    auto* Vector_B = new Matrix   <T>(this->rows, 1, true);
    this->Generate_Random_Vector(*Vector_B);

    auto* Output_Vector = new Matrix   <T>(this->rows, 1, true);

    this->Initialize_Empty_Vector(*Output_Vector);

    DenseMatrix->solve(*Vector_B, *Output_Vector, Solver_Type);

    if (!this->Test_Dense_Solution(*Output_Vector, *Vector_B, *DenseMatrix))
    {
        std::cout << "Incorrect Solve" << "\n";
        this->Fail = true;
    }
    else
    {
        std::cout << "Correct Solve" << "\n";
    }

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
    auto* Sparse_Matrix = new CSRMatrix <double>(this->rows, this->cols, 0, true);
    //this->Generate_Sparse_CSR_Matrix(*Sparse_Matrix);
    this->Initialize_CSRMatrix(*Sparse_Matrix);
    this->Generate_Sparse_CSR_Matrix(*Sparse_Matrix);

    auto* Vector_B = new Matrix   <T>(this->rows, 1, true);
    this->Generate_Random_Vector(*Vector_B);

    auto* Output_Vector = new Matrix   <T>(this->rows, 1, true);

    this->Initialize_Empty_Vector(*Output_Vector);

    Sparse_Matrix->solve(*Vector_B, *Output_Vector, Solver_Type);

    if (!this->Test_Sparse_Solution(*Output_Vector, *Vector_B, *Sparse_Matrix))
    {
        std::cout << "Incorrect Solve" << "\n";
        this->Fail = true;
    }
    else
    {
        std::cout << "Correct Solve" << "\n";
    }

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

    if (this->verbose>=verb_2)
    {
        std::cout << "Model answer is " << "\n";
        Vector_Model_B->printValues();
        std::cout << "Absolute differences between actual and calculated solution" << "\n";
    }

	double abs_difference = 0;

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

    if (this->verbose >= verb_2)
    {
        std::cout << "Absolute differences between actual and calculated solution" << "\n";
    }

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



    if (Test_Sparse)
    {
        this->DIAG_MIN = 100;
        this->DIAG_MAX = 200;
        this->rows = 500;
        this->cols = 500;
        this->percentage = 0.01;


        //this->Generate_Sparse_CSR_Matrix(*Sparse_Matrix);
        //Sparse_Matrix->Convert_CSRMatrix_To_Matrix(*DenseMatrix);
    }
    else
    {
        this->DIAG_MIN = 100;
        this->DIAG_MAX = 200;
        this->rows = 200;
        this->cols = 200;
       //this->Generate_Dense_Matrix(*DenseMatrix);
       //this->Initialize_CSRMatrix(*Sparse_Matrix);
       //Sparse_Matrix->Convert_Matrix_To_CSR_Matrix(*DenseMatrix);
    }
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

    for (int i = Jacobi; i != Last_Dense; i++)
    {
        this->Initialize_Empty_Vector(*Output_Vector);
        std::chrono::steady_clock::time_point Begin_Dense = std::chrono::steady_clock::now();
        DenseMatrix->solve(*Vector_B, *Output_Vector, i);
        std::chrono::steady_clock::time_point End_Dense = std::chrono::steady_clock::now();
        std::cout << "Time difference Dense Method "<<i <<" is "  << std::chrono::duration_cast<std::chrono::microseconds>(End_Dense - Begin_Dense).count() << "[us]" << std::endl;
        if (this->verbose >= 2)
        {
            Output_Vector->printValues();
        }
    }

    for (int i = Jacobi_CSR; i != Cholesky_CSR; i++)
    {
        this->Initialize_Empty_Vector(*Output_Vector);
        std::chrono::steady_clock::time_point Begin_Sparse = std::chrono::steady_clock::now();
        Sparse_Matrix->solve(*Vector_B, *Output_Vector, i);
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
void Test<T>::Run_Test(int verbose, int test_index,int configuration)
{
    std::cout << "###########################################################" << "\n";
    std::cout << "Running Test: " << test_index;
    std::cout << "\n" << "###########################################################" << "\n";
    this->verbose = verbose;
  
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
    this->DIAG_MIN = 40;
    this->DIAG_MAX = 80;
    for (int i = 5; i < 100; i = i + 10)
    {
        this->rows = i;
        this->cols = i;
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
            //this->DIAG_MIN = 10;
            //this->DIAG_MAX = 20;
            //this->GENERAL_MIN = 0;
            //this->GENERAL_MAX = 1;
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
        if (configuration == Gauss_Jordan_inverse || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config :  Gauss_Jordan_inverse" << "\n";
            }
            this->Dense_Matrix_Solver_Test(Gauss_Jordan_inverse);
        }
        if (configuration == Conjugate_Gradient || configuration == All_Dense)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running General Solver Test on config :  Conjugate gradient" << "\n";
            }
            this->Dense_Matrix_Solver_Test(Conjugate_Gradient);
        }
    }
}

template <class T>
void Test<T>::Run_Sparse_Solver(int configuration)
{
    this->DIAG_MIN = 40;
    this->DIAG_MAX = 80;

    this->percentage = 0.001;
    for (int i = 10; i < 500; i = i + 100)
    {
        this->rows = i;
        this->cols = i;

        if (configuration == Jacobi_CSR || configuration == All_Sparse)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running Sparse Solver Test on config : Jacobi" << "\n";
            }
            this->Sparse_Matrix_Solver_Test(Jacobi_CSR);
        }
        if (configuration == LU || configuration == All_Sparse)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running Sparse Solver Test on config : LU" << "\n";
            }
            this->Sparse_Matrix_Solver_Test(Gauss_Siedel_CSR);
        }
        if (configuration == Cholesky_CSR || configuration == All_Sparse)
        {
            if (this->verbose >= 1)
            {
                std::cout << "Running Sparse Solver Test on config : Cholesky_CSR" << "\n";
            }
            std::cout << "Fight to implement this solver continues" << "\n";
            //this->Sparse_Matrix_Solver_Test(Cholesky_CSR);
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


