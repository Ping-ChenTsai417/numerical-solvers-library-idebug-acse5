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
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    srand(time(NULL));
    int single = General_Solver_Test;
    bool single_test = true;

    Test<double> General_Solver;
    std::cout << "Test for Dense solvers:\n";
    General_Solver.Run_Test(verb_0, General_Solver_Test,All_Dense);
    std::cout << "Test for Sparse solvers:\n";
    //General_Solver.Run_Test(verb_0, General_Solver_Test, All_Sparse);
    Test<double> Sparse_Solver;
    Sparse_Solver.Run_Test(verb_0, CSR_Solver_Test, All_Sparse);
 

    system("PAUSE");
}
