#ifndef MY_BANDED_MATRIX_CLASS

#define MY_BANDED_MATRIX_CLASS

#include "CSRMatrix.h"
#include "Matrix.h"
#include <vector>
#include <queue>
using namespace std;

// BandedMatrix class inherits CSR Matrix class
template <class T>
class BandedMatrix:public CSRMatrix<T> 
{
public:
    T rows = 0;
    T cols = 0;
    T* values = nullptr;
    T nnzs = -1;

    //Constructor
    BandedMatrix(int rows, int cols, int nnzs, bool preallocate);
    //destructor
    ~BandedMatrix();
    
    // Function for getting every node degree
    vector<T> get_degree();

    // Print banded matrix function
    void printBandedMatrix();

    // Main function for running cithill-Mckee. ***Note: bugs inside.***
    void Cuthill_Mckee(vector<T> degree_array);

    // Reverse the result from Cuthii-Mckee algorithm
    void reverse_R();

    // Use the Reverse Cuthill-Mckee algorithm result to permutate original matrix
    // It will print out the output banded matrix
    void permutate_to_Band();

    // Main function for running RCM.
    void Run_Reverse_Cuthill_Mckee();
    

protected:
    bool preallocated = false;



private:
    // Initialisation of containers for running RCM algorithm
    queue<int> Q;

    // Initialize the permutation vector. Uncomment if Cuthill_Mckee works.
    // vector<double> R;

    // initialize R like this for testing reason. 
    // Since the Reverse_Cuthill_Mckee does not work, R cannot be obtained by the algorithm
    // The following R is from an online example source: http://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
    vector<double> R{ 0, 4, 2, 1, 5, 7, 3, 6 };

    vector<double> temp_degree;
};
#endif