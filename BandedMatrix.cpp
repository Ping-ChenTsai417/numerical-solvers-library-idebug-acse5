#include "BandedMatrix.h"
#include <iostream>
#include <vector>
#include <iterator> 
#include <algorithm>
#include <queue>
#include <map>
#include <functional> 
#include <set> 

using namespace std;

template <class T>
BandedMatrix<T>::BandedMatrix(int rows1, int cols1, int nnzs1, bool preallocate): CSRMatrix<T>(rows1, cols1, nnzs1, false)
{
    // If we don't pass false in the initialisation list base constructor, it would allocate memory in our base CSRmatrix class
    // So then we need to set it to the real value we had passed in
    this->preallocated = preallocate;
    this->nnzs = nnzs1;
    this->rows = rows1;
    this->cols = cols1;


    // If we want to handle memory ourselves
    if (this->preallocated)
    {
        // Must remember to delete this in the destructor
        this->values = new T[int(this->nnzs)];
        this->row_position = new int[int(this->rows + 1)];
        this->col_index = new int[int(this->nnzs)];
    }
    // The constructor is mostly copied from ACSE-5 lecture.
}

template <class T>
BandedMatrix<T>::~BandedMatrix()
{
    // Delete the values array
    if (this->preallocated) {
        delete[] this->row_position;
        delete[] this->col_index;
    }
    // The super destructor is called after we finish here
    // This will delete this->values if preallocated is true

    // The destructor is copied from ACSE-5 lecture.
}

// The printBandedMatrix() is copied from CSRmatrix.
template <class T>
void BandedMatrix<T>::printBandedMatrix()
{
    std::cout <<endl << "Printing matrix" << std::endl;
    std::cout << "Values: ";
    for (int j = 0; j < this->nnzs; j++)
    {
        std::cout << this->values[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "row_position: ";
    for (int j = 0; j < this->rows + 1; j++)
    {
        std::cout << this->row_position[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "col_index: ";
    for (int j = 0; j < this->nnzs; j++)
    {
        std::cout << this->col_index[j] << " ";
    }
    std::cout << "\n";
    std::cout << "\n";
    //Counter for index of element, which should be printerd
    int printed = 0;
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->cols; j++)
        {
            //Iterate over all rows and columns.
            //If the index points at value in the correct column as well as there are still values to be inserted in the current row we print the value and update index
            //Otherwise we print 0
            if (this->col_index[printed] == j and printed < this->row_position[i + 1])
            {
                std::cout << this->values[printed] << " ";
                printed++;
            }
            else
            {
                std::cout << "0" << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Function for getting every node degree by changing the original matrix into an adjacency matrix.
// The adjacency function can only be created if the original matrix is a sparse-positive-definite matrix 
template <class T>
vector<T> BandedMatrix<T>::get_degree()//CSRMatrix<T>& mat
{
    // make a copy of the original CSR matrix, change all non-zero number to 1
    //Now the copy is an adjacency matrix
    auto* adj_mat = new CSRMatrix<double>(rows, cols, nnzs, true);

    for (int i = 0; i < nnzs; i++)
    {
        //insert 1 if element at current position is non-zero
        adj_mat->values[i] = 1;

        //column index should be the same as the oriinal matrix
        adj_mat->col_index[i] = this->col_index[i];
    }
    for (int i = 0; i < this->rows + 1; i++)
    {
        adj_mat->row_position[i] = this->row_position[i];
    }

    double degree = 0;// initialize degree of each node
    vector<double> degree_array;
    // loop through the matrix row by row, so that we can check the degree of each node
    for (int i = 0; i < adj_mat->rows; i++)
    {
        degree = 0;
        for (int val_index = adj_mat->row_position[i]; val_index < adj_mat->row_position[i + 1]; val_index++)
        {

            if (i != adj_mat->col_index[val_index]) // if the node is not related to itself
            {
                degree += adj_mat->values[val_index]; // calculate number of total neighbouring nodes 
            }
        }
        degree_array.push_back(degree);
    }
    delete adj_mat;
    return degree_array;
}

// This function runs cuthill Mckee.  

/* The algorithm idea is to compute a permutation array R. Brief steps:
1. Exclude all the nodes that have been examined. If this is the first time entering the function, then nothing to exclude.
   Find the node with minimum degree value in degree_array except.  Get the node index and push all its neighbour to Q 
   in ascending degree order. All the elements in Q have not yet been examined.
2. Pop the first element in Q and insert the poped value into R if the value does not exist in R.
3. Everytime the value is poped, check if Q is empty. 
4. If Q is not empty, go back and start from the second step.
5. If Q is empty, check if the size of vector R is equal to the size of Matrix rows. 
6. If R.size() < this->rows, then go back and start from the first step. 
7. If R.size() == this->rows, then the algorithm ends. R eventually becomes a permutation array.
*/

//It has bugs inside!!! Do not run it. Buggy section is commented with /*...*/ below.
template <class T>
void BandedMatrix<T>::Cuthill_Mckee(vector<T> degree_array)
{
    // Initialize the node to be examined
    int node = 0;
    int neigh_col_idx = 0; // Initialize column index fro neighbour

    // look up the min value in temp array, find the index of this min value in degree array
    // if multiple min values, then choose the smallest element as the root node
   
    vector<double>::iterator min_val_ptr = min_element(temp_degree.begin(), temp_degree.end());
    double min_val = *min_element(temp_degree.begin(), temp_degree.end());
    vector<double>::iterator node_ptr = find(degree_array.begin(), degree_array.end(), min_val);

    //if element is found in degree_array
    if (node_ptr != degree_array.end())
    {
        //get root node numer from degree array
        node = distance(degree_array.begin(), node_ptr); 
        // insert the root into result array
        R.push_back(node);

        //clear the minimum value
        temp_degree.erase(min_val_ptr);
    }
    
    // find all the neighbour of a root node and push into Queue
    for (int val_index = this->row_position[node]; val_index < this->row_position[node + 1]; val_index++)
    {
        //We want to insert only neighbour, so exclude oneself
        if (node != this->col_index[val_index])
        {
            // if the neighbour node is not in vector R, then we can add the value into R and add its neighbour to Q
            // Remember to erase the pushed-to-Q value from temp_array
            if (find(R.begin(), R.end(), this->col_index[val_index]) == R.end())
            {
                neigh_col_idx = this->col_index[val_index];
                Q.push(neigh_col_idx);
                temp_degree.erase(find(temp_degree.begin(), temp_degree.end(), degree_array[neigh_col_idx]));
            }
        }
    }
    bool first_loop = true;
    int child_node = Q.front();
    
    //====================================================================================================
    // Buggy section begins here...
    //====================================================================================================

    // The following commentted section shows error when deque. There seems to be a logic problem within 
    // the while loop. The while loop can only run once and it will be broken by bugs .

    /*
    while(!Q.empty())
    {
        if(find(R.begin(), R.end(), child_node) == R.end())// if the element is not in R
        {
            if (first_loop)
            {
                // Take away the first element in Q 
                
                Q.pop();
                //put the element into R
                first_loop = false;
            }
            R.push_back(child_node);

            // find all the neighbour of a root node and push into Queue
            for (int val_index = this->row_position[child_node]; val_index < this->row_position[child_node + 1]; val_index++)
            {
                //We want to insert only neighbour, so exclude oneself
                if (child_node != this->col_index[val_index])
                {
                    // if the neighbour node is not in vector R, then we can add the value into R and add its neighbour to Q
                    // Remember to erase the pushed-to-Q value from temp_array
                    if (find(R.begin(), R.end(), this->col_index[val_index]) == R.end())
                    {
                        neigh_col_idx = this->col_index[val_index];
                        Q.push(neigh_col_idx);
                        temp_degree.erase(find(temp_degree.begin(), temp_degree.end(), degree_array.at(neigh_col_idx)));
                    }
                }
            }
        }
        if(!Q.empty())
        {
        child_node = Q.front();
        Q.pop();
        }
    }
    */

    //====================================================================================================
    // Buggy session ends here...
    //====================================================================================================

    // if vector R size is not the same as matrix's row size, it means there are still nodes remained unexamined.
    // call the function itself to form recursive algorithm if there are still nodes remain.
    if (R.size() != this->rows)
    {
        Cuthill_Mckee(degree_array);
    }

}

// Reverse the order of element in vector R
template <class T>
void BandedMatrix<T>::reverse_R()
{
    cout << endl << "Given permutation array R: ";
    for (auto i : R)
        cout << i << ' ';
    cout << endl;

    vector<double>::iterator first = R.begin();
    vector<double>::iterator last = R.end();
    // reverse order
    while ((first != last) && (first != --last))
        iter_swap(first++, last);

    cout <<endl<< "Reversed permutation array R: " ;
    for (auto i : R)
        cout << i << ' ';
    cout << endl;
}

// Use permutation array 'R' to permutate original CSR matrix to a banded matrix
template <class T>
void BandedMatrix<T>::permutate_to_Band()// CSRMatrix<T>& mat, vector<T> R
{
    //Define a new CSR matrix for executing permutation
    auto* new_mat = new CSRMatrix<double>(this->rows, this->cols, this->nnzs, true);

    // Initialisation of new value index and neighbours of a node( this is basically the colume index)
    int new_val_index = 0;
    int neighbour = 0, new_neighbour = 0;
    int round = 0;
    for (int i = 0; i < R.size(); i++)
    {
        //New CSR matrix row position is filled 
        new_mat->row_position[i] = new_val_index;

        // define a new set to store values and correspoinding coloums, because we want to sort the columns in ascending order
        set<pair<T, T> > set_Col_Val;

        for (int val_index = this->row_position[(int)R[i]]; val_index < this->row_position[(int)R[i] + 1]; val_index++)
        {
            // The neighbours of a node is defined as the column indices of non-zero numbers at current row
            neighbour = this->col_index[val_index];

            // Find the new neighbours of the node by look up the position of current column in R vector
            auto it = find(R.begin(), R.end(), neighbour);
            new_neighbour = distance(R.begin(), it);

            // Create a set for sorting te columns
            set_Col_Val.insert(make_pair(new_neighbour, this->values[val_index]));

        }
        for (auto&& element : set_Col_Val) {
            // the new non-zero value is the neighbours
            new_mat->values[new_val_index] = element.second;
            // the new column index is equal to the neighbours of the original nodes
            new_mat->col_index[new_val_index] = element.first;
            new_val_index++;
        }
    }
    // Fill in the last element of row position for the new matrix. Remeber that 
    // the size of row_position is row number plus 1.
    new_mat->row_position[(int)this->rows] = new_val_index;
    cout << endl << "=================================Print Banded Matrix================================"<<endl<<endl;
    new_mat->printMatrix();
    delete new_mat;
}

// Run the Reverse_Cuthill_Mckee algorithm
template <class T>
void BandedMatrix<T>::Run_Reverse_Cuthill_Mckee()
{
    vector<double> deg_arr = this->get_degree();
    // Make a copy of degree array. 
    // If any node is added to vector R, we will erase that element in temp_degree array(this is a private vector),
    // but keep deg_arr the same to track the correct node position.
    for (auto deg : deg_arr)
    {
        temp_degree.push_back(deg);
    }

    Cuthill_Mckee(deg_arr); // produce permutation array R

    reverse_R();// Reverse element order in vector R

    permutate_to_Band(*this,R); // print out permutated CSR matrix. A banded matrix should be displayed on command window
    

}

