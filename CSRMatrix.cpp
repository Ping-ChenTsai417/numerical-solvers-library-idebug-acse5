#include <iostream>
#include <algorithm>
#include <math.h>
#include "CSRMatrix.h"

// Constructor - using an initialisation list here
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate): Matrix<T>(rows, cols, false), nnzs(nnzs)
{
   // If we don't pass false in the initialisation list base constructor, it would allocate values to be of size
   // rows * cols in our base matrix class
   // So then we need to set it to the real value we had passed in
   this->preallocated = preallocate;

   // If we want to handle memory ourselves
   if (this->preallocated)
   {
      // Must remember to delete this in the destructor
      this->values = new T[this->nnzs];
      this->row_position = new int[this->rows+1];
      this->col_index = new int[this->nnzs];
   }
}

// Constructor - now just setting the value of our T pointer
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_position, int *col_index): Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_position), col_index(col_index)
{}

// Destructor
template <class T>
CSRMatrix<T>::~CSRMatrix()
{
   // Delete the values array
   if (this->preallocated){
      delete[] this->row_position;
      delete[] this->col_index;
   }
   // The super destructor is called after we finish here
   // This will delete this->values if preallocated is true
}

// Explicitly print out the values in values array as if they are a matrix
template <class T>
void CSRMatrix<T>::printMatrix() 
{ 
   std::cout << "Printing matrix" << std::endl;
   std::cout << "Values: ";
   for (int j = 0; j< this->nnzs; j++)
   {  
      std::cout << this->values[j] << " ";      
   }
   std::cout << std::endl;
   std::cout << "row_position: ";
   for (int j = 0; j< this->rows+1; j++)
   {  
      std::cout << this->row_position[j] << " ";      
   }
   std::cout << std::endl;   
   std::cout << "col_index: ";
   for (int j = 0; j< this->nnzs; j++)
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

// Do matrix matrix multiplication
// output = this * mat_right
template <class T>
void CSRMatrix<T>::matMatMult(CSRMatrix<T>& mat_right, CSRMatrix<T>& output)
{
   // Check our dimensions match
   if (this->cols != mat_right.rows)
   {
      std::cerr << "Input dimensions for matrices don't match" << std::endl;
      return;
   }

   // Check if our output matrix has had space allocated to it
   if (output.values != nullptr) 
   {
      // Check our dimensions match
      if (this->rows != output.rows || mat_right.cols != output.cols)
      {
         std::cerr << "Input dimensions for matrices don't match" << std::endl;
         return;
      }      
   }
   // The output hasn't been preallocated
   else
   {
	   std::cerr << "Output haven't been created" << std::endl;
   }
   T tmp= 0;

   T tmp= 0;
   int v1_index;
		//loop through each row in left matrix
       for (int i = 0; i < this->rows; i++)
       {
		   //loop through each column in the right matrix
           for (int j = 0; j < mat_right.cols; j++)
           {
               tmp = 0;
			   //check where are the non-zero values in each row of the left matrix
                for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
                {   
					//for each found non-zero value in the current row (for the left matrix) 
					//check if there is a corresponding value to multiply in the right matrix
                    v1_index = mat_right.Get_Value_Index(this->col_index[val_index],  j);
                    if (v1_index == -1)
                    {

                    }
                    else
                    {//if there is a corresponding value, multiply them together and add to temp
                        tmp = tmp + this->values[val_index] * mat_right.values[v1_index];
                    }

                }
				//write out the value to the vector of output
                output.setvalues(tmp, i, j);
           }
        }
}

// Do a matrix-vector product
// output = this * input
template<class T>
void CSRMatrix<T>::matVecMult(Matrix<T>& input, Matrix<T>& output)
{
    if (input.values == nullptr || output.values == nullptr)
    {
        std::cerr << "Input or output haven't been created" << std::endl;
        return;
    }


    int val_counter = 0;
    // Loop over each row
    for (int i = 0; i < this->rows; i++)
    {
        // Loop over all the entries in this col
        for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
        {
            // This is an example of indirect addressing
            // Can make it harder for the compiler to vectorise!
            output.values[i] += this->values[val_index] * input.values[this->col_index[val_index]];

        }
    }
}


template <class T>
void CSRMatrix<T>::setvalues(T value, int i, int j)//inserts value to the row i and column j (thinking in dense matrix format)
{
    //Store value at which we should insert new element to the values array
    int insert_at_index;
    //Flag to tell us if we found a place to insert a new element
    bool not_found = true;
    //If there are no elements in a row ie. before and after given row num of elements is the same we immiedietely know the index at which we should insert
    //the value onto the values array, it will just be to the row_position[i]
    //we now that number of nnzv before the value being inserted is equal to row_position[i]
    if (this->row_position[i] == this->row_position[i + 1])
    {
        insert_at_index = row_position[i];
    }
    else
    {
        //Otherwise we have to iterate over elements in a given row and compare it to the column we want to insert
        //so bascially we will check if the other value/values in the same row are before or after our value
        for (int counter = this->row_position[i]; counter < this->row_position[i + 1]; counter++)
        {
            //If there is already element in a given column just replace it with different value
            if (this->col_index[counter] == j)
            {
                this->values[counter] = value;
                return;
            }
            //If we went past the column that we want to insert to, we know the position that we should insert
            if (this->col_index[counter] > j)
            {
                insert_at_index = counter;
                not_found = false;
                break;
            }
        }

        //If our element would be the last in the given row, we again know at which position to insert value
        if (not_found)
        {
            insert_at_index = this->row_position[i + 1];
        }
    }

    //Create new arrays, one larger since we will have one more element
    //we need to change 3 things
    //array of values
    //array of columns indexes
    //array of row_positions
    auto tmp_values = new T[this->nnzs + 1];
    auto tmp_col_index = new int[this->nnzs + 1];

    //Account for the additional element in all row positions
    //so increase number of nnzv present in each row after the row to which we will be inserting
    for (int L = i + 1; L < this->rows + 1; L++)
    {
        this->row_position[L]++;
    }

    //We want to insert element at given position. If index is smaller than that positon values stay the same. For index equal to that position we insert element. All elements larger than index will shift one to the right
    for (int counter = 0; counter < this->nnzs + 1; counter++)
    {
        if (counter < insert_at_index)//if our value goes after this one nothing happens
        {
            tmp_values[counter] = this->values[counter];
            tmp_col_index[counter] = this->col_index[counter];
        }
        else if (counter == insert_at_index)//here we insert our value
        {
            tmp_values[counter] = value;
            tmp_col_index[counter] = j;
        }
        else//here we shift all values that come after our inserted value
        {
            tmp_values[counter] = this->values[counter - 1];
            tmp_col_index[counter] = this->col_index[counter - 1];
        }
    }
    //Update arrays of values and col_index as well as increase number of elements.
    auto old_values = this->values;
    auto old_col_index = this->col_index;

    this->nnzs++;
    this->values = tmp_values;
    this->col_index = tmp_col_index;

    delete old_values;
    delete old_col_index;

    return;
}

//if non-zero value is present returns an index in the array of values of the 
//non-zero value at a given row and column, otherwise returns -1
template <class T>
int CSRMatrix<T>::Get_Value_Index(int row,int column)
{
	
		//check if there is a non zero value in the given row
		//if there isn't then this->row_position[row] and this->row_position[row+1] will be the same 
		//so the loop won't execute
        for (int val_index = this->row_position[row]; val_index < this->row_position[row + 1]; val_index++)
        {	//if there is a nnzv and it happens to be at the given column
			//its index will be equal to val_index
            if (this->col_index[val_index] == column)
            {
                return val_index;
            }
			//if the value wasn't find in given column, return -1
            else if (this->col_index[val_index] > column)
            {
                return -1;
            }
        }
    return -1;

}

//calls the corresponding matrix solver, specify the type of solver by typing in an enum "type of solver"
template <class T>
void CSRMatrix<T>::solve( Matrix<T>& vect, Matrix<T>& vect_output, int type_of_solver)
{
    if (type_of_solver == Jacobi_CSR)
    {
        this->Jacobi_CSR_Solver(vect, vect_output);
    }
    else if (type_of_solver == Gauss_Siedel_CSR)
    {
        this->Gauss_Seidel_CSR_Solver(vect, vect_output);
    }
    else if (type_of_solver == Conjugate_Gradient_CSR)
    {
        this->Conjugate_Gradient_CSR_Solver(vect, vect_output);
    }
    else if (type_of_solver == Cholesky_CSR)
    {
        std::cout << "Fight to implement this continues" << "\n";
        //this->Cholesky_CSR_Solver(vect, vect_output);
    }
    else
    {
        std::cerr << "Please enter a valid type of solver!!!!!" << std::endl;
        return;
    }

}

// Solver method 1 
template <class T>
void CSRMatrix<T>::Jacobi_CSR_Solver(Matrix<T>& vect, Matrix<T>& vect_output)
{
    bool big_error = true;
    auto* vector_new = new T[vect_output.rows];

    // set the output vector to zeros, just in case
    for (int i = 0; i < vect_output.rows * vect_output.cols; i++)
    {
        vect_output.values[i] = 0;
    }

    T temp = 0;
    int inter;
    int lp = 0;
    while (big_error)
    {
        lp++;
        //std::cout << "Loop no " << lp << std::endl;
        for (int i = 0; i < this->rows; i++)
        {
            temp = 0;
            for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
            {
                // if (i != this->col_index[val_index])
               //  {
                     //assuming row-order
                     //inter = this->Get_Value_Index(i, this->col_index[val_index]);
                     //if (inter != -1)
                     //{
                     //    temp += this->values[inter] * vect_output.values[this->col_index[val_index]];
                     //}
                if (i != this->col_index[val_index])
                {
                    temp += this->values[val_index] * vect_output.values[this->col_index[val_index]];

                }
                // }
            }
            vector_new[i] = (vect.values[i] - temp) / this->values[Get_Value_Index(i, i)];
        }
        for (int i = 0; i < this->rows; i++)
        {
            vect_output.values[i] = vector_new[i];
        }
        big_error = check_error_CSR(*this, vect, vect_output);

        if (lp > 10000)
        {
            break;
        }

    }

    delete[] vector_new;

}

// Solver method 2
template <class T>
void CSRMatrix<T>::Gauss_Seidel_CSR_Solver( Matrix<T>& vect, Matrix<T>& vect_output)
{
    bool big_error = true;


    T temp = 0;
    int inter;
    int lp = 0;
    while (big_error)
    {
        lp++;
        for (int i = 0; i < this->rows; i++)
        {
            temp = 0;
			//loop through each row and check if there are non-zero values there
			//if there are, loop through each of them
            for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
            {
                if (i != this->col_index[val_index])
                {
                    temp += this->values[val_index] * vect_output.values[this->col_index[val_index]];
                   
                }
            }
            vect_output.values[i] = (vect.values[i] - temp) / this->values[Get_Value_Index(i,i)];
        }
        big_error = check_error(*this, vect, vect_output);
        if (lp > 10000)
        {
            break;
        }
        //vect_output.printValues();
    }
}

// Solver method 3
template <class T>
void CSRMatrix<T>::Conjugate_Gradient_CSR_Solver(const Matrix<T>& vect, Matrix<T>& vect_output)
{
    //double tol = 0.00000001;

    // Initialize p, Ap and residual r 
    //Ap is inner product output of LHS matrix and p. Use CSR::matVectMult(...) for Ap.
    auto* r = new Matrix<double>(vect.rows, vect.cols, true); // r = vect
    auto* p = new Matrix<double>(vect.rows, vect.cols, true); // p = vect
    auto* Ap = new Matrix<double>(vect.rows, vect.cols, true); // Ap = [0, 0, 0] initially

    //ASK: how can I mar r equals vect more efficiently?

    for (int j = 0; j < vect.rows; j++)
    {
        r->values[j] = vect.values[j];
        p->values[j] = vect.values[j];
    }

    //Set count for counting loops
    int count = 0;

    //Loop stops either if the error is smaller than tolerance (already set in Matrix class), or if loops over 1000 times
    while (count < 1000)//check_error(*this, vect, vect_output) || 
    {
        //reset mat*Vect product every new loop
        for (int j = 0; j < vect.rows; j++)
        {
            Ap->values[j] = 0;
        }
        auto* r_old = r;

        this->matVecMult(*p, *Ap); // calculate Ap

        double alpha = this->innerProduct(*r, *r) / std::max(this->innerProduct(*p, *Ap), this->near_zero);

        vect_output.CG_CombineVector(1, alpha, *p);// Set gradient to 1 
        r->CG_CombineVector(1, -alpha, *Ap);

        if (sqrt(this->innerProduct(*r, *r)) < this->tol) // check if residual reach tolerance
            break;

        double beta = this->innerProduct(*r, *r) / std::max(this->innerProduct(*r_old, *r_old), this->near_zero);

        // After first loop , let p = r
        if (count != 0)
        {
            for (int i = 0; i < r->rows; i++)
            {
                p->values[i] = r->values[i];
            }
        }
        p->CG_CombineVector(1, beta, *p);

        count++;
    }

    //Print out final solution
    // vect_output.printMatrix();

    delete r;
    delete p;
    delete Ap;

}

// Solver method 4
template <class T>
void CSRMatrix<T>::Cholesky_CSR_Solver( Matrix<T>& vect, Matrix<T>& vect_output)
{
    //declare an array where values of the lower triangular matrix will be stored
    auto* Lower = new CSRMatrix <T>(this->rows, this->cols, 0, true);

    for (int i = 0; i < Lower->rows + 1; i++)
    {
        Lower->row_position[i] = 0;
    }
    //and initialize it to zeros
    //this->printMatrix();
    //lets decompose our matrix (this) into lower triangular matrix
    for (int r = 0; r < this->rows; r++)
    {
        for (int val_index = this->row_position[r]; val_index < this->row_position[r + 1]; val_index++)
        {

            int c = this->col_index[val_index];
            if (c < r + 1)
            {
                if (c == r)
                {//finding the values for elements on diagonal
                    T sum = 0;
                    for (int new_val_index = this->row_position[r]; new_val_index < val_index; new_val_index++)
                    {
                        int j = this->col_index[new_val_index];

                        if (Lower->Get_Value_Index(c, j) != -1)
                        {
                            sum += Lower->values[Lower->Get_Value_Index(c, j)] * Lower->values[Lower->Get_Value_Index(c, j)];
                        }
                    }
                    T value = sqrt(this->values[this->Get_Value_Index(c, c)] - sum);
                    Lower->setvalues(value, r, r);
                }//now using the values from diagonals we can find other values
                else
                {
                    T sum = 0;
                    int Max_Condition;
                    if (this->col_index[val_index] < this->col_index[val_index + 1])
                    {
                        Max_Condition = this->col_index[val_index + 1];
                    }
                    else
                    {
                        Max_Condition = this->rows;
                    }
                    std::cout << "Max COnditiion " << Max_Condition << "\n";
                    for (int third_val_index = c+1; third_val_index < Max_Condition; third_val_index++)
                    {

                        int new_c = third_val_index;
                        for (int new_val_index = Lower->row_position[r]; new_val_index < Lower->row_position[r + 1]; new_val_index++)
                        {
                            int j = Lower->col_index[new_val_index];
                            std::cout << "Value index of Lower->Get_Value_Index(r, j) is " << Lower->Get_Value_Index(r, j) << "\n";
                            std::cout << "Value index of Lower->Lower->Get_Value_Index(c, j) is " << Lower->Get_Value_Index(third_val_index, j) << "\n";

                            if (Lower->Get_Value_Index(r, j) != -1 and Lower->Get_Value_Index(third_val_index, j) != -1)
                            {
                                T Z = Lower->values[Lower->Get_Value_Index(r, j)];
                                T Q = Lower->values[Lower->Get_Value_Index(third_val_index, j)];
                                std::cout << "For value of row =" << r << " and col=" << c << " Z=" << Z << " Q=" << Q << "\n";
                                sum += Lower->values[Lower->Get_Value_Index(r, j)] * Lower->values[Lower->Get_Value_Index(third_val_index, j)];
                            }

                        }
                    }
                    T X = this->values[val_index];
                    T YN = Lower->values[Lower->Get_Value_Index(c, c)];
                    std::cout << "For value of row =" << r << " and col=" << c << " X=" << X << " YN=" << YN << "\n";
                    T value = (this->values[val_index] - sum) / Lower->values[Lower->Get_Value_Index(c, c)];
                    Lower->setvalues(value, r, c);
                }
            }

        }
    }
    //Lower->printMatrix();


    auto* y = new T[this->rows];
    T s;
    y[0] = vect.values[0];
    for (int r = 0; r < this->rows; r++)
    {
        s = 0;
        for (int val_index = Lower->row_position[r]; val_index < Lower->row_position[r+1]; val_index++)
        {
            int c = Lower->col_index[val_index];
            if (c < r)
            {
                s = s + Lower->values[val_index]  * y[c];
            }
            else
            {
                break;
            }
        }
        y[r] = (vect.values[r] - s) / Lower->values[Lower->Get_Value_Index(r,r)];
    }

    //for (int i = 0; i < this->rows; i++)
    //{
    //    std::cout << y[i] << " ";
    //}
    //std::cout << "\n";

    auto* Lower_Transpose = new CSRMatrix <T>(this->rows, this->cols, 0, true);
    for (int i = 0; i < Lower->rows + 1; i++)
    {
        Lower_Transpose->row_position[i] = 0;
    }

    Lower->Fast_CSR_Transpose(*Lower_Transpose);

    bool Debug = true;
    if (Debug)
    {
        auto* Lower_Dense = new Matrix <T>(this->rows, this->cols,  true);
        auto* Lower_Transpose_Dense = new Matrix <T>(this->rows, this->cols,  true);
        auto* Input_Dense = new Matrix <T>(this->rows, this->cols,  true);
        this->Convert_CSRMatrix_To_Matrix(*Input_Dense);
        Lower->Convert_CSRMatrix_To_Matrix(*Lower_Dense);
        Lower_Transpose->Convert_CSRMatrix_To_Matrix(*Lower_Transpose_Dense);
        auto* Test_Matrix = new Matrix <T>(this->rows, this->cols,  true);
        Lower_Dense->matMatMult(*Lower_Transpose_Dense, *Test_Matrix);
        T sum = 0;
        for (int i = 0; i < Test_Matrix->rows * Test_Matrix->cols; i++)
        {
            sum = sum + abs(Test_Matrix->values[i] - Input_Dense->values[i]);
        }
        std::cout << "=================================" << "\n";
        std::cout << "Difference in factorization is : " << sum << "\n";
        std::cout << "=================================" << "\n";

        delete Lower_Dense;
        delete Lower_Transpose_Dense;
        delete Input_Dense;
        delete Test_Matrix;
    }
    
    //Lower_Transpose->printMatrix();
    //vect.printMatrix();
    // Perform the back substitution to get x from y=L^T*x
    for (int r = this->rows - 1; r >= 0; r--) //i loop from the bottom to the top row
    {
        T sum = 0;
        for (int val_index = Lower_Transpose->row_position[r]; val_index < Lower_Transpose->row_position[r + 1]; val_index++)
        {
            int c = Lower_Transpose->col_index[val_index];
            if (c > r)
            {
                // sum the mat-vec product for all the columns on the right of pivot
                sum += Lower_Transpose->values[val_index] * vect_output.values[c];
                //sum += lower[c * this->cols + r] * vect_output.values[c];
            }
        }
        vect_output.values[r] = (y[r] - sum) / Lower_Transpose->values[Lower_Transpose->Get_Value_Index(r, r)];
    }
    //vect_output.printMatrix();
    //delete[] lower;
    delete Lower;
    delete Lower_Transpose;
    delete[] y;

}

template <class T>
void CSRMatrix<T>::Convert_CSRMatrix_To_Matrix(Matrix<T>& M)
{
	//loop through each row in CSRmatrix
    for (int i = 0; i < this->rows; i++)
    {
		//loop through each column in CSRmatrix
        for (int j = 0; j < this->cols; j++)
        {
			//initially write out all the values to be zero
            M.values[i * this->rows + j] = 0;
        }
		//now loop through all the non-zero values at this row (if there are any)
		//and write out their values
        for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
        {
            M.values[i * this->rows + this->col_index[val_index]] = this->values[val_index];
        }
    }
}

template <class T>
void CSRMatrix<T>::Convert_Matrix_To_CSR_Matrix(Matrix<T>& M)
{
	//loop through each row in normal matrix
    for (int i = 0; i < this->rows; i++)
    {
		//loop through each column in normal matrix
        for (int j = 0; j < this->cols; j++)
        {
			//if the value is non zero, use the setvalue function to 
			//add it to the CSRMatrix
            if (M.values[i * this->rows + j]!=0)
            {
                this->setvalues(M.values[i * this->rows + j], i, j);
            }
        }
    }

}

template <class T>
void CSRMatrix<T>::Fast_CSR_Transpose(CSRMatrix<T>& Output)
{
    for (int i = 0; i < this->rows; i++)
    {
        for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
        {
            Output.setvalues(this->values[val_index], this->col_index[val_index], i);
        }
    }
}

template <class T>
bool check_error_CSR(CSRMatrix<T>& mat, const Matrix<T>& vect, Matrix<T>& vect_output)
{
    T value = 0;
    for (int i = 0; i < mat.rows; i++)
    {
        value = 0;
        // Loop over all the entries in this col
        for (int val_index = mat.row_position[i]; val_index < mat.row_position[i + 1]; val_index++)
        {
            value += mat.values[val_index] * vect_output.values[mat.col_index[val_index]];

        }
        if (abs(value - vect.values[i]) > 0.0001)
        {
            //as soon as one entry has too big error return true
            //so that gauss seidel would continue
            return true;
        }
    }
    return false;
}


