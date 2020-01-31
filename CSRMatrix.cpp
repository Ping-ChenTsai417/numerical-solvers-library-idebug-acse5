#include <iostream>
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
						//if there isn't non zero value, do nothing because adding 0 would not change anything
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
void CSRMatrix<T>::solver(const Matrix<T>& vect_b, Matrix<T>& vect_output, int type_of_solver)
{

	//// Make sure that all dimensions agree
	// Check if our output matrix has had space allocated to it
	if (vect_output.values != nullptr)
	{
		// Check our dimensions match
		if (this->rows != vect_output.rows || vect_b.cols != vect_output.cols || vect_b.cols != 1)
		{
			std::cerr << "Input dimensions for matrices don't match" << std::endl;
			return;
		}
	}
	// The output hasn't been preallocated, so we are going to do that
	else
	{
		vect_output.values = new T[this->rows];
	}
	// Set values to zero before hand in case user didn't do it
	for (int i = 0; i < vect_output.rows*vect_output.cols ; i++)
	{
		vect_output.values[i] = 0;
	}

    if (type_of_solver == Jacobi_CSR)
    {
    this->Jacobi_CSR_Solver(vect_b, vect_output);
    }
    else if (type_of_solver == Gauss_Siedel_CSR)
    {
        this->Gauss_Seidel_CSR_Solver(vect_b, vect_output);
    }
    else if (type_of_solver == Conjugate_Gradient_CSR)
    {
        this->Conjugate_Gradient_CSR_Solver(vect_b, vect_output);
    }
    else if (type_of_solver == Cholesky_CSR)
    {
        this->Cholesky_CSR_Solver(vect_b, vect_output);
    }
    else
    {
        std::cerr << "Please enter a valid type of solver!!!!!" << std::endl;
        return;
    }

}

// Solver method 1 
template <class T>
void CSRMatrix<T>::Jacobi_CSR_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
    bool big_error = true;
    auto* vector_new = new T[vect_output.rows];

    // set the output vector to zeros, just in case
    for (int i = 0; i < vect_output.rows * vect_output.cols; i++)
    {
        vect_output.values[i] = 0;
    }

    T temp = 0;
    int lp = 0;
    while (big_error)
    {
        lp++;
        //For each of the rows in the matrix
        for (int i = 0; i < this->rows; i++)
        {
            temp = 0;
            //Iterate over all values in the row
            for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
            {
                // If we are not on diagonal
                if (i != this->col_index[val_index])
                {
                    //Sum the products for each row
                    temp += this->values[val_index] * vect_output.values[this->col_index[val_index]];

                }
            }
            // Assign the value to temporary vector
            vector_new[i] = (vect_b.values[i] - temp) / this->values[Get_Value_Index(i, i)];
        }
        //Copy content of temporary vector to output vector
        for (int i = 0; i < this->rows; i++)
        {
            vect_output.values[i] = vector_new[i];
        }
        //Check for convergence
        big_error = check_error_CSR(*this, vect_b, vect_output);

        if (lp > 10000)
        {
            break;
        }

    }

    delete[] vector_new;

}

// Solver method 2
template <class T>
void CSRMatrix<T>::Gauss_Seidel_CSR_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
    //Algorithm identical to Solver Method 1 with the only difference being the fact that as the values are updated they are also used in calculation of subsequent values
    //This results in more robust solver, which has slightly less constraining conditions of convergence
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
            vect_output.values[i] = (vect_b.values[i] - temp) / this->values[Get_Value_Index(i,i)];
        }
        big_error = check_error_CSR(*this, vect_b, vect_output);
        if (lp > 10000)
        {
            break;
        }
    }
}

// Solver method 3
template <class T>
void CSRMatrix<T>::Conjugate_Gradient_CSR_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{

    // Initialize p, Ap and residual r 
    //Ap is inner product output of LHS matrix and p. Use CSR::matVectMult(...) for Ap.
    auto* r = new Matrix<double>(vect_b.rows, vect_b.cols, true); // r = vect
    auto* p = new Matrix<double>(vect_b.rows, vect_b.cols, true); // p = vect
    auto* Ap = new Matrix<double>(vect_b.rows, vect_b.cols, true); // Ap = [0, 0, 0] initially


    for (int j = 0; j < vect_b.rows; j++)
    {
        r->values[j] = vect_b.values[j];
        p->values[j] = vect_b.values[j];
    }

    //Set count for counting loops
    int count = 0;

    //Loop stops either if the error is smaller than tolerance (already set in Matrix class), or if loops over 1000 times
    while (count < 1000)//check_error(*this, vect, vect_output) || 
    {
        //reset mat*Vect product every new loop
        for (int j = 0; j < vect_b.rows; j++)
        {
            Ap->values[j] = 0;
        }
        auto* r_old = r;

        this->matVecMult(*p, *Ap); // calculate Ap

        double alpha = this->innerProduct(*r, *r) / ((this->innerProduct(*p, *Ap) > this->near_zero) ? this->innerProduct(*p, *Ap) : this->near_zero);

        vect_output.CG_CombineVector(1, alpha, *p);// Set gradient to 1 
        r->CG_CombineVector(1, -alpha, *Ap);

        if (sqrt(this->innerProduct(*r, *r)) < this->tol) // check if residual reach tolerance
            break;

        double beta = this->innerProduct(*r, *r) / ((this->innerProduct(*p, *Ap) > this->near_zero )? this->innerProduct(*p, *Ap) : this->near_zero);

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


    delete r;
    delete p;
    delete Ap;

}

// Solver method 4
template <class T>
void CSRMatrix<T>::Cholesky_CSR_Solver(const Matrix<T>& vect, Matrix<T>& vect_output)
{
    //In this method we solve the linear system in the form of A*x=b, in following steps:
    // 1. Find matrix L such that L*L' (L transpose ) = A
    // 2. Define L'* x as y and solve L*y=b for y by forward substitution
    // 3. Calculate L' (not trivial since we are using CSR matrix)
    // 4. Knowing y and L' solve L'x = y for x
    //Declare containers to store lower diagonal matrix, 
    auto* L_Values =      new T[this->rows*this->rows];
    auto* L_Row_Position = new int[this->rows+1];
    auto* L_Col_Index =    new int[this->rows * this->rows];
    auto* L_Diagonal = new T[this->rows];

    //Initialize row positions of the lower diagonal matrix to 0
    for (int i = 0; i < this->rows + 1; i++)
    {
        L_Row_Position[i] = 0;
    }
    int elem;
    int elem_to_push=0;
    // 1. Find matrix L
    //For each row of the input matrix
    for (int r = 0; r < this->rows; r++)
    {
        //Track which element you are about to access
        elem = this->row_position[r];
        
        for (int cg = 0; cg <= r; cg++)
        {
                if (cg == r)
                {//finding the values for elements on diagonal
                    T sum = 0;
                    for (int val_index = L_Row_Position[r]; val_index < L_Row_Position[r+1]; val_index++)
                    {
                        sum += L_Values[val_index] * L_Values[val_index];
                    }
                    T value = sqrt(this->values[elem] - sum);
                    //Update L matrix, knowing that every time you add new element it will be at the end of the matrix
                    L_Diagonal[cg] = value;
                    L_Values[elem_to_push] = value;
                    L_Col_Index[elem_to_push] = cg;
                    for (int i = r + 1; i < this->rows + 1; i++)
                    {
                        L_Row_Position[i]++;

                    }
                    elem_to_push++;
                }//fidning other values of the L matrix
                else
                {
                    T sum = 0;
                    //Optimized way to find sum of L[col][i]*L[row][i] for all i<col
                    for (int cr = L_Row_Position[r]; cr < L_Row_Position[r+1]; cr++)
                    {
                        if (L_Col_Index[cr] >= cg)
                        {
                            break;
                        }
                        //else
                        {
                            for (int cc = L_Row_Position[cg]; cc < L_Row_Position[cg+1]; cc++)
                            {
                                if (L_Col_Index[cc] > L_Col_Index[cr])
                                {
                                    break;
                                } 
                                else if (L_Col_Index[cc] == L_Col_Index[cr])
                                {
                                    sum = sum - L_Values[cr] * L_Values[cc];
                                }
                            }
                        }
                    }

                    //Check if element at given column exist in the input matrix
                    if (this->col_index[elem] == cg)
                    {
                        sum = sum + this->values[elem];
                        elem++;
                    }
                    //If sum is different than 0 update the L matrix
                    if (sum != 0)
                    {
                        //In order to make this more optimized we store values of the diagonal 
                        T value = sum / L_Diagonal[cg];
                        
                        L_Values[elem_to_push] = value;
                        L_Col_Index[elem_to_push] = cg;
                        for (int i = r + 1; i < this->rows+1; i++)
                        {
                            L_Row_Position[i]++;
                        }
                        elem_to_push++;
                    }
            }
        }
    }
    
    auto* y = new T[this->rows];
    T s;
    y[0] = vect.values[0];
    //2 Find y by forward substitution
    for (int r = 0; r < this->rows; r++)
    {
        s = 0;
        for (int val_index = L_Row_Position[r]; val_index < L_Row_Position[r+1]; val_index++)
        {
            int c = L_Col_Index[val_index];
            if (c < r)
            {
                s = s + L_Values[val_index]  * y[c];
            }
            else
            {
                break;
            }
        }
        y[r] = (vect.values[r] - s) / L_Diagonal[r];
    }

    

    auto* Lower_Transpose = new CSRMatrix <T>(this->rows, this->cols, L_Row_Position[this->rows], true);
    auto* values_in_row = new int[this->rows];
    auto* values_in_col = new int[this->rows];
    // 3 Transpose of the matrix in an optimized way
    // Initialize row_positions of lower tranpose to 0
    for (int i = 0; i < this->rows + 1; i++)
    {
        Lower_Transpose->row_position[i] = 0;
    }
    //Initialize two temporary arrays which help in calculating transpose
    for (int i = 0; i < this->rows; i++)
    {
        
        values_in_row[i] = 0;
        values_in_col[i] = 0;
    }
    // Calculate number of occurences in each column
    for (int i = 0; i < L_Row_Position[this->rows]; i++)
    {
        values_in_col[L_Col_Index[i]]++;
    }
    // Based on that update Lower_Transpose row positions 
    for (int i = 1; i < this->rows + 1; i++)
    {
        Lower_Transpose->row_position[i] = Lower_Transpose->row_position[i-1]+ values_in_col[i-1];
    }
    //Knowing row positions iterate over all values and assign them in correct order to the vector of values of tranposed matrix
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = L_Row_Position[i]; j < L_Row_Position[i + 1]; j++)
        {


            Lower_Transpose->values[Lower_Transpose->row_position[L_Col_Index[j]] + values_in_row[L_Col_Index[j]]] = L_Values[j];
            
            Lower_Transpose->col_index[Lower_Transpose->row_position[L_Col_Index[j]] + values_in_row[L_Col_Index[j]]] = L_Col_Index[j];
            values_in_row[L_Col_Index[j]]++;
        }
    }
    
    delete[] values_in_row;
    delete[] values_in_col;
    //4 Finally use back substituion on transpoed vector 
    for (int r = this->rows - 1; r >= 0; r--) //i loop from the bottom to the top row
    {
        T sum = 0;
        for (int val_index = Lower_Transpose->row_position[r]; val_index < Lower_Transpose->row_position[r + 1]; val_index++)
        {
            int c = Lower_Transpose->col_index[val_index];
            if (c > r)
            {

                sum += Lower_Transpose->values[val_index] * vect_output.values[c];
 
            }
        }
        vect_output.values[r] = (y[r] - sum) / Lower_Transpose->values[Lower_Transpose->Get_Value_Index(r, r)];
    }

    delete[] L_Values;
    delete[] L_Row_Position;
    delete[] L_Col_Index;
    delete[] L_Diagonal;
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
bool check_error_CSR(CSRMatrix<T>& mat,const Matrix<T>& vect_b, Matrix<T>& vect_output)
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
        if (abs(value - vect_b.values[i]) > 0.000001)
        {
            //as soon as one entry has too big error return true
            //so that gauss seidel would continue
            return true;
        }
    }
    return false;
}


