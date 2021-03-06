#include <iostream>
#include "Matrix.h"
#include <assert.h>
#include <memory>
#include <math.h>

// Constructor - using an initialisation list here
template <class T>
Matrix<T>::Matrix(int rows, int cols, bool preallocate): rows(rows), cols(cols), size_of_values(rows * cols), preallocated(preallocate)
{
   // If we want to handle memory ourselves
   if (this->preallocated)
   {
      // Must remember to delete this in the destructor
      this->values = new T[size_of_values];
   }
}

// Constructor - now just setting the value of our T pointer
template <class T>
Matrix<T>::Matrix(int rows, int cols, T *values_ptr): rows(rows), cols(cols), size_of_values(rows * cols), values(values_ptr)
{}

// Destructor
template <class T>
Matrix<T>::~Matrix()
{
   // Delete the values array
   if (this->preallocated){
      delete[] this->values;
   }
}

// Just print out the values in our values array
template <class T>
void Matrix<T>::printValues() 
{ 
   std::cout << "Printing values" << std::endl;
	for (int i = 0; i< this->size_of_values; i++)
   {
      std::cout << this->values[i] << " ";
   }
   std::cout << std::endl;
}

// Explicitly print out the values in values array as if they are a matrix
template <class T>
void Matrix<T>::printMatrix() 
{ 
   std::cout << "Printing matrix";
   for (int j = 0; j< this->rows; j++)
   {  
      std::cout << std::endl;
      for (int i = 0; i< this->cols; i++)
      {
         // We have explicitly used a row-major ordering here
         std::cout << this->values[i + j * this->cols] << " ";
      }
   }
   std::cout << std::endl;
}

// Do matrix matrix multiplication
// this * mat_right = output
template <class T>
void Matrix<T>::matMatMult(Matrix<T>& mat_right, Matrix<T>& output)
{
	// Check if our dimensions match
	if (this->cols != mat_right.rows)
	{
		std::cerr << "Input dimensions for matrices don't match" << std::endl;
		return;
	}

	// Check if our output matrix has had space allocated to it
	if (output.values != nullptr)
	{
		// Check our dimensions match
		if (this->rows != output.rows)
		{
			std::cerr << "Input dimensions for matrices don't match" << std::endl;
			return;
		}
	}
	// The output hasn't been preallocated, so we are going to do that
	else
	{
		output.values = new T[this->rows * mat_right.cols];
		output.preallocated = true;
	}

	// Set values to zero before hand
	for (int i = 0; i < output.size_of_values; i++)
	{
		output.values[i] = 0;
	}

	for (int i = 0; i < this->rows; i++)
	{
		for (int k = 0; k < this->cols; k++)
		{
			for (int j = 0; j < mat_right.cols; j++)
			{
				output.values[i * output.cols + j] += this->values[i * this->cols + k] * mat_right.values[k * mat_right.cols + j];
			}
		}
	}
}

// Do matrix vector multiplication
// this * mat_right = output
// mat_right is a vector
template <class T>
void Matrix<T>::matVecMult(Matrix<T>& mat_right, Matrix<T>& output)
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
		// Check if our dimensions match
		if (this->rows != output.rows)
		{
			std::cerr << "Input dimensions for matrices don't match" << std::endl;
			return;
		}
	}
	// The output hasn't been preallocated, so we are going to do that
	else
	{
		output.values = new T[this->rows * mat_right.cols];
		output.preallocated = true;
	}

	// Set values to zero before hand
	for (int i = 0; i < output.size_of_values; i++)
	{
		output.values[i] = 0;
	}

	for (int i = 0; i < this->rows; i++)
	{
		for (int k = 0; k < this->cols; k++)
		{
			for (int j = 0; j < mat_right.cols; j++)
			{
				output.values[i * output.cols + j] += this->values[i * this->cols + k] * mat_right.values[k * mat_right.cols + j];
			}
		}
	}
}

// Do matrix vector multiplication
// vect_out = this * vect_in
template <class T>
void Matrix<T>::matVecMult(T* vect_in, T* vect_out)
{
	// Need to ensure the dimension is correct when passing vec_in and vec_out
	// and the pointer is valid to use this function

	for (int i = 0; i < this->rows; i++) // loop over rows of the matrix
	{
		T sum = 0;
		for (int j = 0; j < this->cols; j++) // loop over columns of the matrix
			sum += this->values[i * this->cols + j] * vect_in[j];
		vect_out[i] = sum;
	}
}

// Calculate the vector inner product
// vec1 . vec2 = scalar
template<class T>
double Matrix<T>::innerProduct(Matrix<T>& vec1, Matrix<T>& vec2)
{
	double scalar = 0;
	for (int i = 0; i < vec1.rows; i++)
	{
		scalar += vec1.values[i] * vec2.values[i];
	}
	return scalar;
}

// Calculate determinant of a matrix
// using LU decomposition method
template <class T>
double Matrix<T>::determinant()
{
	// det(A) = det(P) * det(L) * det(U)
	// det upper, U and lower, L triangular matrices are the product of diagonal elements
	// det(L) = 1 always so dont need to compute here
	// det(P) = (-1)^(number of row swap), so dont need to explicitly store P

	// create a copy to store the upper matrix
	auto* upper = new Matrix<T>(this->rows, this->cols, true);
	for (int i = 0; i < this->size_of_values; i++)
		upper->values[i] = this->values[i];

	// store the number of times row-swapping occurs 
	// and product of diagonal values
	int num_swap = 0;
	double product = 1;

	// Calculate the upper triangular matrix
	for (int k = 0; k < this->rows; k++) //k loop over pivot
	{
		//// Implement partial pivoting
		int max_row = k;
		T max_value = upper->values[k * this->cols + k];
		for (int i = k + 1; i < this->rows; i++) //i loop over each row below pivot
		{
			// find row below the pivot with maximum absolute value
			if (fabs(upper->values[i * this->cols + k]) > max_value)
			{
				max_value = upper->values[i * this->cols + k];
				max_row = i;
			}
		}
		// make sure the pivot has max value
		if (max_row != k)
		{
			swap_rows(*upper, k, max_row);
			num_swap++;
		}

		for (int i = k + 1; i < this->rows; i++) //i loop over each row below pivot
		{
			T s = upper->values[i * this->cols + k] / upper->values[k * this->cols + k];
			for (int j = k; j < this->rows; j++)  //j loop over elements for each row
				upper->values[i * this->cols + j] -= s * upper->values[k * this->cols + j];
		}
		product *= upper->values[k * this->cols + k];
	}
	
	double det = pow(-1, num_swap) * product;
	delete upper;

	return det;

	
}


// Calculate the inverse of a matrix
// users need to allocate the memory themselves and pass it as an input argument
// this implement the Gauss-jordan method to calculate the inverse
template <class T>
void Matrix<T>::inverse(Matrix<T>& inverse_mat)
{
	// we dont want to change the original matrix so create a temporary storage
	auto* temp = new Matrix<T>(this->cols, this->rows, true);

	// initialise values in the matrices
	for (int i = 0; i < this->rows; i++)
	{
		for (int j = 0; j < this->cols; j++)
		{
			// initialise temp to be the same as matrix we're gonna solve
			temp->values[i * this->cols + j] = this->values[i * this->cols + j];
			// initialise P to be an identiy matrix
			if (i == j) inverse_mat.values[i * this->cols + j] = 1;
			else inverse_mat.values[i * this->cols + j] = 0;
		}
	}

	// form upper triangular matrix from temp
	for (int k = 0; k < this->rows - 1; k++) //k loop over pivot except the final row
	{
		for (int i = k + 1; i < this->rows; i++) //i loop over each row below pivot
		{
			T s = temp->values[i * this->cols + k] / temp->values[k * this->cols + k];
			for (int j = 0; j < this->rows; j++)  //j loop over elements within each row
			//cannot start iterating j from k (like LU) because we need to change inverse_mat as well
			{
				temp->values[i * this->cols + j] -= s * temp->values[k * this->cols + j];
				// apply the same calculation to the inverse_mat
				inverse_mat.values[i * this->cols + j] -= s * inverse_mat.values[k * this->cols + j];
			}
		}
	}

	// form lower triangular matrix from temp
	// if an upper triangular matirx is passed the result will be a diagonal matrix
	for (int k = this->rows - 1; k >= 0; k--) //k loop over pivot elements from bottom to top
	{
		for (int i = k - 1; i >= 0; i--) //i loop over each row above pivot
		{
			T s = temp->values[i * this->cols + k] / temp->values[k * this->cols + k];
			for (int j = 0; j < this->rows; j++)  //j loop over elements within each row
			{
				temp->values[i * this->cols + j] -= s * temp->values[k * this->cols + j];
				// apply the same calculation to the inverse_mat
				inverse_mat.values[i * this->cols + j] -= s * inverse_mat.values[k * this->cols + j];
			}
		}
	}

	// divide both matrices with the diagonal values of temp
	// ensure diagonal values of temp equals to 1 (temp becomes identity matrix)
	// the inverse_mat is now the inverse of temp matrix
	for (int i = 0; i < this->rows; i++) // loop over all rows
	{
		for (int j = 0; j < this->cols; j++) // loop over all columns
			inverse_mat.values[i * this->cols + j] /= temp->values[i * this->cols + i];
	}

	delete temp;
}

// Main solver which calls each solver method depending on the input type_of_solver
template <class T>
void Matrix<T>::solver(const Matrix<T>& vect_b, Matrix<T>& vect_output, int type_of_solver)
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
		vect_output.preallocated = true;
    }
    // Set values to zero before hand in case user didn't do it
    for (int i = 0; i < vect_output.size_of_values; i++)
    {
        vect_output.values[i] = 0;
    }

	//// Call the solver method depending on the type_of solver
	if (type_of_solver == Jacobi)
	{
		this->Jacobi_Solver(vect_b, vect_output);
	}
	else if (type_of_solver == Gauss_Siedel)
	{
		this->Gauss_Siedel_Solver(vect_b, vect_output);
	}
	else if (type_of_solver == Gaussian)
	{
		this->Gaussian_Solver(vect_b, vect_output);
	}
	else if (type_of_solver == LU)
	{
		this->LU_Solver(vect_b, vect_output);
	}
	else if (type_of_solver == Inverse)
	{
		this->Inverse_Solver(vect_b, vect_output);
	}
	else if (type_of_solver == Cholesky)
	{
		this->Cholesky_Solver(vect_b, vect_output);
	}
	else if (type_of_solver == Conjugate_Gradient)
	{
		this->Conjugate_Gradient_Solver(vect_b, vect_output);
	}
	else if (type_of_solver == Gauss_Jordan)
	{
		this->Gauss_Jordan_Solver(vect_b, vect_output);
	}
	else if (type_of_solver == Cramers)
	{
		this->Cramers_Solver(vect_b, vect_output);
	}
	else
	{
		std::cerr << "Please enter a valid type of solver!!!!!" << std::endl;
		return;
	}
}

// Solver method 1 
template <class T>
void Matrix<T>::Jacobi_Solver(const Matrix<T>& vect_b,Matrix<T>& vect_output)
{
	//value which will control if the loop should continue
	//if true- continue, else break
	bool big_error = true;

	//vector to store updated output values
	auto* vector_new = new T[vect_output.size_of_values];
	//temporary value needed for our looping
	double temp = 0;
	//counter of loops
	int lp = 0;

	// continuue loop as long as error is bigger than we want
	while (big_error)
	{
		lp++;
		for (int i = 0; i < this->rows; i++)
		{
			temp = 0;
			for (int j = 0; j < this->cols; j++)
			{
				if (i != j)
				{
					//assuming row-order here
					temp += this->values[j + i * this->cols] * vect_output.values[j];
				}
			}
			vector_new[i] = (vect_b.values[i] - temp) / (this->values[i + i * this->cols]);
		}
		//once all elements of vector_new have been updated, set them as new vector_output
		for (int i = 0; i < this->rows; i++)
		{
			vect_output.values[i] = vector_new[i];
		}
		//check how big error is
		big_error = check_error(*this, vect_b, vect_output);
		//if there were 10000 loops already, break no matter how big the error is 
		if (lp > 1000)
		{
			break;
		}
	}
	delete[] vector_new;
}

// Solver method 2
template <class T>
void Matrix<T>::Gauss_Siedel_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
	//value which will control if the loop should continue
	// if true- continue solver, else -break
	bool big_error = true;

	//temporary value needed for our looping
	double temp = 0;
	//counter of loops
	int lp = 0;

	//continuue loop as long as error is bigger than we want
	while (big_error)
	{
		lp++;
		for (int i = 0; i < this->rows; i++)
		{
			temp = 0;
			for (int j = 0; j < this->cols; j++)
			{
				if (i != j)
				{
					//assuming row-order
					temp += this->values[j + i * this->cols] * vect_output.values[j];
				}
			}
			//update value in the vector_output as soon as it is calculated
			//do not wait untill the entire vector is changed - this differentiaties Gauss-Seidel from Jacobian method
			vect_output.values[i] = (vect_b.values[i] - temp) / (this->values[i + i * this->cols]);
		}
		//check how big the error is
		big_error = check_error(*this, vect_b, vect_output);

		//if there were 10000 loops already, break no matter how big the error is 
		if (lp > 10000)
		{
			break;
		}
	}
}

// Solver method 3
template <class T>
void Matrix<T>::Gaussian_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
	//// This method impelment Gaussian with partial pivoting

	// we dont want to change the matrix we want to solve
	// thus we will create copies of matrix A and vector b
	auto* upper = new Matrix<T>(this->rows, this->cols, true);
	auto* b = new Matrix<T>(vect_b.rows, vect_b.cols, true);

	// copy the values
	for (int i = 0; i < this->size_of_values; i++)
		upper->values[i] = this->values[i];
	for (int i = 0; i < vect_b.size_of_values; i++)
		b->values[i] = vect_b.values[i];

	for (int k = 0; k < this->rows - 1; k++) //k loop over pivot except the final row
	{
		//// Implement partial pivoting
		int max_row = k;
		T max_value = upper->values[k * this->cols + k];
		for (int i = k + 1; i < this->rows; i++) //i loop over each row below pivot
		{
			// find row below the pivot with maximum absolute value
			if (fabs(upper->values[i * this->cols + k]) > max_value)
			{
				max_value = upper->values[i * this->cols + k];
				max_row = i;
			}
		}
		// swap the current pivot row, k with row with the maximum leading value 
		// to avoid division by zero/very small number
		if (max_row != k)
		{
			swap_rows(*upper, k, max_row);
			swap_rows(*b, k, max_row);
		}

		for (int i = k + 1; i < this->rows; i++) //i loop over each row below pivot
		{
			T s = upper->values[i * this->cols + k] / upper->values[k * this->cols + k];
			for (int j = k; j < this->rows; j++)  //j loop over elements within each row
				upper->values[i * this->cols + j] -= s * upper->values[k * this->cols + j];

			// update rhs matrix, b as well
			b->values[i] -= s * b->values[k];
		}
	}

	// do backward substitution to obtain the solution, vect_output
	back_substitution(upper, b->values, vect_output.values);

	delete upper;
	delete b;
}

// Solver method 4
template <class T>
void Matrix<T>::LU_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
	// Create and allocate memory for the upper - U, lower - L, and permut - P matrices
	auto* upper = new Matrix<T>(this->rows, this->cols, true);
	auto* lower = new Matrix<T>(this->rows, this->cols, true);
	auto* permut = new Matrix<T>(this->rows, this->cols, true);

	// Perform LU decomposition with partial pivoting
	this->decompose_LU(upper, lower, permut);

	// Create and allocate memory to store the output arrays of type T
	// explicitly initialising the values to be zero (important!!)
	auto* d = new T[vect_output.size_of_values]();
	auto* c = new T[vect_output.size_of_values]();

	// Simplified theory; Ax = b <=> LUx = Pb <=> LUx = d <=> Lc = d ; where Ux = c

	// swap rows in vect using permut (Pb = d); where b = vect
	permut->matVecMult(vect_b.values, d);

	// Perform the forward substitution to resolve c from Lc = d
	forward_substitution(lower, d, c);

	// Perform the back substitution to resolve x from Ux = c; where x = vect_output
	back_substitution(upper, c, vect_output.values);

	// deallocate all the memory pointers to avoid memory leak
	delete upper; delete lower; delete permut;
	delete[] c; delete[] d;
}

// Solver method 5
template <class T>
void Matrix<T>::Inverse_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
	// allocate memory to store the inverse
	auto* inverse_mat = new Matrix<T>(rows, cols, true);

	//calculate the inverse for this matrix (using gauss-jordan method)
	this->inverse(*inverse_mat);

	// obtain the solution Ax = b <=> x = A^{-1}b
	inverse_mat->matVecMult(vect_b.values, vect_output.values);

	//deallocate the memory
	delete inverse_mat;
}

// Solver method 6
template <class T>
void Matrix<T>::Cholesky_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
	//declare an array where values of the lower triangular matrix will be stored
	auto* lower = new T[this->size_of_values];
	//and initialize it to zeros

	for (int k = 0; k < this->size_of_values; k++)
	{
		lower[k] = 0;
	}
	//lets decompose our matrix(this) into lower triangular matrix
	for (int r = 0; r < this->rows; r++)
	{
		for (int c = 0; c < r + 1; c++)
		{
			if (c == r)
			{//finding the values for elements on diagonal
				double sum = 0;
				for (int j = 0; j < c; j++)
				{
					sum += lower[j + c * this->cols] * lower[j + c * this->cols];
				}
				lower[c + c * this->cols] = sqrt(this->values[c + c * this->cols] - sum);
			}//now using the values from diagonals we can find other values
			else
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
				{
					sum += (lower[j + r * this->cols] * lower[j + c * this->cols]);
				}
				lower[c + r * this->cols] = (this->values[c + r * this->cols] - sum) / lower[c + c * this->cols];
			}
		}
	}
	
	//M*x=b
	//L*L^T*x=b
	//lets assume y=L^T*x
	//so L*y=b
	//lets perform the forward substitution to get y
	auto* y = new T[vect_output.size_of_values];
	T s;
	y[0] = vect_b.values[0];
	for (int r = 0; r < this->rows; r++)
	{
		s = 0;
		for (int c = 0; c < r; c++)
		{
			s = s + lower[c + r * this->cols] * y[c];
		}
		y[r] = (vect_b.values[r] - s) / lower[r + r * this->cols];
	}

	// Perform the back substitution to get x from y=L^T*x
	for (int r = this->rows - 1; r >= 0; r--) //i loop from the bottom to the top row
	{
		T sum = 0;

		for (int c = r + 1; c < this->rows; c++)
			// sum the mat-vec product for all the columns on the right of pivot
			sum += lower[c * this->cols + r] * vect_output.values[c];

		vect_output.values[r] = (y[r] - sum) / lower[(r)*this->cols + (r)];
	}

	delete[] lower;
	delete[] y;

}

// Solver method 7
template <class T>
void Matrix<T>::Conjugate_Gradient_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
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
		

		double alpha = this->innerProduct(*r, *r) / (this->innerProduct(*p, *Ap) > near_zero ? this->innerProduct(*p, *Ap) : near_zero);      


		vect_output.CG_CombineVector(1, alpha, *p);// Set gradient to 1 
		r->CG_CombineVector(1, -alpha, *Ap);

		if (sqrt(this->innerProduct(*r, *r)) < tol) // check if residual reach tolerance
			break;

		double beta = this->innerProduct(*r, *r) / (this->innerProduct(*p, *Ap) > near_zero ? this->innerProduct(*p, *Ap) : near_zero);

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

// Solver method 8
template <class T>
void Matrix<T>::Gauss_Jordan_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
	// we dont want to change the original matrix and vector
	// so create a temporary storage here
	auto* temp = new Matrix<T>(this->cols, this->rows, true);
	auto* b = new Matrix<T>(vect_b.rows, vect_b.cols, true);

	// copy the values
	for (int i = 0; i < this->size_of_values; i++)
		temp->values[i] = this->values[i];
	for (int i = 0; i < vect_b.size_of_values; i++)
		b->values[i] = vect_b.values[i];

	// form upper triangular matrix from temp
	for (int k = 0; k < this->rows - 1; k++) //k loop over pivot except the final row
	{
		for (int i = k + 1; i < this->rows; i++) //i loop over each row below pivot
		{
			T s = temp->values[i * this->cols + k] / temp->values[k * this->cols + k];
			for (int j = k; j < this->rows; j++)  //j loop over elements within each row
			{
				temp->values[i * this->cols + j] -= s * temp->values[k * this->cols + j];
			}
			// update rhs matrix, b as well
			b->values[i] -= s * b->values[k];
		}
	}

	// form lower triangular matrix from temp
	// if an upper triangular matirx is passed the result will be a diagonal matrix
	for (int k = this->rows - 1; k >= 0; k--) //k loop over pivot elements from bottom to top
	{
		for (int i = k - 1; i >= 0; i--) //i loop over each row above pivot
		{
			T s = temp->values[i * this->cols + k] / temp->values[k * this->cols + k];
			for (int j = this->rows - 1; j >= k ; j--)  //j loop over elements within each row
			{
				temp->values[i * this->cols + j] -= s * temp->values[k * this->cols + j];
			}
			// update rhs matrix, b as well
			b->values[i] -= s * b->values[k];
		}
	}

	// divide vector b with the diagonal values of temp
	// doing this to temp will transform it into an identity matrix (but we dont need to do that here)
	// we just want the RHS vector which is now the solution, vect_out (x)
	for (int i = 0; i < vect_output.size_of_values; i++) // loop over all rows
		vect_output.values[i] = b->values[i] / temp->values[i * this->cols + i];

	delete temp;
	delete b;
}

// Solver method 9
template <class T>
void Matrix<T>::Cramers_Solver(const Matrix<T>& vect_b, Matrix<T>& vect_output)
{
	// calculate the determinant of the matrix we want to solve
	double main_det = this->determinant();

	// allocate memory to store a matrix where the columns will be swapped
	auto* temp = new Matrix<T>(this->rows, this->cols, true);

	// copy the original matrix to temp
	for (int n = 0; n < this->size_of_values; n++)
		temp->values[n] = this->values[n];

	for (int j = 0; j < this->cols; j++)
	{
		for (int i = 0; i < this->rows; i++)
		{
			// set the values in previous column back to be 
			// the same as the values in the original matrix
			if (j > 0) temp->values[i * this->cols + j - 1] = this->values[i * this->cols + j - 1];
			// add vector_b to the column which corresponds to the unknown we are solving
			temp->values[i * this->cols + j] = vect_b.values[i];
		}
		vect_output.values[j] = temp->determinant() / main_det;
	}

	delete temp;
}


// Decompose this matrix into lower, L and upper, U triangular form with partial pivoting
// stand on its own because can be called once to solve for multiple RHS vectors
// Ax = b  <=> LUx = Pb;  where P; permutation matrix (store information on row-swapping)
template <class T>
void Matrix<T>::decompose_LU(Matrix<T>* upper, Matrix<T>* lower, Matrix<T>* permut)
{
	// Initialise the values in the matrices
	// we dont want to change the matrix we want to solve, thus we create a copy for upper, U
	for (int i = 0; i < this->rows; i++)
	{
		for (int j = 0; j < this->cols; j++)
		{
			// initialise U to be the same as the matrix we want to solve
			upper->values[i * this->cols + j] = this->values[i * this->cols + j];
			// initialise L to be zero (should be an identity matrix)
			// but will add 1 to diagonal later after partial pivotting
			lower->values[i * this->cols + j] = 0;
			// initialise P to be an identiy matrix
			if (i == j) permut->values[i * this->cols + j] = 1;
			else permut->values[i * this->cols + j] = 0;
		}
	}

	// Calculate the upper and lower matrices using Doolittle algorithm
	for (int k = 0; k < this->rows - 1; k++) //k loop over pivot except the final row
	{
		//// Implement partial pivoting
		int max_row = k;
		T max_value = upper->values[k * this->cols + k];
		for (int i = k + 1; i < this->rows; i++) //i loop over each row below pivot
		{
			// find row below the pivot with maximum absolute value
			if (fabs(upper->values[i * this->cols + k]) > max_value)
			{
				max_value = upper->values[i * this->cols + k];
				max_row = i;
			}
		}
		// swap the rows if current row does not has the maximum leading value
		if (max_row != k)
		{
			// swap the current pivot row, k with row with the maximum leading value 
			// to avoid division by zero/very small number
			swap_rows(*upper, k, max_row);
			swap_rows(*lower, k, max_row);
			swap_rows(*permut, k, max_row);

		} lower->values[k * this->cols + k] = 1; //set the diagonal of the lower matrix equals to 1

		for (int i = k + 1; i < this->rows; i++) //i loop over each row below pivot
		{
			// store the factors outside the j-loop to avoid overwriting the values
			T s = upper->values[i * this->cols + k] / upper->values[k * this->cols + k];
			for (int j = k; j < this->rows; j++)  //j loop over elements for each row
			{
				upper->values[i * this->cols + j] -= s * upper->values[k * this->cols + j];
				lower->values[i * this->cols + k] = s;
			}
		}
	} lower->values[this->size_of_values - 1] = 1; // value of the final element in the lower matrix always equal to 1
}




// Perform forward substitution on a lower triangular matrix, L
// lower * vect_out = vect_in 
template <class T>
void Matrix<T>::forward_substitution(Matrix<T>* lower, T* vect_in, T* vect_out)
{
	for (int i = 0; i < lower->rows; i++) //i loop from the top to the bottom row
	{
		T sum = 0;
		for (int j = i - 1; j >= 0; j--)
			// sum the mat-vec product for all the columns on the left of pivot
			sum += lower->values[i * lower->cols + j] * vect_out[j];
		vect_out[i] = (vect_in[i] - sum) / lower->values[i * lower->cols + i];
	}
}

// Perform back substitution on an upper triangular matrix, U
// upper * vect_out = vect_in 
template <class T>
void Matrix<T>::back_substitution(Matrix<T>* upper, T* vect_in, T* vect_out)
{
	for (int i = upper->rows - 1; i >= 0; i--) //i loop from the bottom to the top row
	{
		T sum = 0;
		for (int j = i + 1; j < upper->rows; j++)
			// sum the mat-vec product for all the columns on the right of pivot
			sum += upper->values[i * upper->cols + j] * vect_out[j];
		vect_out[i] = (vect_in[i] - sum) / upper->values[i * upper->cols + i];
	}
}

// Swap the rows within objct of type matrix 
// mainly used for partial pivoting
template <class T>
void Matrix<T>::swap_rows(Matrix<T>& matrix, int current_row, int max_row)
{
	// row[current_row]<-->row[max_row]
	// also works for a vector of type Matrix<T>
	T temp;
	for (int j = 0; j < matrix.cols; j++)
	{
		// temporary store the value to avoid overwriting it
		temp = matrix.values[current_row * matrix.cols + j];
		matrix.values[current_row * matrix.cols + j] = matrix.values[max_row * matrix.cols + j];
		matrix.values[max_row * matrix.cols + j] = temp;
	}
}

//vector operation for conjugate gradient
template<class T>
void Matrix<T>::CG_CombineVector(T gradient, double alpha, const Matrix<T>& vect_b)
{
	for (int j = 0; j < this->rows; j++)
		this->values[j] = gradient * this->values[j] + alpha * vect_b.values[j];
}

//function which calculates the error for Gauss-Seidel and Jacobian methods
template <class T>
bool check_error(Matrix<T>& mat,const Matrix<T>& vect_b,Matrix<T>& vect_output)
{
	double value = 0;
	for (int i = 0; i < mat.rows; i++)
	{
		value = 0;
		for (int j = 0; j < mat.cols; j++)
		{
			//calculate values of b vector based on current solution
			value += mat.values[j + i * mat.cols] * vect_output.values[j];
		
		}
		if (abs(value - vect_b.values[i]) > mat.tol)
		{
			return true;
		}
	}
	return false;
}
