package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func main() {
	// u := mat.NewVecDense(3, []float64{1, 2, 3})
	// println("u: ")
	// matPrint(u)
	// v := mat.NewVecDense(3, []float64{4, 5, 6})
	// println("v: ")
	// matPrint(v)
	// w := mat.NewVecDense(3, nil)
	// w.AddVec(u, v)
	// println("u + v: ")
	// matPrint(w)
	// // Or, you can overwrite u with the result of your operation to
	// //save space.
	// u.AddVec(u, v)
	// println("u (overwritten):")
	// matPrint(u)
	// // Add u + alpha * v for some scalar alpha
	// w.AddScaledVec(u, 2, v)
	// println("u + 2 * v: ")
	// matPrint(w)
	// // Subtract v from u
	// w.SubVec(u, v)
	// println("v - u: ")
	// matPrint(w)
	// // Scale u by alpha
	// w.ScaleVec(23, u)
	// println("u * 23: ")
	// matPrint(w)
	// // Compute the dot product of u and v
	// // Since float64’s don’t have a dot method, this is not done
	// //inplace
	// d := mat.Dot(u, v)
	// println("u dot v: ", d)
	// // Find length of v
	// l := v.Len()
	// println("Length of v: ", l)
	// // We can also find the length of v in Euclidean space
	// // The 2 parameter specifices that this is the Frobenius norm
	// // Rather than the maximum absolute column sum
	// // This distinction is more important when Norm is applied to
	// // Matrices since vectors only have one column and the the
	// // maximum absolute column sum is the Frobenius norm squared.
	// println(mat.Norm(v, 2))

	// fmt.Println("\n\n\n\n")

	v := make([]float64, 12)
	for i := 0; i < 12; i++ {
		v[i] = float64(i)
	}
	// Create a new matrix
	A := mat.NewDense(3, 4, v)
	println("A:")
	matPrint(A)
	// Setting and getting are pretty simple
	a := A.At(0, 2)
	println("A[0, 2]: ", a)
	A.Set(0, 2, -1.5)
	matPrint(A)
	// However, we can also set and get rows and columns
	// Rows and columns are returned as vectors, which can be used for
	// other computations
	println("Row 1 of A:")
	matPrint(A.RowView(1))
	println("Column 0 of A:")
	matPrint(A.ColView(0))
	// Rows and columns may be set,
	// But this is done from slices of floats
	row := []float64{10, 9, 8, 7}
	A.SetRow(0, row)
	matPrint(A)
	col := []float64{3, 2, 1}
	A.SetCol(0, col)
	matPrint(A)
	// Addition and subtraction are identical for Matrices and vectors
	// However, if the dimensions don't match,
	// the function will throw a panic.
	B := mat.NewDense(3, 4, nil)
	B.Add(A, A)
	println("B:")
	matPrint(B)
	C := mat.NewDense(3, 4, nil)
	C.Sub(A, B)
	println("A - B:")
	matPrint(C)
	// We can scale all elements of the matrix by a constant
	C.Scale(3.5, B)
	println("3.5 * B:")
	matPrint(C)
	// Transposing a matrix is a little funky.
	// A.T() is no longer *Dense, but is now a Matrix
	// We can still do most of the same operations with it,
	// but we cannot set any of its values.
	println("A'")
	matPrint(A.T())
	// Multiplication is pretty straightforward
	D := mat.NewDense(3, 3, nil)
	D.Product(A, B.T())
	println("A * B'")
	matPrint(D)
	// We can use Product to multiply as many matrices as we want,
	// provided the receiver has the appropriate dimensions
	// The order of operations is optimized to reduce operations.
	D.Product(D, A, B.T(), D)
	println("D * A * B' * D")
	matPrint(D)
	// We can also apply a function to elements of the matrix.
	// This function must take two integers and a float64,
	// representing the row and column indices and the value in the
	// input matrix.  It must return a float.  See sumOfIndices below.
	C.Apply(sumOfIndices, A)
	println("C:")
	matPrint(C)
	// Once again, we have some functions that return scalar values
	// For example, we can compute the determinant
	E := A.Slice(0, 3, 0, 3)
	d := mat.Det(E)
	println("det(E): ", d)
	// And the trace
	t := mat.Trace(E)
	println("tr(E)", t)
}

func sumOfIndices(i, j int, v float64) float64 {
	return float64(i + j)
}
