package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Gonum References:
// https://medium.com/wireless-registry-engineering/gonum-tutorial-linear-algebra-in-go-21ef136fc2d7
// https://talks.godoc.org/github.com/gonum/talks/2017/gonumtour.slide#14

// Gonum notes:
// type mat.Matrix is an interface, mat.Dense is a type that conforms to mat.Matrix
// a mat.Matrix can be cast to only a POINTER of a type mat.Dense, I guess
//		b/c mat.Dense methods from mat.Matrix have pointer receivers (e.g. func (m *mat.Dense) Dims)
// https://stackoverflow.com/questions/40823315/x-does-not-implement-y-method-has-a-pointer-receiver
// a matrix receiver must have same dimensions as result of its method (eg. numerRHS & Mul)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// Multiplicative Update: H = H * ((Wt @ V) / (Wt @ 1 W @ H))
func updateH(H *mat.Dense, W *mat.Dense, A *mat.Dense) {
	update := &mat.Dense{}
	update.Mul(W.T(), A) // k x n

	denom1 := &mat.Dense{}
	denom1.Mul(W.T(), W) // k x k
	denom := &mat.Dense{}
	denom.Mul(denom1, H) // k x n

	update.DivElem(update, denom)
	H.MulElem(H, update)
}

// Multiplicative Update: W = W * ((V @ Ht) / (W @ H @ 1 @ Ht))
func updateW(W *mat.Dense, H *mat.Dense, A *mat.Dense) {
	update := &mat.Dense{}
	update.Mul(A, H.T()) // m x k

	denom1 := &mat.Dense{}
	denom1.Mul(H, H.T()) // k x k
	denom := &mat.Dense{}
	denom.Mul(W, denom1) // m x k

	update.DivElem(update, denom)
	W.MulElem(W, update)
}

func nmf(W *mat.Dense, H *mat.Dense, A *mat.Dense, maxIter int) {
	fmt.Println("Doing NMF")
	for iter := 0; iter < maxIter; iter++ {
		updateW(W, H, A)
		updateH(H, W, A)
	}
}

const m, n, k = 18, 12, 4

func main() {
	// Initialize input matrix A
	a := make([]float64, m*n)
	for i := 0; i < m*n; i++ {
		a[i] = float64(i)
	}
	A := mat.NewDense(m, n, a)
	println("A:")
	matPrint(A)

	// Initialize factors W & H
	w := make([]float64, m*k)
	for i := range w {
		w[i] = rand.NormFloat64()
	}
	W := mat.NewDense(m, k, w)

	h := make([]float64, k*n)
	for i := range h {
		h[i] = rand.NormFloat64()
	}
	H := mat.NewDense(k, n, h)

	startTime := time.Now()

	nmf(W, H, A, 100)

	// fmt.Println("W:")
	// matPrint(W)
	// fmt.Println("H:")
	// matPrint(H)

	// approx := W.At(0, 0)*H.At(0, 2) + W.At(0, 1)*H.At(1, 2)
	// fmt.Println("Approximate A[0][2] using W & H:", approx)

	// approx = W.At(1, 0)*H.At(0, 1) + W.At(1, 1)*H.At(1, 1)
	// fmt.Println("Approximate A[1][1] using W & H:", approx)

	approxA := &mat.Dense{}
	approxA.Mul(W, H)
	// Truncate values to no decimal
	aA := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			aA[(i*n)+j] = math.Round(approxA.At(i, j))
		}
	}
	approxA = mat.NewDense(m, n, aA)
	duration := time.Now().Sub(startTime)
	fmt.Println("\nApproximation of A:")
	matPrint(approxA)
	fmt.Println("Took", duration)
}
