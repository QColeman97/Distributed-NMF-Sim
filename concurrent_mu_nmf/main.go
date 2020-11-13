package main

import (
	"fmt"
	"math/rand"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// MPI-FAUN notation:
// A = matrix A, a = vector a
// Ai = ith row block of A, A^i = ith column block of A
// ai = ith row of A, a^i = ith column of A

// implement MPI collectives
//	- reduce-scatter 	- used across all proc rows
//		every node retrieve V/Y from every node
//		every node performs reduction
//		scatter data equally to other nodes
//  - all-gather 		- used across all proc columns
//		every node send their Hj/Wi to every node
//	- all-reduce 		- used across all proc
//		every node retrieve U/X from every node
//		every node performs reduction
type Node struct {
	nodeID        int
	nodeChans     [numNodes]chan mat.Matrix
	inChan        chan mat.Matrix
	aPiece        mat.Matrix
	wPiece        mat.Matrix
	hPiece        mat.Matrix
	allGatherDone bool
	allReduceDone bool
	scatterDone   bool
}

func (node *Node) parallelNMF(maxIter int) {
	fmt.Println("Running node", node.nodeID)

	// Initialize Hij

	for iter := 0; iter < maxIter; iter++ {
		// Update W

		// Update H

	}
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// Line 14 of MPI-FAUN
// Multiplicative Update: H = H * ((Wt @ A) / (Wt @ W @ H))
func updateH(H *mat.Dense, W *mat.Dense, A *mat.Dense) {
	// Compute WtW

	// Compute (WtA)^j

	// Update ((H^j)^i)t

	update := &mat.Dense{}
	update.Mul(W.T(), A) // k x n

	denom1 := &mat.Dense{}
	denom1.Mul(W.T(), W) // k x k
	denom := &mat.Dense{}
	denom.Mul(denom1, H) // k x n

	update.DivElem(update, denom)
	H.MulElem(H, update)
}

// Line 8 of MPI-FAUN
// Multiplicative Update: W = W * ((A @ Ht) / (W @ H @ Ht))
func updateW(W *mat.Dense, H *mat.Dense, A *mat.Dense) {
	// Compute HHt

	// Compute (AHt)i

	update := &mat.Dense{}
	update.Mul(A, H.T()) // m x k

	denom1 := &mat.Dense{}
	denom1.Mul(H, H.T()) // k x k
	denom := &mat.Dense{}
	denom.Mul(W, denom1) // m x k

	update.DivElem(update, denom)
	W.MulElem(W, update)
}

func partitionMatrices(A *mat.Dense, W *mat.Dense, H *mat.Dense) ([]mat.Matrix, []mat.Matrix, []mat.Matrix) {
	var piecesOfA, piecesOfW, piecesOfH []mat.Matrix
	rowsPerNode, colsPerNode := m/nodesR, n/nodesC
	// fmt.Println("rows per node:", rowsPerNode, "columns per node:", colsPerNode)

	for i := 0; i < nodesR; i++ {
		for j := 0; j < nodesC; j++ {
			// fmt.Println("A", rowsPerNode*i, rowsPerNode*(i+1), colsPerNode*j, colsPerNode*(j+1))
			aPiece := A.Slice(rowsPerNode*i, rowsPerNode*(i+1), colsPerNode*j, colsPerNode*(j+1))
			// Make pieces each their own copies of the data
			piecesOfA = append(piecesOfA, mat.DenseCopyOf(aPiece))
		}
	}

	rowsPerNodeW := m / numNodes
	for i := 0; i < rowsPerNode; i++ {
		wPiece := W.Slice(rowsPerNodeW*i, rowsPerNodeW*(i+1), 0, k)
		// Make pieces each their own copies of the data
		piecesOfW = append(piecesOfW, mat.DenseCopyOf(wPiece))
	}
	colsPerNodeH := n / numNodes
	for j := 0; j < colsPerNode; j++ {
		hPiece := H.Slice(0, k, colsPerNodeH*j, colsPerNodeH*(j+1))
		// Make pieces each their own copies of the data
		piecesOfH = append(piecesOfH, mat.DenseCopyOf(hPiece))
	}

	return piecesOfA, piecesOfW, piecesOfH
}

func makeNode(chans [numNodes]chan mat.Matrix, id int, aPiece mat.Matrix, wPiece mat.Matrix, hPiece mat.Matrix) Node {
	return Node{
		nodeID:    id,
		nodeChans: chans,
		inChan:    chans[id],
		aPiece:    aPiece,
		wPiece:    wPiece,
		hPiece:    hPiece,
	}
}

func makeMatrixChans() [numNodes]chan mat.Matrix {
	var chans [numNodes]chan mat.Matrix
	for ch := range chans {
		chans[ch] = make(chan mat.Matrix, 10)
	}
	return chans
}

const m, n, k = 18, 12, 5
const numNodes, nodesR, nodesC = 6, 3, 2

func main() {
	var wg sync.WaitGroup

	maxIter := 100

	// Initialize input matrix A
	a := make([]float64, m*n)
	for i := 0; i < m*n; i++ {
		a[i] = float64(i)
	}
	A := mat.NewDense(m, n, a)
	aRows, aCols := A.Dims()
	fmt.Println("A:", aRows, aCols)

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
	wRows, wCols := W.Dims()
	fmt.Println("W:", wRows, wCols)
	hRows, hCols := H.Dims()
	fmt.Println("H:", hRows, hCols)

	// Partition A,W,H into pieces for nodes
	piecesOfA, piecesOfW, piecesOfH := partitionMatrices(A, W, H)
	// Init nodes
	chans := makeMatrixChans()
	var nodes [numNodes]Node
	for i := 0; i < numNodes; i++ {
		nodes[i] = makeNode(chans, i, piecesOfA[i], piecesOfW[i], piecesOfH[i])
	}

	// Launch nodes with their A,W,H pieces
	for _, node := range nodes {
		wg.Add(1)
		go func() {
			node.parallelNMF(maxIter)
			wg.Done()
		}()
	}
	wg.Wait()
}
