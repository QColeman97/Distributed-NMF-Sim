package main

import (
	"fmt"
	"math/rand"
	"sync"

	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// MPI-FAUN notation:
// X = matrix X, x = vector x
// Xi = ith row block of X, X^i = ith column block of X
// xi = ith row of X, x^i = ith column of X

func parallelNMF(node *Node, maxIter int) (mat.Dense, mat.Dense) {
	defer wg.Done()
	// TODO
	// Synchronize nodes, so no node goes to next collective
	// until all nodes are done with that collective
	// Loop over each node & check bool checkpt vars

	// Local matrices
	var Wij, Hji mat.Dense

	wRowsPerNode := m / numNodes // m/p
	hColsPerNode := n / numNodes // n/p

	// MPI-FAUN steps in comments
	// 1) Initialize Hji - dims = k x (n/p)
	h := make([]float64, k*hColsPerNode)
	for i := range h {
		h[i] = rand.NormFloat64()
	}
	Hji = *mat.NewDense(k, hColsPerNode, h)
	// Not in paper, but maybe initialize Wij too - dims = (m/p) x k
	w := make([]float64, wRowsPerNode*k)
	for i := range w {
		w[i] = rand.NormFloat64()
	}
	Wij = *mat.NewDense(wRowsPerNode, k, w)

	for iter := 0; iter < maxIter; iter++ {
		// fmt.Println(node.nodeID, "ITER #", iter+1)
		// Update W below
		// 3)
		Uij := &mat.Dense{}
		Uij.Mul(&Hji, Hji.T()) // k x k
		// 4)
		HGramMat := node.newAllReduce(Uij)
		// fmt.Println(node.nodeID, "did allReduce")
		// 5)
		Hj := node.newAllGatherAcrossNodeColumns(&Hji, hColsPerNode) // k x (n/p_c)
		// fmt.Println(node.nodeID, "did allGatherCols")
		// 6)
		Vij := &mat.Dense{}
		Vij.Mul(node.aPiece, Hj.T()) // (m/pr) x k
		// 7)
		HProductMatij := node.newReduceScatterAcrossNodeRows(Vij)
		// fmt.Println(node.nodeID, "did reduceScatterRow")
		// 8)
		updateW(&Wij, HGramMat, HProductMatij)
		// fmt.Println(node.nodeID, "updated W")
		// Update H below
		// 9)
		Xij := &mat.Dense{}
		Xij.Mul(Wij.T(), &Wij) // k x k
		// 10)
		WGramMat := node.newAllReduce(Xij)
		// fmt.Println(node.nodeID, "did allReduce")
		// 11)
		Wi := node.newAllGatherAcrossNodeRows(&Wij, wRowsPerNode) // (m/p_r) x k
		// fmt.Println(node.nodeID, "did allGatherRows")
		// 12)
		Yij := &mat.Dense{}
		Yij.Mul(Wi.T(), node.aPiece) // k x (n/p_c)
		// 13)
		WProductMatij := node.newReduceScatterAcrossNodeColumns(Yij)
		// fmt.Println(node.nodeID, "did reduceScatterCols")
		// 14)
		updateH(&Hji, WGramMat, WProductMatij)
		// fmt.Println(node.nodeID, "updated H")
	}

	// TODO Send Wij & Hji to client
	return Wij, Hji
}

// Line 8 of MPI-FAUN
// Multiplicative Update: W = W * ((A @ Ht) / (W @ (H @ Ht)))
// Formula uses: Gram matrix, matrix product w/ A, and W
func updateW(W *mat.Dense, HGramMat *mat.Dense, HProductMatij mat.Matrix) {
	// W dims = (m/p) x k
	// HGramMat dims = k x k
	// HProductMatij dims = (m/p) x k

	update := &mat.Dense{}
	update.Mul(W, HGramMat) // (m/p) x k

	update.DivElem(HProductMatij, update)
	W.MulElem(W, update)
}

// Line 14 of MPI-FAUN
// Multiplicative Update: H = H * ((Wt @ A) / ((Wt @ W) @ H))
// Formula uses: Gram matrix, matrix product w/ A, and H
func updateH(H *mat.Dense, WGramMat *mat.Dense, WProductMatij mat.Matrix) {
	// H dims = k x (n/p)
	// WGramMat dims = k x k
	// WProductMatij dims = k x (n/p)

	update := &mat.Dense{}
	update.Mul(WGramMat, H) // k x (n/p)

	update.DivElem(WProductMatij, update)
	H.MulElem(H, update)
}

func partitionAMatrix(A *mat.Dense) []mat.Matrix {
	var piecesOfA []mat.Matrix
	aRowsPerNode, aColsPerNode := m/nodeRows, n/nodeCols // (m/p_r), (n/p_c)
	// fmt.Println("A rows per node:", aRowsPerNode, "A columns per node:", aColsPerNode)

	for i := 0; i < nodeRows; i++ {
		for j := 0; j < nodeCols; j++ {
			// fmt.Println("A", rowsPerNode*i, rowsPerNode*(i+1), colsPerNode*j, colsPerNode*(j+1))
			aPiece := A.Slice(aRowsPerNode*i, aRowsPerNode*(i+1), aColsPerNode*j, aColsPerNode*(j+1))
			// Make pieces each their own copies of the data
			piecesOfA = append(piecesOfA, mat.DenseCopyOf(aPiece))
		}
	}

	return piecesOfA
}

func makeNode(chans [numNodes]chan MatMessage, akChans [numNodes]chan bool, id int, aPiece mat.Matrix) *Node {
	return &Node{
		nodeID:    id,
		nodeChans: chans,
		nodeAks:   akChans,
		inChan:    chans[id],
		aPiece:    aPiece,
		aks:       akChans[id],
	}
}

func makeMatrixChans() [numNodes]chan MatMessage {
	var chans [numNodes]chan MatMessage
	for ch := range chans {
		chans[ch] = make(chan MatMessage, numNodes*3) // numNodes-1)
	}
	return chans
}

func makeAkChans() [numNodes]chan bool {
	var chans [numNodes]chan bool
	for ch := range chans {
		chans[ch] = make(chan bool, numNodes*3) // numNodes-1)
	}
	return chans
}

var wg sync.WaitGroup

const m, n, k = 18, 12, 5
const numNodes, nodeRows, nodeCols = 6, 3, 2

const largeBlockSizeW = m / nodeRows
const largeBlockSizeH = n / nodeCols
const smallBlockSizeW = m / numNodes
const smallBlockSizeH = n / numNodes

func main() {
	maxIter := 100

	// Initialize input matrix A
	a := make([]float64, m*n)
	for i := 0; i < m*n; i++ {
		a[i] = float64(i)
	}
	A := mat.NewDense(m, n, a)
	aRows, aCols := A.Dims()
	fmt.Println("A dims:", aRows, aCols)
	fmt.Println("W dims:", m, k)
	fmt.Println("H dims:", k, n)
	fmt.Println("\nA:")
	matPrint(A)

	// Partition A into pieces for nodes
	piecesOfA := partitionAMatrix(A)
	// Init nodes
	chans := makeMatrixChans()
	akChans := makeAkChans()
	var nodes [numNodes]*Node
	for i := 0; i < numNodes; i++ {
		id := i
		nodes[i] = makeNode(chans, akChans, id, piecesOfA[i])
	}

	// Launch nodes with their A pieces
	for _, node := range nodes {
		wg.Add(1)
		go parallelNMF(node, maxIter)
	}

	// TODO wait for W & H blocks from nodes

	// TODO construct W & H full matrices

	wg.Wait()

	// approxA := &mat.Dense{}
	// approxA.Mul(W, H)
	// fmt.Println("\nA Approximation:")
	// matPrint(approxA)
	fmt.Println("Done")
}
