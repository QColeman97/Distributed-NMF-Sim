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

const debugNode = 5
const debugIter = 99

// Corresponding MPI-FAUN steps in comments
func parallelNMF(node *Node, maxIter int) {
	// Local matrices
	var Wij, Hji mat.Dense

	// 1) Initialize Hji - dims = k x (n/p)
	h := make([]float64, k*smallBlockSizeH)
	for i := range h {
		h[i] = rand.NormFloat64()
	}
	Hji = *mat.NewDense(k, smallBlockSizeH, h)
	// if node.nodeID == debugNode {
	// 	fmt.Println("Initial Hji:")
	// 	matPrint(&Hji)
	// }
	// Not in paper, but maybe initialize Wij too - dims = (m/p) x k
	w := make([]float64, smallBlockSizeW*k)
	for i := range w {
		w[i] = rand.NormFloat64()
	}
	Wij = *mat.NewDense(smallBlockSizeW, k, w)
	// if node.nodeID == debugNode {
	// 	fmt.Println("Initial Wij:")
	// 	matPrint(&Wij)
	// }

	for iter := 0; iter < maxIter; iter++ {
		// fmt.Println(node.nodeID, "ITER #", iter+1)
		// Update W Part
		// 3)
		Uij := &mat.Dense{}
		Uij.Mul(&Hji, Hji.T()) // k x k
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("Uij:")
		// 	matPrint(Uij)
		// }
		// 4)
		HGramMat := node.allReduce(Uij)
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("HGramMat:")
		// 	matPrint(HGramMat)
		// }
		// fmt.Println(node.nodeID, "did allReduce")
		// 5)
		Hj := node.allGatherAcrossNodeColumns(&Hji) // k x (n/p_c)
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("Hj:")
		// 	matPrint(Hj)
		// }
		// fmt.Println(node.nodeID, "did allGatherCols")
		// 6)
		Vij := &mat.Dense{}
		Vij.Mul(node.aPiece, Hj.T()) // (m/pr) x k
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("Vij:")
		// 	matPrint(Vij)
		// }
		// 7)
		HProductMatij := node.reduceScatterAcrossNodeRows(Vij) // (m/p) x k
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("HProdMatij:")
		// 	matPrint(HProductMatij)
		// }
		// fmt.Println(node.nodeID, "did reduceScatterRow")
		// 8)
		updateW(&Wij, HGramMat, HProductMatij)
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("UPDATED Wij:")
		// 	matPrint(&Wij)
		// }
		// fmt.Println(node.nodeID, "updated W")
		// Update H Part
		// 9)
		Xij := &mat.Dense{}
		Xij.Mul(Wij.T(), &Wij) // k x k
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("Xij:")
		// 	matPrint(Xij)
		// }
		// 10)
		WGramMat := node.allReduce(Xij)
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("WGramMat:")
		// 	matPrint(WGramMat)
		// }
		// fmt.Println(node.nodeID, "did allReduce")
		// 11)
		Wi := node.allGatherAcrossNodeRows(&Wij) // (m/p_r) x k
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("Wi:")
		// 	matPrint(Wi)
		// }
		// fmt.Println(node.nodeID, "did allGatherRows")
		// 12)
		Yij := &mat.Dense{}
		Yij.Mul(Wi.T(), node.aPiece) // k x (n/p_c)
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("Yij:")
		// 	matPrint(Yij)
		// }
		// 13)
		WProductMatji := node.reduceScatterAcrossNodeColumns(Yij) // k x (n/p)
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("WProdMatji:")
		// 	matPrint(WProductMatji)
		// }
		// fmt.Println(node.nodeID, "did reduceScatterCols")
		// 14)
		updateH(&Hji, WGramMat, WProductMatji)
		// if node.nodeID == debugNode && iter == debugIter {
		// 	fmt.Println("UPDATED Hji:")
		// 	matPrint(&Hji)
		// }
		// fmt.Println(node.nodeID, "updated H")
	}

	// Send Wij & Hji to client
	node.clientChan <- MatMessage{Wij, node.nodeID, true, false}
	node.clientChan <- MatMessage{Hji, node.nodeID, false, true}

	wg.Done()
}

// Line 8 of MPI-FAUN - Multiplicative Update: W = W * ((A @ Ht) / (W @ (H @ Ht)))
// Formula uses: Gram matrix, matrix product w/ A, and W
// 		W dims = (m/p) x k
// 		HGramMat dims = k x k
// 		HProductMatij dims = (m/p) x k
func updateW(W *mat.Dense, HGramMat *mat.Dense, HProductMatij mat.Matrix) {
	update := &mat.Dense{}
	update.Mul(W, HGramMat) // (m/p) x k

	update.DivElem(HProductMatij, update)
	W.MulElem(W, update)
}

// Line 14 of MPI-FAUN - Multiplicative Update: H = H * ((Wt @ A) / ((Wt @ W) @ H))
// Formula uses: Gram matrix, matrix product w/ A, and H
// 		H dims = k x (n/p)
// 		WGramMat dims = k x k
// 		WProductMatji dims = k x (n/p)
func updateH(H *mat.Dense, WGramMat *mat.Dense, WProductMatji mat.Matrix) {
	update := &mat.Dense{}
	update.Mul(WGramMat, H) // k x (n/p)

	update.DivElem(WProductMatji, update)
	H.MulElem(H, update)
}

func partitionAMatrix(A *mat.Dense) []mat.Matrix {
	var piecesOfA []mat.Matrix

	for i := 0; i < numNodeRows; i++ {
		for j := 0; j < numNodeCols; j++ {
			aPiece := A.Slice(largeBlockSizeW*i, largeBlockSizeW*(i+1), largeBlockSizeH*j, largeBlockSizeH*(j+1))
			// Make pieces each their own copies of the data
			piecesOfA = append(piecesOfA, mat.DenseCopyOf(aPiece))
		}
	}

	return piecesOfA
}

func makeNode(chans [numNodes]chan MatMessage, akChans [numNodes]chan bool, clientChan chan MatMessage, id int, aPiece mat.Matrix) *Node {
	return &Node{
		nodeID:     id,
		nodeChans:  chans,
		nodeAks:    akChans,
		inChan:     chans[id],
		aPiece:     aPiece,
		aks:        akChans[id],
		clientChan: clientChan,
	}
}

func makeMatrixChans() [numNodes]chan MatMessage {
	var chans [numNodes]chan MatMessage
	for ch := range chans {
		chans[ch] = make(chan MatMessage, numNodes*3)
	}
	return chans
}

func makeAkChans() [numNodes]chan bool {
	var chans [numNodes]chan bool
	for ch := range chans {
		chans[ch] = make(chan bool, numNodes*3)
	}
	return chans
}

var wg sync.WaitGroup

const m, n, k = 18, 12, 5
const numNodes, numNodeRows, numNodeCols = 6, 3, 2

const largeBlockSizeW = m / numNodeRows
const largeBlockSizeH = n / numNodeCols
const smallBlockSizeW = m / numNodes
const smallBlockSizeH = n / numNodes

func main() {
	maxIter := 100

	// Initialize input matrix A
	a := make([]float64, m*n)
	for i := 0; i < m*n; i++ {
		a[i] = float64(i) // / 10 // make smaller values, overflow error?
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
	clientChan := make(chan MatMessage, numNodes*3)
	var nodes [numNodes]*Node
	for i := 0; i < numNodes; i++ {
		id := i
		nodes[i] = makeNode(chans, akChans, clientChan, id, piecesOfA[i])
	}

	// Launch nodes with their A pieces
	for _, node := range nodes {
		wg.Add(1)
		go parallelNMF(node, maxIter)
	}

	// Wait for W & H blocks from nodes
	wPieces, hPieces := make([]mat.Dense, numNodes), make([]mat.Dense, numNodes)
	for w, h := 0, 0; w < numNodes || h < numNodes; {
		next := <-clientChan
		if next.isFinalW {
			wPieces[next.sentID] = next.mtx
			w++
		} else if next.isFinalH {
			hPieces[next.sentID] = next.mtx
			h++
		}
	}
	wg.Wait()

	// Construct W
	w := make([]float64, m*k)
	for i := 0; i < numNodes; i++ {
		for j := 0; j < smallBlockSizeW; j++ {
			for l := 0; l < k; l++ {
				w[(i*smallBlockSizeW*k)+(j*k)+l] = wPieces[i].At(j, l)
			}
		}
	}
	W := mat.NewDense(m, k, w)

	// Construct H
	h := make([]float64, k*n)
	for j := 0; j < k; j++ {
		for i := 0; i < numNodes; i++ {
			for l := 0; l < smallBlockSizeH; l++ {
				h[(j*numNodes*smallBlockSizeH)+(i*smallBlockSizeH)+l] = hPieces[i].At(j, l)
			}
		}
	}
	H := mat.NewDense(k, n, h)

	// fmt.Println("\nW:")
	// matPrint(W)
	// fmt.Println("\nH:")
	// matPrint(H)

	approxA := &mat.Dense{}
	approxA.Mul(W, H)
	fmt.Println("\nApproximation of A:")
	matPrint(approxA)
}
