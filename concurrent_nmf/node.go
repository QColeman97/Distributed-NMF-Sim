package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Node - has info each goroutine needs
type Node struct {
	nodeID int
	// nodeChans     [numNodes]chan *mat.Dense // used to be mat.Matrix
	// inChan        chan *mat.Dense
	nodeChans     [numNodes]chan MatMessage
	inChan        chan MatMessage
	aPiece        mat.Matrix
	allGatherDone bool
	allReduceDone bool
	scatterDone   bool
}

type MatMessage struct {
	mtx    *mat.Dense
	sentID int
}

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

// Remember butterfly pattern for all-collectives

// Across all nodes
// func (node *Node) allReduce(smallGramMatrix *mat.Dense) [numNodes]*mat.Dense {
func (node *Node) allReduce(smallGramMatrix *mat.Dense) *mat.Dense {
	var allSmallGramMatrices [numNodes]*mat.Dense
	allSmallGramMatrices[node.nodeID] = smallGramMatrix
	// dummy code
	// uRows, uCols := smallGramMatrix.Dims()

	// Perform allGather
	// send my smallGramMatrix to all others
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// Only send copies of matrix
			smallGramMatrixMsg := MatMessage{mat.DenseCopyOf(smallGramMatrix), node.nodeID}
			// smallGramMatrixMsg := MatMessage{smallGramMatrix, node.nodeID}
			ch <- smallGramMatrixMsg
		}
	}
	// Collect all others smallGramMatrices
	for i := 0; i < numNodes; i++ {
		if i != node.nodeID {
			// Block on wait for others
			for recvSuccess := false; !recvSuccess; {
				select {
				case otherMtxMsg := <-node.inChan:
					// if node.nodeID == 1 {
					// 	fmt.Println("Node 1 got u from", i)
					// }
					allSmallGramMatrices[otherMtxMsg.sentID] = otherMtxMsg.mtx
					recvSuccess = true
				default:
				}
			}
		}

		// dummy code
		// x := make([]float64, uRows*uCols)
		// for i := range x {
		// 	x[i] = rand.NormFloat64()
		// }
		// allSmallGramMatrices[i] = mat.NewDense(uRows, uCols, x)
	}
	// return allSmallGramMatrices

	// Perform reduce
	gramMat := &mat.Dense{} // k x k
	for i, u := range allSmallGramMatrices {
		if i == 0 {
			gramMat = u
		} else {
			// uRows, uCols := u.Dims()
			// gramMatRows, gramMatCols := gramMat.Dims()
			// fmt.Println(uRows, uCols, "=", gramMatRows, gramMatCols)
			gramMat.Add(gramMat, u)
		}
	}

	return gramMat
}

// Combine these 2 methods into 1?
func (node *Node) allGatherAcrossNodeColumns(smallColumnBlock *mat.Dense, hColsPerNode int) mat.Matrix {
	var allSmallColBlocks [nodeRows]*mat.Dense
	thisSmallBlockIndex := node.nodeID / nodeCols
	allSmallColBlocks[thisSmallBlockIndex] = smallColumnBlock

	// Only concerned w/ nodes in same column
	thisCol := node.nodeID % nodeCols
	var colIDs [nodeRows]int
	colIDsIdx := 0
	for i := 0; i < numNodes; i++ {
		if (i % nodeCols) == thisCol {
			colIDs[colIDsIdx] = i
			colIDsIdx++
		}
	}
	// Perform allGather
	// send my smallColumnBlock to others in my column
	for _, id := range colIDs {
		if id != node.nodeID {
			smallColumnBlockMsg := MatMessage{smallColumnBlock, node.nodeID}
			node.nodeChans[id] <- smallColumnBlockMsg
		}
	}
	// Collect from my column others smallColumnBlocks
	// for _, id := range colIDs {
	for i := 0; i < (nodeRows - 1); i++ {
		// if id != node.nodeID {
		// Block on wait for others
		for recvSuccess := false; !recvSuccess; {
			select {
			case otherMtxMsg := <-node.inChan:
				thisSmallBlockIndex = otherMtxMsg.sentID / nodeCols
				allSmallColBlocks[thisSmallBlockIndex] = otherMtxMsg.mtx
				recvSuccess = true
			default:
			}
		}
		// }
	}
	// Concatenate allSmallColBlocks column-wise
	cols := n / nodeCols
	x := make([]float64, k*cols)
	for j := 0; j < k; j++ {
		for i := 0; i < nodeRows; i++ {
			for l := 0; l < hColsPerNode; l++ {
				x[i*j*l] = allSmallColBlocks[i].At(j, l)
			}
		}
	}

	// temp
	// cols := n / nodeCols
	// x := make([]float64, k*cols)
	// for i := range x {
	// 	x[i] = rand.NormFloat64()
	// }
	return mat.NewDense(k, cols, x)
}

// Within W row blocks
func (node *Node) allGatherAcrossNodeRows(smallRowBlock *mat.Dense, wRowsPerNode int) mat.Matrix {
	var allSmallRowBlocks [nodeCols]*mat.Dense
	thisSmallBlockIndex := node.nodeID % nodeCols
	allSmallRowBlocks[thisSmallBlockIndex] = smallRowBlock

	// Only concerned w/ nodes in same row
	thisRow := node.nodeID / nodeCols
	var rowIDs [nodeCols]int
	rowIDsIdx := 0
	for i := 0; i < numNodes; i++ {
		if (i / nodeCols) == thisRow {
			rowIDs[rowIDsIdx] = i
			rowIDsIdx++
		}
	}
	// Perform allGather
	// send my smallRowBlock to others in my row
	for _, id := range rowIDs {
		if id != node.nodeID {
			smallRowBlockMsg := MatMessage{smallRowBlock, node.nodeID}
			node.nodeChans[id] <- smallRowBlockMsg
		}
	}
	// Collect from my row others smallRowBlocks
	// for _, id := range rowIDs {
	for i := 0; i < (nodeCols - 1); i++ {
		// if id != node.nodeID {
		// Block on wait for others
		for recvSuccess := false; !recvSuccess; {
			select {
			case otherMtxMsg := <-node.inChan:
				thisSmallBlockIndex = otherMtxMsg.sentID % nodeCols
				allSmallRowBlocks[thisSmallBlockIndex] = otherMtxMsg.mtx
				recvSuccess = true
			default:
			}
		}
		// }
	}
	// Concatenate allSmallRowBlocks row-wise
	rows := m / nodeRows
	x := make([]float64, rows*k)
	for i := 0; i < nodeCols; i++ {
		for j := 0; j < wRowsPerNode; j++ {
			for l := 0; l < k; l++ {
				x[i*j*l] = allSmallRowBlocks[i].At(j, l)
			}
		}
	}

	// temp
	// rows := m / nodeRows
	// x := make([]float64, rows*k)
	// for i := range x {
	// 	x[i] = rand.NormFloat64()
	// }
	return mat.NewDense(rows, k, x)
}

// // Only performs across node rows or columns
// func (node *Node) allGatherAcross(rows bool, smallBlock *mat.Dense) mat.Matrix {
// 	var allSmallBlocks [nodeCols]*mat.Dense
// 	allSmallBlocks[node.nodeID] = smallBlock

// 	inBlockIDs := []int{}
// 	if rows {
// 		nodeRow := node.nodeID / nodeCols
// 		// rowIDs := []int{}
// 		for i := 0; i < numNodes; i++ {
// 			if (i / nodeCols) == nodeRow {
// 				// rowIDs = append(rowIDs, i)
// 				inBlockIDs = append(inBlockIDs, i)
// 			}
// 		}
// 	} else {
// 		nodeCol := node.nodeID % nodeCols
// 		// colIDs := []int{}
// 		for i := 0; i < numNodes; i++ {
// 			if (i % nodeCols) == nodeCol {
// 				// colIDs = append(colIDs, i)
// 				inBlockIDs = append(inBlockIDs, i)
// 			}
// 		}
// 	}

// 	// Perform allGather
// 	// send my smallRowBlock to others in my row
// 	for _, id := range inBlockIDs {
// 		if id != node.nodeID {
// 			node.nodeChans[id] <- smallBlock
// 		}
// 	}
// 	// Collect from my row others smallRowBlocks
// 	for _, id := range inBlockIDs {
// 		if id != node.nodeID {
// 			select {
// 			case otherMtx := <-node.inChan:
// 				allSmallBlocks[id] = otherMtx
// 			default:
// 			}
// 		}
// 	}

// 	retMatrix := &mat.Dense{}
// 	if rows {
// 		wRowsPerNode := m / numNodes // m/p
// 		// Concatenate allSmallRowBlocks along rows
// 		numRows := m / nodeRows
// 		x := make([]float64, numRows*k)
// 		for i := 0; i < 2; i++ {
// 			for j := 0; j < wRowsPerNode; j++ {
// 				for l := 0; l < k; l++ {
// 					x[i*j*l] = allSmallBlocks[i].At(j, l)
// 				}
// 			}
// 		}
// 		retMatrix = mat.NewDense(numRows, k, x)
// 		// return mat.NewDense(numRows, k, x)
// 	} else {
// 		hColsPerNode := n / numNodes // n/p
// 		// Concatenate allSmallRowBlocks along rows
// 		numCols := n / nodeCols
// 		x := make([]float64, k*numCols)
// 		for i := 0; i < 2; i++ {
// 			for l := 0; l < k; l++ {
// 				// for j := 0; j < wRowsPerNode; j++ {
// 				for j := 0; j < hColsPerNode; j++ {
// 					// for l := 0; l < k; l++ {
// 					x[i*j*l] = allSmallBlocks[i].At(j, l)
// 				}
// 			}
// 		}
// 		retMatrix = mat.NewDense(k, numCols, x)
// 		// return mat.NewDense(k, numCols, x)
// 	}

// 	// temp
// 	// rows := m / nodeRows
// 	// x := make([]float64, rows*k)
// 	// for i := range x {
// 	// 	x[i] = rand.NormFloat64()
// 	// }
// 	// return mat.NewDense(rows, k, x)
// 	return retMatrix
// }

// Combine these 2 methods into 1?
func (node *Node) reduceScatterAcrossNodeRows(smallProductMatrix *mat.Dense) [nodeRows]*mat.Dense {
	var allSmallProductMatrices [nodeRows]*mat.Dense

	// vRows, vCols := smallProductMatrix.Dims()
	vRows, vCols := (m / numNodes), k

	// temp
	for i := 0; i < nodeRows; i++ {
		x := make([]float64, vRows*vCols)
		for i := range x {
			x[i] = rand.NormFloat64()
		}
		// mat = mat.NewDense(vRows, vCols, x)
		// allSmallProductMatrices[i] = &mat
		allSmallProductMatrices[i] = mat.NewDense(vRows, vCols, x)
	}

	return allSmallProductMatrices
}

func (node *Node) reduceScatterAcrossNodeColumns(smallProductMatrix *mat.Dense) [nodeCols]*mat.Dense {
	var allSmallProductMatrices [nodeCols]*mat.Dense

	// yRows, yCols := smallProductMatrix.Dims()
	yRows, yCols := k, (n / numNodes)

	// temp
	for i := 0; i < nodeCols; i++ {
		y := make([]float64, yRows*yCols)
		for i := range y {
			y[i] = rand.NormFloat64()
		}
		// mat = mat.NewDense(yRows, yCols, y)
		// allSmallProductMatrices[i] = &mat
		allSmallProductMatrices[i] = mat.NewDense(yRows, yCols, y)
	}

	return allSmallProductMatrices
}
