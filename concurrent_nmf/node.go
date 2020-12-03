package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

const iAmDoneType = 0
const allGatherType = 1
const colGatherType = 2
const rowGatherType = 3

// Node - has info each goroutine needs
type Node struct {
	nodeID int
	// nodeChans     [numNodes]chan *mat.Dense // used to be mat.Matrix
	// inChan        chan *mat.Dense
	nodeChans         [numNodes]chan MatMessage
	inChan            chan MatMessage
	aPiece            mat.Matrix
	allGatherDone     bool // useless
	allReduceDone     bool // useless
	reduceScatterDone bool // useless
}

// MatMessage - give sender information & expected action along with matrix
type MatMessage struct {
	// mtx    *mat.Dense
	mtx     mat.Dense
	sentID  int
	msgType int // 0 = blockForAll, 1 =
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
// Remember - sending a variable thru channel, is giving away that memory (can't use it afterwards - null pointer)

// Like Python all method
func allDone(eachDone [numNodes]bool) bool {
	for i := 0; i < numNodes; i++ {
		if !eachDone[i] {
			return false
		}
	}
	return true
}

// Block on all others finish current collective
func (node *Node) allFinishedAck() {
	// Let everyone know I'm done
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// Give dummy (unused) matrix
			finishedMsg := MatMessage{*mat.NewDense(1, 1, []float64{0}), node.nodeID, iAmDoneType}
			// finishedMsg := MatMessage{nil, node.nodeID}
			ch <- finishedMsg
		}
	}
	var eachDone [numNodes]bool
	eachDone[node.nodeID] = true
	// Wait until everyone done
	for !allDone(eachDone) {
		select {
		case otherMtxMsg := <-node.inChan:
			// oRows, oCols := otherMtxMsg.mtx.Dims()
			if otherMtxMsg.sentID != node.nodeID && otherMtxMsg.msgType == iAmDoneType {
				// if otherMtxMsg.sentID != node.nodeID &&
				// 	(otherMtxMsg.mtx.At(0, 0) == 0 && oRows == 1 && oCols == 1) {
				// if otherMtxMsg.sentID != node.nodeID && otherMtxMsg.mtx == nil {
				eachDone[otherMtxMsg.sentID] = true
			}
		default:
		}
	}
}

// TODO - Make sure each node gets COPY of data, like in real dist system

// Across all nodes
// func (node *Node) allReduce(smallGramMatrix *mat.Dense) [numNodes]*mat.Dense {
func (node *Node) allReduce(smallGramMatrix *mat.Dense) *mat.Dense {
	// var allSmallGramMatrices [numNodes]*mat.Dense
	// allSmallGramMatrices[node.nodeID] = smallGramMatrix
	// FIX change into array of matrices (copies), not array of ptrs (same matrices)
	var allSmallGramMatrices [numNodes]mat.Dense
	allSmallGramMatrices[node.nodeID] = *smallGramMatrix // fine to keep same mem - local to goroutine

	// dummy code
	// uRows, uCols := smallGramMatrix.Dims()
	// fmt.Println("Node", node.nodeID, "In allReduce")

	// Perform allGather
	// send my smallGramMatrix to all others
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// TODO - fix mem issue
			// Only send copies of matrix
			smallGramMatrixMsg := MatMessage{*smallGramMatrix, node.nodeID, allGatherType} // Copies, but we lose pointer, but maybe?
			// smallGramMatrixMsg := MatMessage{mat.DenseCopyOf(smallGramMatrix), node.nodeID}
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
					// Safety check
					if otherMtxMsg.msgType == allGatherType {
						allSmallGramMatrices[otherMtxMsg.sentID] = otherMtxMsg.mtx
						recvSuccess = true
					}
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
			gramMat = &u
		} else {
			// uRows, uCols := u.Dims()
			// gramMatRows, gramMatCols := gramMat.Dims()
			// fmt.Println(uRows, uCols, "=", gramMatRows, gramMatCols)
			gramMat.Add(gramMat, &u)
		}
	}
	// Wait until everyone done
	node.allFinishedAck()
	return gramMat
}

// Combine these 2 methods into 1?
func (node *Node) allGatherAcrossNodeColumns(smallColumnBlock *mat.Dense, hColsPerNode int) mat.Matrix {
	// var allSmallColBlocks [nodeRows]*mat.Dense
	// thisSmallBlockIndex := node.nodeID / nodeCols
	// allSmallColBlocks[thisSmallBlockIndex] = smallColumnBlock
	// FIX change into array of matrices (copies), not array of ptrs (same matrices)
	var allSmallColBlocks [nodeRows]mat.Dense
	thisSmallBlockIndex := node.nodeID / nodeCols
	allSmallColBlocks[thisSmallBlockIndex] = *smallColumnBlock

	// fmt.Println("\nNode", node.nodeID, "All Small Col Blocks Initially:", allSmallColBlocks)
	// fmt.Println("\nNode", node.nodeID, "All Small Col Blocks Initially:")
	// for i := 0; i < nodeRows; i++ {
	// 	if i != thisSmallBlockIndex {
	// 		// matPrint(&allSmallColBlocks[i])
	// 		fmt.Println(allSmallColBlocks[i])
	// 	}
	// }

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
	// fmt.Println("Node", node.nodeID, "column IDs:", colIDs)
	// Perform allGather
	// send my smallColumnBlock to others in my column
	for _, id := range colIDs {
		if id != node.nodeID {
			smallColumnBlockMsg := MatMessage{*smallColumnBlock, node.nodeID, colGatherType}
			// smallColumnBlockMsg := MatMessage{mat.DenseCopyOf(smallColumnBlock), node.nodeID}
			node.nodeChans[id] <- smallColumnBlockMsg
		}
	}
	// Collect from my column others smallColumnBlocks
	// for _, id := range colIDs {
	// ERROR - exiting loop before ALL column-blocks have been gathered
	for i := 0; i < (nodeRows - 1); i++ {
		// if id != node.nodeID {
		// Block on wait for others
		for recvSuccess := false; !recvSuccess; {
			select {
			case otherMtxMsg := <-node.inChan:
				// Safety check
				if otherMtxMsg.msgType == colGatherType {
					thisSmallBlockIndex = otherMtxMsg.sentID / nodeCols
					allSmallColBlocks[thisSmallBlockIndex] = otherMtxMsg.mtx
					recvSuccess = true
				}
			default:
			}
		}
		// }
	}
	// fmt.Println("\nNode", node.nodeID, "All (other) Small Col Blocks After Gather:")
	// for i := 0; i < nodeRows; i++ {
	// 	if i != thisSmallBlockIndex {
	// 		// matPrint(&allSmallColBlocks[i])
	// 		fmt.Println(allSmallColBlocks[i])
	// 	}
	// }

	// for i, x := range allSmallColBlocks {
	// 	fmt.Println("Node", node.nodeID, "small col block", i, "(after gather)")
	// 	nRows, nCols := x.Dims() // ERROR Here - happens b/c of conflict w/in same goroutine columns
	// 	fmt.Println("Node", node.nodeID, "Small Col Block Dims:", nRows, nCols)
	// }

	// Perform concatenate allSmallColBlocks column-wise
	largeBlockCols := n / nodeCols
	x := make([]float64, k*largeBlockCols)
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

	// Wait until everyone done
	node.allFinishedAck()
	return mat.NewDense(k, largeBlockCols, x)
}

// Within W row blocks
func (node *Node) allGatherAcrossNodeRows(smallRowBlock *mat.Dense, wRowsPerNode int) mat.Matrix {
	// var allSmallRowBlocks [nodeCols]*mat.Dense
	// thisSmallBlockIndex := node.nodeID % nodeCols
	// allSmallRowBlocks[thisSmallBlockIndex] = smallRowBlock
	// FIX change into array of matrices (copies), not array of ptrs (same matrices)
	var allSmallRowBlocks [nodeCols]mat.Dense
	thisSmallBlockIndex := node.nodeID % nodeCols
	allSmallRowBlocks[thisSmallBlockIndex] = *smallRowBlock

	// fmt.Println("Node", node.nodeID, "All Small Row Blocks Initially:")
	// for i := 0; i < nodeCols; i++ {
	// 	matPrint(&allSmallRowBlocks[i])
	// }

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
	// fmt.Println("Node", node.nodeID, "row IDs:", rowIDs)
	// Perform allGather
	// send my smallRowBlock to others in my row
	for _, id := range rowIDs {
		if id != node.nodeID {
			smallRowBlockMsg := MatMessage{*smallRowBlock, node.nodeID, rowGatherType}
			// smallRowBlockMsg := MatMessage{mat.DenseCopyOf(smallRowBlock), node.nodeID}
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
				// Safety check
				if otherMtxMsg.msgType == rowGatherType {
					thisSmallBlockIndex = otherMtxMsg.sentID % nodeCols
					allSmallRowBlocks[thisSmallBlockIndex] = otherMtxMsg.mtx
					recvSuccess = true
				}
			default:
			}
		}
		// }
	}

	// for i, x := range allSmallRowBlocks {
	// 	fmt.Println("Node", node.nodeID, "small row block", i, "(after gather)")
	// 	nRows, nCols := x.Dims()
	// 	fmt.Println("Node", node.nodeID, "Small Row Block Dims:", nRows, nCols)
	// }

	// Perform concatenate allSmallRowBlocks row-wise
	largeBlockRows := m / nodeRows
	x := make([]float64, largeBlockRows*k)
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

	// Wait until everyone done
	node.allFinishedAck()
	return mat.NewDense(largeBlockRows, k, x)
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
	// Wait until everyone done
	node.allFinishedAck()
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
	// Wait until everyone done
	node.allFinishedAck()
	return allSmallProductMatrices
}
