package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// MatMessage message types - for synchronization help
const iAmDoneType = 0
const allGatherType = 1
const colGatherType = 2
const rowGatherType = 3
const reduceScatterType = 4

// Node - has info each goroutine needs
type Node struct {
	nodeID int
	// nodeChans     [numNodes]chan *mat.Dense // used to be mat.Matrix
	// inChan        chan *mat.Dense
	nodeChans [numNodes]chan MatMessage
	nodeAks   [numNodes]chan bool
	aks       chan bool
	inChan    chan MatMessage
	aPiece    mat.Matrix
	state     int // monotonically increasing state ID - increment after each collective - for synchronization help
}

// MatMessage - give sender information & expected action along with matrix
type MatMessage struct {
	// mtx    *mat.Dense
	mtx       mat.Dense
	sentID    int
	msgType   int
	sentState int
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

// Remember - sending a variable thru channel, is giving away that memory (can't use it afterwards - null pointer)

// Utility Functions
func in(slice []int, val int) bool {
	for _, item := range slice {
		if item == val {
			return true
		}
	}
	return false
}

// Methods like Python all method
func allTrue(eachDone [numNodes]bool) bool {
	for i := 0; i < numNodes; i++ {
		if !eachDone[i] {
			return false
		}
	}
	return true
}
func allMatricesFilled(eachMatrix [numNodes]mat.Dense) bool {
	for i := 0; i < numNodes; i++ {
		if eachMatrix[i].IsEmpty() {
			return false
		}
	}
	return true
}
func allRowMatricesFilled(eachMatrix [nodeCols]mat.Dense) bool {
	for i := 0; i < nodeCols; i++ {
		if eachMatrix[i].IsEmpty() {
			return false
		}
	}
	return true
}
func allColMatricesFilled(eachMatrix [nodeRows]mat.Dense) bool {
	for i := 0; i < nodeRows; i++ {
		if eachMatrix[i].IsEmpty() {
			return false
		}
	}
	return true
}

// Debug prints for allGathers w/in collective methods
// func debugMatricesFilledV2(eachMatrix []mat.Dense) [numNodes]bool {
// 	var filled [numNodes]bool
// 	for i := 0; i < numNodes; i++ {
// 		filled[i] = !eachMatrix[i].IsEmpty()
// 	}
// 	return filled
// }
func debugMatricesFilled(eachMatrix []mat.Dense) [numNodes]bool {
	var filled [numNodes]bool
	for i := 0; i < numNodes; i++ {
		filled[i] = !eachMatrix[i].IsEmpty()
	}
	return filled
}
func debugRowMatricesFilled(eachMatrix []mat.Dense) [nodeCols]bool {
	var filled [nodeCols]bool
	for i := 0; i < nodeCols; i++ {
		filled[i] = !eachMatrix[i].IsEmpty()
	}
	return filled
}
func debugColMatricesFilled(eachMatrix []mat.Dense) [nodeRows]bool {
	var filled [nodeRows]bool
	for i := 0; i < nodeRows; i++ {
		filled[i] = !eachMatrix[i].IsEmpty()
	}
	return filled
}

// Block on all others finish current collective
func (node *Node) allFinishedAck() {
	// Let everyone know I'm done
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// Give dummy (unused) matrix
			finishedMsg := MatMessage{*mat.NewDense(1, 1, []float64{0}), node.nodeID, iAmDoneType, node.state}
			// finishedMsg := MatMessage{*mat.NewDense(1, 1, []float64{0}), node.nodeID, iAmDoneType}
			// finishedMsg := MatMessage{nil, node.nodeID}
			ch <- finishedMsg
		}
	}
	var eachDone [numNodes]bool
	eachDone[node.nodeID] = true
	// Wait until everyone else done
	for !allTrue(eachDone) {
		select {
		case otherMtxMsg := <-node.inChan:
			// oRows, oCols := otherMtxMsg.mtx.Dims()
			if otherMtxMsg.msgType == iAmDoneType && otherMtxMsg.sentState == node.state {
				// if otherMtxMsg.sentID != node.nodeID && otherMtxMsg.msgType == iAmDoneType && otherMtxMsg.sentState == node.state {
				// if otherMtxMsg.sentID != node.nodeID && otherMtxMsg.msgType == iAmDoneType {
				// if otherMtxMsg.sentID != node.nodeID &&
				// 	(otherMtxMsg.mtx.At(0, 0) == 0 && oRows == 1 && oCols == 1) {
				// if otherMtxMsg.sentID != node.nodeID && otherMtxMsg.mtx == nil {
				eachDone[otherMtxMsg.sentID] = true
			}
		default:
		}
	}
	node.state++
	// fmt.Println(node.nodeID, "checkpoint")
}

func (node *Node) localReduce(parts []mat.Dense) mat.Dense {
	start := parts[0]
	for i := 1; i < len(parts); i++ {
		start.Add(&start, &parts[i])
	}
	return start
}

func (node *Node) newAllReduce(part *mat.Dense) *mat.Dense {
	// send out my part
	for i, c := range node.nodeChans {
		if i != node.nodeID {
			c <- MatMessage{
				mtx:       *part,
				sentID:    node.nodeID,
				msgType:   0,
				sentState: 0,
			}
		}
	}

	parts := make([]mat.Dense, numNodes)
	parts[node.nodeID] = *part

	// get parts from each other node
	done := 1
	for done < numNodes {
		next := <-node.inChan
		parts[next.sentID] = next.mtx
		node.nodeAks[next.sentID] <- true
		done++
	}

	// put those parts together
	ret := node.localReduce(parts)

	// wait for all others to have received my matrix
	for i := 0; i < numNodes-1; i++ {
		<-node.aks
	}

	return &ret
}

// Across all nodes
// func (node *Node) allReduce(smallGramMatrix *mat.Dense) [numNodes]*mat.Dense {
func (node *Node) allReduce(smallGramMatrix *mat.Dense) *mat.Dense {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]", "in allReduce")

	// var allSmallGramMatrices [numNodes]*mat.Dense
	// allSmallGramMatrices[node.nodeID] = smallGramMatrix
	// FIX change into array of matrices (copies), not array of ptrs (same matrices)
	var allSmallGramMatrices [numNodes]mat.Dense
	allSmallGramMatrices[node.nodeID] = *smallGramMatrix // fine to keep same mem - local to goroutine

	// if node.nodeID == 0 {
	// 	fmt.Println("Node 0 all matrices:", debugMatricesFilled(allSmallGramMatrices))
	// 	// fmt.Println("Value of empty matrix:", allSmallGramMatrices[1].At(0, 0))
	// 	// fmt.Println("Value of full matrix:", allSmallGramMatrices[node.nodeID])
	// }

	// dummy code
	// uRows, uCols := smallGramMatrix.Dims()
	// fmt.Println("Node", node.nodeID, "In allReduce")

	// Perform allGather
	// send my smallGramMatrix to all others
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// TODO - fix mem issue
			// Only send copies of matrix
			smallGramMatrixMsg := MatMessage{*smallGramMatrix, node.nodeID, allGatherType, node.state}
			// smallGramMatrixMsg := MatMessage{*smallGramMatrix, node.nodeID, allGatherType} // Copies, but we lose pointer, but maybe?
			// smallGramMatrixMsg := MatMessage{mat.DenseCopyOf(smallGramMatrix), node.nodeID}
			// smallGramMatrixMsg := MatMessage{smallGramMatrix, node.nodeID}
			ch <- smallGramMatrixMsg
		}
	}
	// fmt.Println("---", node.nodeID, "["+strconv.Itoa(node.state)+"]", "in allReduce SENT to all others")
	// ERROR? - not all nodes complete collect others
	// Collect all others smallGramMatrices
	// Error -> don't go through each node, b/c if not in order, will just ignore it - never place it in

	// Block on wait for others
	for !allMatricesFilled(allSmallGramMatrices) {
		select {
		case otherMtxMsg := <-node.inChan:
			// if node.nodeID == 1 {
			// 	fmt.Println("Node 1 got u from", i)
			// }
			// Safety check
			if otherMtxMsg.msgType == allGatherType && otherMtxMsg.sentState == node.state {
				// if otherMtxMsg.msgType == allGatherType && otherMtxMsg.sentState == node.state {
				// if otherMtxMsg.sentID != node.nodeID && otherMtxMsg.msgType == allGatherType &&
				// 	otherMtxMsg.sentState == node.state {
				// if otherMtxMsg.msgType == allGatherType {
				allSmallGramMatrices[otherMtxMsg.sentID] = otherMtxMsg.mtx
				// if node.nodeID == 0 {
				// 	fmt.Println("Node 0 all matrices:", debugMatricesFilled(allSmallGramMatrices))
				// }
			}
		default:
		}

		// dummy code
		// x := make([]float64, uRows*uCols)
		// for i := range x {
		// 	x[i] = rand.NormFloat64()
		// }
		// allSmallGramMatrices[i] = mat.NewDense(uRows, uCols, x)
	}
	// fmt.Println("---", node.nodeID, "["+strconv.Itoa(node.state)+"]", "in allReduce GOT from all others")
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

	// Wait until everyone done, replace below with barrier?
	node.allFinishedAck()
	// fmt.Println("---", node.nodeID, "["+strconv.Itoa(node.state)+"]", "in allReduce ALL NODES DONE")
	return gramMat
}

func (node *Node) localConcatenateColWise(parts []mat.Dense, colsPerNode int) mat.Dense {
	// Perform concatenate allSmallColBlocks column-wise
	largeBlockCols := n / nodeCols
	x := make([]float64, k*largeBlockCols)
	for j := 0; j < k; j++ {
		for i := 0; i < nodeRows; i++ {
			for l := 0; l < colsPerNode; l++ {
				x[i*j*l] = parts[i].At(j, l)
			}
		}
	}

	return *mat.NewDense(k, largeBlockCols, x)
}

func (node *Node) localConcatenateRowWise(parts []mat.Dense, rowsPerNode int) mat.Dense {
	// Perform concatenate allSmallRowBlocks row-wise
	largeBlockRows := m / nodeRows
	x := make([]float64, largeBlockRows*k)
	for i := 0; i < nodeCols; i++ {
		for j := 0; j < rowsPerNode; j++ {
			for l := 0; l < k; l++ {
				x[i*j*l] = parts[i].At(j, l)
			}
		}
	}
	return *mat.NewDense(largeBlockRows, k, x)
}

func (node *Node) newAllGatherAcrossNodeColumns(smallColumnBlock *mat.Dense, hColsPerNode int) mat.Matrix {
	// Only concerned w/ nodes in same column
	thisCol := node.nodeID % nodeCols
	colIDs := make([]int, nodeRows)
	colIDsIdx := 0
	for i := 0; i < numNodes; i++ {
		if (i % nodeCols) == thisCol {
			colIDs[colIDsIdx] = i
			colIDsIdx++
		}
	}

	// send out my part (send to all for synchronization)
	for i, c := range node.nodeChans {
		if i != node.nodeID {
			// if i != node.nodeID && in(colIDs, i) {
			c <- MatMessage{
				mtx:    *smallColumnBlock,
				sentID: node.nodeID,
			}
		}
	}

	parts := make([]mat.Dense, nodeRows)
	thisSmallBlockIndex := node.nodeID / nodeCols
	parts[thisSmallBlockIndex] = *smallColumnBlock

	// get parts from each other node (only record if node in same column)
	done := 1
	for done < numNodes {
		// for done < nodeRows {
		next := <-node.inChan
		if in(colIDs, next.sentID) {
			thisSmallBlockIndex := next.sentID / nodeCols
			parts[thisSmallBlockIndex] = next.mtx
		}
		node.nodeAks[next.sentID] <- true
		done++
	}

	// fmt.Println("Gathered across cols:", debugColMatricesFilled(parts))

	// put those parts together
	ret := node.localConcatenateColWise(parts, hColsPerNode)

	// wait for all others to have received my matrix
	for i := 0; i < numNodes-1; i++ {
		// for i := 0; i < nodeRows-1; i++ {
		<-node.aks
	}

	return &ret
}

// Combine these 2 methods into 1?
func (node *Node) allGatherAcrossNodeColumns(smallColumnBlock *mat.Dense, hColsPerNode int) mat.Matrix {
	// fmt.Println(node.nodeID, "in allGatherCol")

	// var allSmallColBlocks [nodeRows]*mat.Dense
	// thisSmallBlockIndex := node.nodeID / nodeCols
	// allSmallColBlocks[thisSmallBlockIndex] = smallColumnBlock
	// FIX changed into array of matrices (copies), not array of ptrs (same matrices)
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
			smallColumnBlockMsg := MatMessage{*smallColumnBlock, node.nodeID, colGatherType, node.state}
			// smallColumnBlockMsg := MatMessage{mat.DenseCopyOf(smallColumnBlock), node.nodeID}
			node.nodeChans[id] <- smallColumnBlockMsg
		}
	}
	// Collect from my column others smallColumnBlocks
	// for _, id := range colIDs {
	// ERROR - exiting loop before ALL column-blocks have been gathered
	// Block on wait for others
	for !allColMatricesFilled(allSmallColBlocks) {
		// for i := 0; i < (nodeRows - 1); i++ {
		// if id != node.nodeID {
		// for recvSuccess := false; !recvSuccess; {
		select {
		case otherMtxMsg := <-node.inChan:
			// Safety check
			if otherMtxMsg.msgType == colGatherType && otherMtxMsg.sentState == node.state {
				thisSmallBlockIndex = otherMtxMsg.sentID / nodeCols
				allSmallColBlocks[thisSmallBlockIndex] = otherMtxMsg.mtx
				// recvSuccess = true
			}
		default:
		}
		// }
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

	// Wait until everyone done
	node.allFinishedAck()
	return mat.NewDense(k, largeBlockCols, x)
}

func (node *Node) allGatherAcrossNodeColumnsDummy(smallColumnBlock *mat.Dense, hColsPerNode int) mat.Matrix {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]in allGatherCol")

	largeBlockCols := n / nodeCols
	x := make([]float64, k*largeBlockCols)
	for i := range x {
		x[i] = rand.NormFloat64()
	}
	// Wait until everyone done
	// node.allFinishedAck()
	// fmt.Println(node.nodeID, "in allGatherCol ALL done!")
	return mat.NewDense(k, largeBlockCols, x)
}

func (node *Node) newAllGatherAcrossNodeRows(smallRowBlock *mat.Dense, wRowsPerNode int) mat.Matrix {
	// Only concerned w/ nodes in same row
	thisRow := node.nodeID / nodeCols
	rowIDs := make([]int, nodeCols)
	rowIDsIdx := 0
	for i := 0; i < numNodes; i++ {
		if (i / nodeCols) == thisRow {
			rowIDs[rowIDsIdx] = i
			rowIDsIdx++
		}
	}

	// send out my part (send to all for synchronization)
	for i, c := range node.nodeChans {
		if i != node.nodeID {
			c <- MatMessage{
				mtx:    *smallRowBlock,
				sentID: node.nodeID,
			}
		}
	}

	parts := make([]mat.Dense, nodeCols)
	thisSmallBlockIndex := node.nodeID % nodeCols
	parts[thisSmallBlockIndex] = *smallRowBlock

	// get parts from each other node (only record if node in same row)
	done := 1
	for done < numNodes {
		next := <-node.inChan
		if in(rowIDs, next.sentID) {
			thisSmallBlockIndex := next.sentID % nodeCols
			parts[thisSmallBlockIndex] = next.mtx
		}
		node.nodeAks[next.sentID] <- true
		done++
	}

	// fmt.Println("Gathered across rows:", debugRowMatricesFilled(parts))

	// put those parts together
	ret := node.localConcatenateRowWise(parts, wRowsPerNode)

	// wait for all others to have received my matrix
	for i := 0; i < numNodes-1; i++ {
		<-node.aks
	}

	return &ret
}

// Within W row blocks
func (node *Node) allGatherAcrossNodeRows(smallRowBlock *mat.Dense, wRowsPerNode int) mat.Matrix {
	// fmt.Println(node.nodeID, "in allGatherRow")

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
			smallRowBlockMsg := MatMessage{*smallRowBlock, node.nodeID, rowGatherType, node.state}
			// smallRowBlockMsg := MatMessage{mat.DenseCopyOf(smallRowBlock), node.nodeID}
			node.nodeChans[id] <- smallRowBlockMsg
		}
	}
	// Collect from my row others smallRowBlocks
	// for _, id := range rowIDs {
	// Block on wait for others
	for !allRowMatricesFilled(allSmallRowBlocks) {
		// for i := 0; i < (nodeCols - 1); i++ {
		// if id != node.nodeID {
		// for recvSuccess := false; !recvSuccess; {
		select {
		case otherMtxMsg := <-node.inChan:
			// Safety check
			if otherMtxMsg.msgType == rowGatherType && otherMtxMsg.sentState == node.state {
				thisSmallBlockIndex = otherMtxMsg.sentID % nodeCols
				allSmallRowBlocks[thisSmallBlockIndex] = otherMtxMsg.mtx
				// recvSuccess = true
			}
		default:
		}
		// }
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

func (node *Node) allGatherAcrossNodeRowsDummy(smallRowBlock *mat.Dense, wRowsPerNode int) mat.Matrix {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]in allGatherRow")

	largeBlockRows := m / nodeRows
	x := make([]float64, largeBlockRows*k)
	for i := range x {
		x[i] = rand.NormFloat64()
	}
	// Wait until everyone done
	// node.allFinishedAck()
	// fmt.Println(node.nodeID, "in allGatherRow ALL done!")
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

func (node *Node) reduceScatterAcrossNodeRows(smallGramMatrix *mat.Dense) *mat.Dense {
	var allSmallGramMatrices [numNodes]mat.Dense
	allSmallGramMatrices[node.nodeID] = *smallGramMatrix // fine to keep same mem - local to goroutine

	// Perform allGather
	// send my smallGramMatrix to all others
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// TODO - fix mem issue
			// Only send copies of matrix
			smallGramMatrixMsg := MatMessage{*smallGramMatrix, node.nodeID, allGatherType, node.state}
			ch <- smallGramMatrixMsg
		}
	}

	// Block on wait for others
	for !allMatricesFilled(allSmallGramMatrices) {
		select {
		case otherMtxMsg := <-node.inChan:
			// Safety check
			if otherMtxMsg.msgType == allGatherType && otherMtxMsg.sentState == node.state {
				allSmallGramMatrices[otherMtxMsg.sentID] = otherMtxMsg.mtx
			}
		default:
		}
	}

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

	// scatter gramMat to all others evenly
	vRows, vCols := (m / numNodes), k
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// TODO - fix mem issue
			// Only send copies of matrix
			sliceToScatter := gramMat.Slice(i*vRows, (i+1)*vRows, 0, vCols)
			message := mat.DenseCopyOf(sliceToScatter)
			scatterMatrixMsg := MatMessage{*message, node.nodeID, reduceScatterType, node.state}
			ch <- scatterMatrixMsg
		}
	}

	// Wait until everyone done
	node.allFinishedAck()
	return gramMat
}

func (node *Node) reduceScatterAcrossNodeColumns(smallGramMatrix *mat.Dense) *mat.Dense {
	var allSmallGramMatrices [numNodes]mat.Dense
	allSmallGramMatrices[node.nodeID] = *smallGramMatrix // fine to keep same mem - local to goroutine

	// Perform allGather
	// send my smallGramMatrix to all others
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// TODO - fix mem issue
			// Only send copies of matrix
			smallGramMatrixMsg := MatMessage{*smallGramMatrix, node.nodeID, allGatherType, node.state}
			ch <- smallGramMatrixMsg
		}
	}

	// Block on wait for others
	for !allMatricesFilled(allSmallGramMatrices) {
		select {
		case otherMtxMsg := <-node.inChan:
			// Safety check
			if otherMtxMsg.msgType == allGatherType && otherMtxMsg.sentState == node.state {
				allSmallGramMatrices[otherMtxMsg.sentID] = otherMtxMsg.mtx
			}
		default:
		}
	}

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

	// scatter gramMat to all others evenly
	yCols := n / numNodes
	for i, ch := range node.nodeChans {
		if i != node.nodeID {
			// TODO - fix mem issue
			// Only send copies of matrix
			sliceToScatter := gramMat.Slice(0, k, i*yCols, (i+1)*yCols)
			message := mat.DenseCopyOf(sliceToScatter)
			scatterMatrixMsg := MatMessage{*message, node.nodeID, reduceScatterType, node.state}
			ch <- scatterMatrixMsg
		}
	}

	// Wait until everyone done
	node.allFinishedAck()
	return gramMat
}

// Combine these 2 methods into 1?
func (node *Node) reduceScatterAcrossNodeRowsDummy(smallProductMatrix *mat.Dense) *mat.Dense {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]in reduceScatterRow")
	smallBlockRows := m / numNodes
	x := make([]float64, smallBlockRows*k)
	for i := range x {
		x[i] = rand.NormFloat64()
	}

	// Wait until everyone done
	// node.allFinishedAck()
	// fmt.Println(node.nodeID, "in reduceScatterRow ALL done!")
	return mat.NewDense(smallBlockRows, k, x)
}

func (node *Node) reduceScatterAcrossNodeColumnsDummy(smallProductMatrix *mat.Dense) *mat.Dense {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]in reduceScatterRow")
	smallBlockCols := n / numNodes
	x := make([]float64, k*smallBlockCols)
	for i := range x {
		x[i] = rand.NormFloat64()
	}

	// Wait until everyone done
	// node.allFinishedAck()
	// fmt.Println(node.nodeID, "in reduceScatterCol ALL done!")
	return mat.NewDense(k, smallBlockCols, x)
}
