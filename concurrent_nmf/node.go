package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Node - has info each goroutine needs
type Node struct {
	nodeID     int
	nodeChans  [numNodes]chan MatMessage
	nodeAks    [numNodes]chan bool
	aks        chan bool
	inChan     chan MatMessage
	clientChan chan MatMessage
	aPiece     mat.Matrix
}

// MatMessage - give sender ID & extra info along with matrix
type MatMessage struct {
	mtx      mat.Dense
	sentID   int
	isFinalW bool // for return to client
	isFinalH bool // for return to client
}

// Implement MPI collectives
//	- reduce-scatter 	- used across all proc rows/columns
//		every node retrieve V/Y from every node
//		every node performs reduction & returns piece of reduction
//  - all-gather 		- used across all proc columns/rows
//		every node send their Hj/Wi to every node in row/col
//		return concatenation of pieces
//	- all-reduce 		- used across all proc
//		every node retrieve U/X from every node
//		every node performs & returns reduction

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

func (node *Node) localReduce(parts []mat.Dense) mat.Dense {
	start := parts[0]
	for i := 1; i < len(parts); i++ {
		start.Add(&start, &parts[i])
	}
	return start
}

func (node *Node) allReduce(part *mat.Dense) *mat.Dense {
	// send out my part
	for i, c := range node.nodeChans {
		if i != node.nodeID {
			c <- MatMessage{
				mtx:    *part,
				sentID: node.nodeID,
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

func (node *Node) localConcatenateColWise(parts []mat.Dense) mat.Dense {
	// Perform concatenate column-wise
	x := make([]float64, k*largeBlockSizeH)
	for j := 0; j < k; j++ {
		for i := 0; i < numNodeRows; i++ {
			for l := 0; l < smallBlockSizeH; l++ {
				x[i*j*l] = parts[i].At(j, l)
			}
		}
	}

	return *mat.NewDense(k, largeBlockSizeH, x)
}

func (node *Node) localConcatenateRowWise(parts []mat.Dense) mat.Dense {
	// Perform concatenate row-wise
	x := make([]float64, largeBlockSizeW*k)
	for i := 0; i < numNodeCols; i++ {
		for j := 0; j < smallBlockSizeW; j++ {
			for l := 0; l < k; l++ {
				x[i*j*l] = parts[i].At(j, l)
			}
		}
	}
	return *mat.NewDense(largeBlockSizeW, k, x)
}

func (node *Node) allGatherAcrossNodeColumns(smallColumnBlock *mat.Dense) mat.Matrix {
	// Only concerned w/ nodes in same column
	thisCol := node.nodeID % numNodeCols
	colIDs := make([]int, numNodeRows)
	colIDsIdx := 0
	for i := 0; i < numNodes; i++ {
		if (i % numNodeCols) == thisCol {
			colIDs[colIDsIdx] = i
			colIDsIdx++
		}
	}

	// send out my part (send to all for synchronization)
	for i, c := range node.nodeChans {
		if i != node.nodeID {
			c <- MatMessage{
				mtx:    *smallColumnBlock,
				sentID: node.nodeID,
			}
		}
	}

	parts := make([]mat.Dense, numNodeRows)
	thisSmallBlockIndex := node.nodeID / numNodeCols
	parts[thisSmallBlockIndex] = *smallColumnBlock

	// get parts from each other node (only record if node in same column)
	done := 1
	for done < numNodes {
		next := <-node.inChan
		if in(colIDs, next.sentID) {
			thisSmallBlockIndex := next.sentID / numNodeCols
			parts[thisSmallBlockIndex] = next.mtx
		}
		node.nodeAks[next.sentID] <- true
		done++
	}

	// put those parts together
	ret := node.localConcatenateColWise(parts)

	// wait for all others to have received my matrix
	for i := 0; i < numNodes-1; i++ {
		<-node.aks
	}

	return &ret
}

func (node *Node) allGatherAcrossNodeColumnsDummy(smallColumnBlock *mat.Dense) mat.Matrix {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]in allGatherCol")
	x := make([]float64, k*largeBlockSizeH)
	for i := range x {
		x[i] = rand.NormFloat64()
	}

	return mat.NewDense(k, largeBlockSizeH, x)
}

func (node *Node) allGatherAcrossNodeRows(smallRowBlock *mat.Dense) mat.Matrix {
	// Only concerned w/ nodes in same row
	thisRow := node.nodeID / numNodeCols
	rowIDs := make([]int, numNodeCols)
	rowIDsIdx := 0
	for i := 0; i < numNodes; i++ {
		if (i / numNodeCols) == thisRow {
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

	parts := make([]mat.Dense, numNodeCols)
	thisSmallBlockIndex := node.nodeID % numNodeCols
	parts[thisSmallBlockIndex] = *smallRowBlock

	// get parts from each other node (only record if node in same row)
	done := 1
	for done < numNodes {
		next := <-node.inChan
		if in(rowIDs, next.sentID) {
			thisSmallBlockIndex := next.sentID % numNodeCols
			parts[thisSmallBlockIndex] = next.mtx
		}
		node.nodeAks[next.sentID] <- true
		done++
	}

	// put those parts together
	ret := node.localConcatenateRowWise(parts)

	// wait for all others to have received my matrix
	for i := 0; i < numNodes-1; i++ {
		<-node.aks
	}

	return &ret
}

func (node *Node) allGatherAcrossNodeRowsDummy(smallRowBlock *mat.Dense) mat.Matrix {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]in allGatherRow")
	x := make([]float64, largeBlockSizeW*k)
	for i := range x {
		x[i] = rand.NormFloat64()
	}

	// fmt.Println(node.nodeID, "in allGatherRow ALL done!")
	return mat.NewDense(largeBlockSizeW, k, x)
}

func (node *Node) reduceScatterAcrossNodeRows(smallRowBlock *mat.Dense) mat.Matrix {
	// Only concerned w/ nodes in same row
	thisRow := node.nodeID / numNodeCols
	rowIDs := make([]int, numNodeCols)
	rowIDsIdx := 0
	for i := 0; i < numNodes; i++ {
		if (i / numNodeCols) == thisRow {
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

	parts := make([]mat.Dense, numNodeCols)
	thisSmallBlockIndex := node.nodeID % numNodeCols
	parts[thisSmallBlockIndex] = *smallRowBlock

	// get parts from each other node (only record if node in same row)
	done := 1
	for done < numNodes {
		next := <-node.inChan
		if in(rowIDs, next.sentID) {
			thisSmallBlockIndex := next.sentID % numNodeCols
			parts[thisSmallBlockIndex] = next.mtx
		}
		node.nodeAks[next.sentID] <- true
		done++
	}

	// put those parts together
	reduceProduct := node.localReduce(parts)

	// scatter reduceProduct to others in row evenly
	ret := reduceProduct.Slice(thisSmallBlockIndex*smallBlockSizeW, (thisSmallBlockIndex+1)*smallBlockSizeW, 0, k)

	// wait for all others to have received my matrix
	for i := 0; i < numNodes-1; i++ {
		<-node.aks
	}

	return ret
}

func (node *Node) reduceScatterAcrossNodeColumns(smallColumnBlock *mat.Dense) mat.Matrix {
	// Only concerned w/ nodes in same column
	thisCol := node.nodeID % numNodeCols
	colIDs := make([]int, numNodeRows)
	colIDsIdx := 0
	for i := 0; i < numNodes; i++ {
		if (i % numNodeCols) == thisCol {
			colIDs[colIDsIdx] = i
			colIDsIdx++
		}
	}

	// send out my part (send to all for synchronization)
	for i, c := range node.nodeChans {
		if i != node.nodeID {
			c <- MatMessage{
				mtx:    *smallColumnBlock,
				sentID: node.nodeID,
			}
		}
	}

	parts := make([]mat.Dense, numNodeRows)
	thisSmallBlockIndex := node.nodeID / numNodeCols
	parts[thisSmallBlockIndex] = *smallColumnBlock

	// get parts from each other node (only record if node in same column)
	done := 1
	for done < numNodes {
		// for done < numNodeRows {
		next := <-node.inChan
		if in(colIDs, next.sentID) {
			thisSmallBlockIndex := next.sentID / numNodeCols
			parts[thisSmallBlockIndex] = next.mtx
		}
		node.nodeAks[next.sentID] <- true
		done++
	}

	// put those parts together
	reduceProduct := node.localReduce(parts)

	// scatter reduceProduct to others in row evenly
	ret := reduceProduct.Slice(0, k, thisSmallBlockIndex*smallBlockSizeH, (thisSmallBlockIndex+1)*smallBlockSizeH)

	// wait for all others to have received my matrix
	for i := 0; i < numNodes-1; i++ {
		<-node.aks
	}

	return ret
}

// Combine these 2 methods into 1?
func (node *Node) reduceScatterAcrossNodeRowsDummy(smallProductMatrix *mat.Dense) *mat.Dense {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]in reduceScatterRow")
	x := make([]float64, smallBlockSizeW*k)
	for i := range x {
		x[i] = rand.NormFloat64()
	}

	// fmt.Println(node.nodeID, "in reduceScatterRow ALL done!")
	return mat.NewDense(smallBlockSizeW, k, x)
}

func (node *Node) reduceScatterAcrossNodeColumnsDummy(smallProductMatrix *mat.Dense) *mat.Dense {
	// fmt.Println(node.nodeID, "["+strconv.Itoa(node.state)+"]in reduceScatterRow")
	x := make([]float64, k*smallBlockSizeH)
	for i := range x {
		x[i] = rand.NormFloat64()
	}

	// fmt.Println(node.nodeID, "in reduceScatterCol ALL done!")
	return mat.NewDense(k, smallBlockSizeH, x)
}
