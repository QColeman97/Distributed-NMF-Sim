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

// MatMessage - give sender information & expected action along with matrix
type MatMessage struct {
	mtx      mat.Dense
	sentID   int
	isFinalW bool
	isFinalH bool
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

func (node *Node) newReduceScatterAcrossNodeRows(smallRowBlock *mat.Dense) mat.Matrix {
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

	// put those parts together
	reduceProduct := node.localReduce(parts)

	// scatter reduceProduct to others in row evenly
	ret := reduceProduct.Slice(thisSmallBlockIndex*smallBlockSizeW, (thisSmallBlockIndex+1)*smallBlockSizeW, 0, k)

	// vRows, vCols := (m / numNodes), k
	// for i, ch := range node.nodeChans {
	// 	if i != node.nodeID {
	// 		// TODO - fix mem issue
	// 		// Only send copies of matrix
	// 		sliceToScatter := reduceProduct.Slice(i*vRows, (i+1)*vRows, 0, vCols)
	// 		message := mat.DenseCopyOf(sliceToScatter)
	// 		scatterMatrixMsg := MatMessage{*message, node.nodeID, reduceScatterType, node.state}
	// 		ch <- scatterMatrixMsg
	// 	}
	// }

	// wait for all others to have received my matrix
	for i := 0; i < numNodes-1; i++ {
		<-node.aks
	}

	return ret
}

func (node *Node) newReduceScatterAcrossNodeColumns(smallColumnBlock *mat.Dense) mat.Matrix {
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
