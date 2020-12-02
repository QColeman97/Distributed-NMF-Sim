package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Node - has info each goroutine needs
type Node struct {
	nodeID        int
	nodeChans     [numNodes]chan mat.Matrix
	inChan        chan mat.Matrix
	aPiece        mat.Matrix
	allGatherDone bool
	allReduceDone bool
	scatterDone   bool
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

// Ryan
// For making sure all goroutines are executing the same method at the same time, we need something like MPI_Barrier
// i.e. "wait here until all coroutines get here"
// possibly a sync.WaitGroup

// Across all nodes
func (node *Node) allReduce(smallGramMatrix *mat.Dense) [numNodes]*mat.Dense {
	var allSmallGramMatrices [numNodes]*mat.Dense

	uRows, uCols := smallGramMatrix.Dims()

	// temp
	for i := 0; i < numNodes; i++ {
		x := make([]float64, uRows*uCols)
		for i := range x {
			x[i] = rand.NormFloat64()
		}
		// mat = mat.NewDense(uRows, uCols, x)
		// allSmallGramMatrices[i] = &mat
		allSmallGramMatrices[i] = mat.NewDense(uRows, uCols, x)
	}

	return allSmallGramMatrices
}

// Combine these 2 methods into 1?
func (node *Node) allGatherAcrossNodeColumns() mat.Matrix {
	// temp
	cols := n / nodeCols
	x := make([]float64, k*cols)
	for i := range x {
		x[i] = rand.NormFloat64()
	}
	return mat.NewDense(k, cols, x)
}

func (node *Node) allGatherAcrossNodeRows() mat.Matrix {
	// temp
	rows := m / nodeRows
	x := make([]float64, rows*k)
	for i := range x {
		x[i] = rand.NormFloat64()
	}
	return mat.NewDense(rows, k, x)
}

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
