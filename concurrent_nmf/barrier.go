package main

import "sync"

// Taken concept from this:
// https://medium.com/golangspec/reusable-barriers-in-golang-156db1f75d0b

// Barrier - for synchronization
// 6 gates to be placed after each collective call in 1 NMF loop iteration
type Barrier struct {
	c     int
	n     int
	m     sync.Mutex
	gate1 chan int
	gate2 chan int
	gate3 chan int
	gate4 chan int
	gate5 chan int
	gate6 chan int
}

// initBarrier (New) - initialize barrier
func initBarrier(n int) *Barrier {
	b := Barrier{
		n:     n,
		gate1: make(chan int, n),
		gate2: make(chan int, n),
		gate3: make(chan int, n),
		gate4: make(chan int, n),
		gate5: make(chan int, n),
		gate6: make(chan int, n),
	}
	// open all gates, except for first gate
	b.gate2 <- 1
	b.gate3 <- 1
	b.gate4 <- 1
	b.gate5 <- 1
	b.gate6 <- 1
	return &b
}

// Opens gate 1, and closes gate 2
func (b *Barrier) openGate1CloseGate2() {
}

// etc.
func (b *Barrier) openGate2() {
}
func (b *Barrier) openGate3() {
}
func (b *Barrier) openGate4() {
}
func (b *Barrier) openGate5() {
}

// Opens gate 6, closes gate 1
func (b *Barrier) openGate6() {
}
