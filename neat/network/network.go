package network

import (
	"bytes"
	"errors"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat/math"
)

// Network is a LIST of input NODEs and a LIST of output NODEs.
// The point of the network is to define a single entity which can evolve
// or learn on its own, even though it may be part of a larger framework.
type Network struct {
	// A network id
	Id int
	// Is a name of this network */
	Name string
	// NNodes that output from the network
	Outputs []*NNode

	// The number of links in the net (-1 means not yet counted)
	numLinks int
	// A list of all the nodes in the network except MIMO control ones
	allNodes []*NNode
	// NNodes that input into the network
	inputs []*NNode
	// NNodes that connect network modules
	controlNodes []*NNode
}

// NewNetwork Creates new network
func NewNetwork(in, out, all []*NNode, netId int) *Network {
	n := Network{
		Id:       netId,
		inputs:   in,
		Outputs:  out,
		allNodes: all,
		numLinks: -1,
	}
	return &n
}

// NewModularNetwork Creates new modular network with control nodes
func NewModularNetwork(in, out, all, control []*NNode, netId int) *Network {
	n := NewNetwork(in, out, all, netId)
	n.controlNodes = control
	return n
}

// FastNetworkSolver Creates fast network solver based on the architecture of this network. It's primarily aimed for
// big networks to improve processing speed.
func (n *Network) FastNetworkSolver() (Solver, error) {
	// calculate neurons per layer
	outputNeuronCount := len(n.Outputs)
	// build bias, input and hidden neurons lists
	biasNeuronCount := 0
	inList := make([]*NNode, 0)
	biasList := make([]*NNode, 0)
	hiddenList := make([]*NNode, 0)
	for _, ne := range n.allNodes {
		switch ne.NeuronType {
		case BiasNeuron:
			biasNeuronCount += 1
			biasList = append(biasList, ne)
		case InputNeuron:
			inList = append(inList, ne)
		case HiddenNeuron:
			hiddenList = append(hiddenList, ne)
		}
	}
	inputNeuronCount := len(inList)
	totalNeuronCount := len(n.allNodes)

	// create activation functions array
	activations := make([]math.NodeActivationType, totalNeuronCount)
	neuronLookup := make(map[int]int) // id:index

	// walk through neuron nodes in order: bias, input, output, hidden
	neuronIndex := processList(0, biasList, activations, neuronLookup)
	neuronIndex = processList(neuronIndex, inList, activations, neuronLookup)
	neuronIndex = processList(neuronIndex, n.Outputs, activations, neuronLookup)
	processList(neuronIndex, hiddenList, activations, neuronLookup)

	// walk through neurons in order: input, output, hidden and create bias and connections lists
	biases := make([]float64, totalNeuronCount)
	connections := make([]*FastNetworkLink, 0)

	if inConnects, err := processIncomingConnections(inList, biases, neuronLookup); err == nil {
		connections = append(connections, inConnects...)
	} else {
		return nil, err
	}
	if inConnects, err := processIncomingConnections(hiddenList, biases, neuronLookup); err == nil {
		connections = append(connections, inConnects...)
	} else {
		return nil, err
	}
	if inConnects, err := processIncomingConnections(n.Outputs, biases, neuronLookup); err == nil {
		connections = append(connections, inConnects...)
	} else {
		return nil, err
	}

	// walk through control neurons
	modules := make([]*FastControlNode, len(n.controlNodes))
	for i, cn := range n.controlNodes {
		// collect inputs
		inputs := make([]int, len(cn.Incoming))
		for j, in := range cn.Incoming {
			if inIndex, ok := neuronLookup[in.InNode.Id]; ok {
				inputs[j] = inIndex
			} else {
				return nil, fmt.Errorf("failed to lookup for input neuron with id: %d at control neuron: %d",
					in.InNode.Id, cn.Id)
			}
		}
		// collect outputs
		outputs := make([]int, len(cn.Outgoing))
		for j, out := range cn.Outgoing {
			if outIndex, ok := neuronLookup[out.OutNode.Id]; ok {
				outputs[j] = outIndex
			} else {
				return nil, fmt.Errorf("failed to lookup for output neuron with id: %d at control neuron: %d",
					out.InNode.Id, cn.Id)
			}
		}
		// build control node
		modules[i] = &FastControlNode{InputIndexes: inputs, OutputIndexes: outputs, ActivationType: cn.ActivationType}
	}

	return NewFastModularNetworkSolver(biasNeuronCount, inputNeuronCount, outputNeuronCount, totalNeuronCount,
		activations, connections, biases, modules), nil
}

func processList(startIndex int, nList []*NNode, activations []math.NodeActivationType, neuronLookup map[int]int) int {
	for _, ne := range nList {
		activations[startIndex] = ne.ActivationType
		neuronLookup[ne.Id] = startIndex
		startIndex += 1
	}
	return startIndex
}

func processIncomingConnections(nList []*NNode, biases []float64, neuronLookup map[int]int) (connections []*FastNetworkLink, err error) {
	connections = make([]*FastNetworkLink, 0)
	for _, ne := range nList {
		if targetIndex, ok := neuronLookup[ne.Id]; ok {
			for _, in := range ne.Incoming {
				if sourceIndex, ok := neuronLookup[in.InNode.Id]; ok {
					if in.InNode.NeuronType == BiasNeuron {
						// store bias for target neuron
						biases[targetIndex] += in.Weight
					} else {
						// save connection
						conn := FastNetworkLink{
							SourceIndex: sourceIndex,
							TargetIndex: targetIndex,
							Weight:      in.Weight,
						}
						connections = append(connections, &conn)
					}
				} else {
					err = fmt.Errorf("failed to lookup for source neuron with id: %d", in.InNode.Id)
					break
				}
			}
		} else {
			err = fmt.Errorf("failed to lookup for target neuron with id: %d", ne.Id)
			break
		}
	}
	if err != nil {
		return nil, err
	}
	return connections, err
}

func (n *Network) Flush() (res bool, err error) {
	res = true
	// Flush back recursively
	for _, node := range n.allNodes {
		node.Flushback()
		err = node.FlushbackCheck()
		if err != nil {
			// failed - no need to continue
			res = false
			break
		}
	}
	return res, err
}

// PrintActivation Prints the values of network outputs to the console
func (n *Network) PrintActivation() string {
	out := bytes.NewBufferString(fmt.Sprintf("Network %s with id %d outputs: (", n.Name, n.Id))
	for i, node := range n.Outputs {
		_, _ = fmt.Fprintf(out, "[Output #%d: %s] ", i, node)
	}
	_, _ = fmt.Fprint(out, ")")
	return out.String()
}

// PrintInput Print the values of network inputs to the console
func (n *Network) PrintInput() string {
	out := bytes.NewBufferString(fmt.Sprintf("Network %s with id %d inputs: (", n.Name, n.Id))
	for i, node := range n.inputs {
		_, _ = fmt.Fprintf(out, "[Input #%d: %s] ", i, node)
	}
	_, _ = fmt.Fprint(out, ")")
	return out.String()
}

// OutputIsOff If at least one output is not active then return true
func (n *Network) OutputIsOff() bool {
	for _, node := range n.Outputs {
		if node.ActivationsCount == 0 {
			return true
		}

	}
	return false
}

// ActivateSteps Attempts to activate the network given number of steps before returning error.
func (n *Network) ActivateSteps(maxSteps int) (bool, error) {
	// For adding to the active sum
	addAmount := 0.0
	// Make sure we at least activate once
	oneTime := false
	// Used in case the output is somehow truncated from the network
	abortCount := 0

	// Keep activating until all the outputs have become active
	// (This only happens on the first activation, because after that they are always active)
	for n.OutputIsOff() || !oneTime {

		if abortCount >= maxSteps {
			return false, NetErrExceededMaxActivationAttempts
		}

		// For each neuron node, compute the sum of its incoming activation
		for _, np := range n.allNodes {
			if np.IsNeuron() {
				np.ActivationSum = 0.0 // reset activation value

				// For each node's incoming connection, add the activity from the connection to the activesum
				for _, link := range np.Incoming {
					// Handle possible time delays
					if !link.IsTimeDelayed {
						addAmount = link.Weight * link.InNode.GetActiveOut()
						if link.InNode.isActive || link.InNode.IsSensor() {
							np.isActive = true
						}
					} else {
						addAmount = link.Weight * link.InNode.GetActiveOutTd()
					}
					np.ActivationSum += addAmount
				} // End {for} over incoming links
			} // End if != SENSOR
		} // End {for} over all nodes

		// Now activate all the neuron nodes off their incoming activation
		for _, np := range n.allNodes {
			if np.IsNeuron() {
				// Only activate if some active input came in
				if np.isActive {
					// Now run the net activation through an activation function
					err := ActivateNode(np, math.NodeActivators)
					if err != nil {
						return false, err
					}
				}
			}
		}

		// Now activate all MIMO control genes to propagate activation through genome modules
		for _, cn := range n.controlNodes {
			cn.isActive = false
			// Activate control MIMO node as control module
			err := ActivateModule(cn, math.NodeActivators)
			if err != nil {
				return false, err
			}
			// mark control node as active
			cn.isActive = true
		}

		oneTime = true
		abortCount += 1
	}
	return true, nil
}

// Activate Activates the net such that all outputs are active
func (n *Network) Activate() (bool, error) {
	return n.ActivateSteps(20)
}

func (n *Network) ForwardSteps(steps int) (res bool, err error) {
	for i := 0; i < steps; i++ {
		res, err = n.Activate()
		if err != nil {
			// failure - no need to continue
			break
		}
	}
	return res, err
}

func (n *Network) RecursiveSteps() (bool, error) {
	return false, errors.New("RecursiveSteps is not implemented")
}

func (n *Network) Relax(_ int, _ float64) (bool, error) {
	return false, errors.New("relax is not implemented")
}

func (n *Network) LoadSensors(sensors []float64) error {
	counter := 0
	if len(sensors) == len(n.inputs) {
		// BIAS value provided as input
		for _, node := range n.inputs {
			if node.IsSensor() {
				node.SensorLoad(sensors[counter])
				counter += 1
			}
		}
	} else {
		// use default BIAS value
		for _, node := range n.inputs {
			if node.NeuronType == InputNeuron {
				node.SensorLoad(sensors[counter])
				counter += 1
			} else {
				node.SensorLoad(1.0) // default BIAS value
			}
		}
	}

	return nil
}

func (n *Network) ReadOutputs() []float64 {
	outs := make([]float64, len(n.Outputs))
	for i, o := range n.Outputs {
		outs[i] = o.Activation
	}
	return outs
}

func (n *Network) NodeCount() int {
	if len(n.controlNodes) == 0 {
		return len(n.allNodes)
	} else {
		return len(n.allNodes) + len(n.controlNodes)
	}
}

func (n *Network) LinkCount() int {
	n.numLinks = 0
	for _, node := range n.allNodes {
		n.numLinks += len(node.Incoming)
	}
	if len(n.controlNodes) != 0 {
		for _, node := range n.controlNodes {
			n.numLinks += len(node.Incoming)
			n.numLinks += len(node.Outgoing)
		}
	}
	return n.numLinks
}

// Complexity Returns complexity of this network which is sum of nodes count and links count
func (n *Network) Complexity() int {
	return n.NodeCount() + n.LinkCount()
}

// IsRecurrent This checks a POTENTIAL link between a potential in_node
// and potential out_node to see if it must be recurrent.
// Use count and thresh to jump out in the case of an infinite loop.
func (n *Network) IsRecurrent(inNode, outNode *NNode, count *int, thresh int) bool {
	// Count the node as visited
	*count++

	if *count > thresh {
		return false // Short out the whole thing - loop detected
	}

	if inNode == outNode {
		return true
	} else {
		// Check back on all links ...
		for _, link := range inNode.Incoming {
			// But skip links that are already recurrent -
			// We want to check back through the forward flow of signals only
			if !link.IsRecurrent {
				if n.IsRecurrent(link.InNode, outNode, count, thresh) {
					return true
				}
			}
		}
	}
	return false
}

// MaxDepth Find the maximum number of neurons between an output and an input
func (n *Network) MaxDepth() (int, error) {
	if len(n.controlNodes) > 0 {
		return -1, errors.New("unsupported for modular networks")
	}
	// The quick case when there are no hidden nodes
	if len(n.allNodes) == len(n.inputs)+len(n.Outputs) && len(n.controlNodes) == 0 {
		return 1, nil // just one layer depth
	}

	max := 0 // The max depth
	for _, node := range n.Outputs {
		currDepth, err := node.Depth(0)
		if err != nil {
			return currDepth, err
		}
		if currDepth > max {
			max = currDepth
		}
	}

	return max, nil
}

// AllNodes Returns all nodes in the network
func (n *Network) AllNodes() []*NNode {
	return n.allNodes
}
