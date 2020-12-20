import sys
import numpy as np
import copy
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
import time

useSbt = True

# Load in the network
#networkFile = "./mnist10x10.nnet"
networkFile = "./mnist10x10.nnet"

network = Marabou.read_nnet(networkFile, useSbt)
inputVars = network.inputVars.flatten()
numInputs = len(inputVars)

inputFile = "./MNISTlabel_0_index_0_.npy"
exampleInput = np.reshape(np.load(inputFile), (1, -1))

# Introduce a variable corresponding to your input radius
radiusVariable = network.getNewVariable()

lower = 0.0
upper = 1.0

label = 0
target = 4


delta_0 = 0.068211 # found by some approximate means
# Input Constraints
for var, val in zip(network.inputVars.flatten(), exampleInput.flatten()):
	# Create upper and lower bound equations based on the radius
	#x_i - x_i_nom <= r --> x_i - r <= x_i_nom
	upperBoundEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
	upperBoundEquation.addAddend(1.0, var)
	upperBoundEquation.addAddend(-1.0, radiusVariable)
	upperBoundEquation.setScalar(val)
	network.addEquation(upperBoundEquation)

	# x_i_nom - x_i <= r --> x_i + r >= x_i_nom
	lowerBoundEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
	lowerBoundEquation.addAddend(1.0, var)
	lowerBoundEquation.addAddend(1.0, radiusVariable)
	lowerBoundEquation.setScalar(val)

	network.addEquation(lowerBoundEquation)
	# Also set an upper and lower bound for each
	network.setLowerBound(var, max(val - delta_0, lower))
	network.setUpperBound(var, min(val + delta_0, upper))

# Set requirements that target be larger than all other values
outputVars = network.outputVars.flatten()
num_outputs = len(outputVars)
for i in range(num_outputs):
	if (i != target):
		curEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
		# cur_output <= target output --> target_output - cur_output >= 0
		curEquation.addAddend(1.0, outputVars[target])
		curEquation.addAddend(-1.0, outputVars[i])
		curEquation.setScalar(0.0)
		network.addEquation(curEquation)

# Enforce the radius is non-negative
posRadiusEqn = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
posRadiusEqn.addAddend(1.0, radiusVariable)
posRadiusEqn.setScalar(0.0)
network.addEquation(posRadiusEqn)

# Enforce the radius is <= 1/2 our total range
radiusRangeEqn = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
radiusRangeEqn.addAddend(1.0, radiusVariable)
radiusRangeEqn.setScalar((upper - lower)/2.0)
network.addEquation(radiusRangeEqn)


# Introduce a negative radius to maximize since we just support max and not min at the moment
negativeRadiusVariable = network.getNewVariable()
negRadiusEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
negRadiusEquation.addAddend(1.0, radiusVariable)
negRadiusEquation.addAddend(1.0, negativeRadiusVariable)
negRadiusEquation.setScalar(0.0)
network.addEquation(negRadiusEquation)

# # maximize the negative radius variable  --> minimize the radius variable
network.setOptimizationVariable(negativeRadiusVariable)

# outputVar = network.outputVars.flatten()[0]

# optEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
# optEquation.addAddend(1.0, outputVar)
# # Introduce your optimization variable
# optVariable = network.getNewVariable()
# # Equation of the form previous_addends - cost_fcn_var = 0 --> previous_addends = cost_fcn_var
# optEquation.addAddend(-1.0, optVariable)
# optEquation.setScalar(0.0)

# # Add the equation and let the network know this is the variable that we'd like to optimize
# network.addEquation(optEquation)
# network.setOptimizationVariable(optVariable)



# Set the options
options = MarabouCore.Options()
options._optimize = True
options._verbosity = 1
options._timeoutInSeconds = 3600
options._dnc = False
options._divideStrategy = MarabouCore.DivideStrategy.EarliestReLU

# Run the solver
print("Right before solve")
vals, state = network.solve(filename="", options=options)

# # Compute the optimum output from the optimum input
inputList = [vals[i] for i in range(0, numInputs)]
marabouOptimizerResult = network.evaluateWithMarabou([inputList])[0]
print("Optimum Output: ", marabouOptimizerResult)
