import sys
import numpy as np
import copy
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
import time
import os

# Set parameters for the problem that we'd like to solve
networkFile = os.path.abspath( sys.argv[1] )
property = int(sys.argv[2])
dims_to_optimize = [0] # Dimensions that we constraint to be less than epsilon fromo their centroid.

# TODO: FIGURE OUT HOW TO TURN ON AND OFF SBT
network = Marabou.read_nnet(networkFile, normalize=False)

# Hardcode in the different properties
inputVars = network.inputVars.flatten()
num_inputs = len(inputVars)

lower_bounds_one = [0.6, -0.5, -0.5, 0.45, -0.5]
upper_bounds_one = [0.6798577687, 0.5, 0.5, 0.5, -0.45]
centroid_one = list(map(lambda x, y: (x + y) / 2.0, lower_bounds_one, upper_bounds_one))

lower_bounds_two = [0.6, -0.5, -0.5, 0.45, -0.5]
upper_bounds_two = [0.6798577687, 0.5, 0.5, 0.5, -0.45]
centroid_two = list(map(lambda x, y: (x + y) / 2.0, lower_bounds_two, upper_bounds_two))

lower_bounds_three = [-0.3035311561, -0.0095492967, 0.4933803236, 0.3, 0.3]
upper_bounds_three = [-0.2985528119, 0.0095492966, 0.5, 0.5, 0.5]
centroid_three = list(map(lambda x, y: (x + y) / 2.0, lower_bounds_three, upper_bounds_three))

lower_bounds_four = [-0.3035311561, -0.0095492967, 0, 0.3181818182, 0.0833333333]
upper_bounds_four = [-0.2985528119, 0.0095492966, 0.0000000001, 0.5, 0.1666666667]
centroid_four = list(map(lambda x, y: (x + y) / 2.0, lower_bounds_four, upper_bounds_four))


lower_bounds = [lower_bounds_one, lower_bounds_two, lower_bounds_three, lower_bounds_four]
upper_bounds = [upper_bounds_one, upper_bounds_two, upper_bounds_three, upper_bounds_four]
centroids = [centroid_one, centroid_two, centroid_three, centroid_four]

# Print out the output from the centroid
centroidOutput = network.evaluateWithMarabou(centroids[property-1])
print("Centroid output: ", centroidOutput)


# Find the upper bound on the radius
interval_for_dims = [upper_bounds[property-1][i] - lower_bounds[property-1][i] for i in range(len(upper_bounds[property-1])) if i in dims_to_optimize]
max_interval = max(interval_for_dims) # find the max interval of the dimensions you're optimizing

# Input Constraints
# we can only maximize, so introduce a negative radius to maximize
negative_epsilon = network.getNewVariable()
network.setUpperBound(negative_epsilon, 0.0)
network.setLowerBound(negative_epsilon, -max_interval / 2.0)

for i in range(num_inputs):
    # Assign it bounds based on the estimate of the region
    network.setLowerBound(inputVars[i], lower_bounds[property-1][i])
    network.setUpperBound(inputVars[i], upper_bounds[property-1][i])

    # Make the relationship with the radius variable
    if i in dims_to_optimize:
        # x[i] - centroids[i] <= epsilon --> x[i] - epsilon <= centroids[i]
        # x[i] - centroids[i] <= -(negepsilon) --> x[i] + negepsilon <= centroids[i]
        upperBoundEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
        upperBoundEquation.addAddend(1.0, inputVars[i])
        upperBoundEquation.addAddend(1.0, negative_epsilon)
        upperBoundEquation.setScalar(centroids[property-1][i])
        network.addEquation(upperBoundEquation)

        # centroids[i] - x[i] <= epsilon --> -x[i] - epsilon <= -centroids[i]
        # centroids[i] - x[i] <= -(negepsilon) --> -x[i] + negepsilon <= -centroids[i]
        lowerBoundEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
        lowerBoundEquation.addAddend(-1.0, inputVars[i])
        lowerBoundEquation.addAddend(1.0, negative_epsilon)
        lowerBoundEquation.setScalar(-centroids[property-1][i])
        network.addEquation(lowerBoundEquation)

outputVars = network.outputVars.flatten()

if (property == 1):
    # y0 >= 3.9911256459
    outputConstraintEquation = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
    outputConstraintEquation.addAddend(1.0, outputVars[0])
    outputConstraintEquation.setScalar(3.9911256459)
    network.addEquation(outputConstraintEquation)
elif (property == 2):
    # +y0 -y1 >= 0
    # +y0 -y2 >= 0
    # +y0 -y3 >= 0
    # +y0 -y4 >= 0
    outputConstraintEquation_one = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    outputConstraintEquation_one.addAddend(1.0, outputVars[0])
    outputConstraintEquation_one.addAddend(-1.0, outputVars[1])
    outputConstraintEquation_one.setScalar(0.0)
    network.addEquation(outputConstraintEquation_one)


    outputConstraintEquation_two = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    outputConstraintEquation_two.addAddend(1.0, outputVars[0])
    outputConstraintEquation_two.addAddend(-1.0, outputVars[2])
    outputConstraintEquation_two.setScalar(0.0)
    network.addEquation(outputConstraintEquation_two)

    outputConstraintEquation_three = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    outputConstraintEquation_three.addAddend(1.0, outputVars[0])
    outputConstraintEquation_three.addAddend(-1.0, outputVars[3])
    outputConstraintEquation_three.setScalar(0.0)
    network.addEquation(outputConstraintEquation_three)

    outputConstraintEquation_four = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    outputConstraintEquation_four.addAddend(1.0, outputVars[0])
    outputConstraintEquation_four.addAddend(-1.0, outputVars[4])
    outputConstraintEquation_four.setScalar(0.0)
    network.addEquation(outputConstraintEquation_four)

elif (property == 3 or property == 4):
    # +y0 -y1 <= 0
    # +y0 -y2 <= 0
    # +y0 -y3 <= 0
    # +y0 -y4 <= 0
    outputConstraintEquation_one = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
    outputConstraintEquation_one.addAddend(1.0, outputVars[0])
    outputConstraintEquation_one.addAddend(-1.0, outputVars[1])
    outputConstraintEquation_one.setScalar(0.0)
    network.addEquation(outputConstraintEquation_one)


    outputConstraintEquation_two = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
    outputConstraintEquation_two.addAddend(1.0, outputVars[0])
    outputConstraintEquation_two.addAddend(-1.0, outputVars[2])
    outputConstraintEquation_two.setScalar(0.0)
    network.addEquation(outputConstraintEquation_two)

    outputConstraintEquation_three = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
    outputConstraintEquation_three.addAddend(1.0, outputVars[0])
    outputConstraintEquation_three.addAddend(-1.0, outputVars[3])
    outputConstraintEquation_three.setScalar(0.0)
    network.addEquation(outputConstraintEquation_three)

    outputConstraintEquation_four = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
    outputConstraintEquation_four.addAddend(1.0, outputVars[0])
    outputConstraintEquation_four.addAddend(-1.0, outputVars[4])
    outputConstraintEquation_four.setScalar(0.0)
    network.addEquation(outputConstraintEquation_four)
else:
    assert False, "Unsupported property"

# # Objective is a tuple with a list of variable indices
# # and a list of coefficient indices
# for i in range(len(objectives[property-1][0])):
#     out_index = objectives[property-1][0][i]
#     out_coefficient = objectives[property-1][1][i]
#     optEquation.addAddend(out_coefficient, outputVars[out_index])
#     print("Adding: ", out_coefficient, " to output var: ", out_index)
#     print("Variable: ", outputVars[out_index])

print("Property: ", property)

network.setOptimizationVariable(negative_epsilon) # maximize negative_epsilon --> minimize radius epsilon

# Set the options
options = MarabouCore.Options()
options._optimize = True
options._verbosity = 1
options._timeoutInSeconds = 1200
options._dnc = False
options._divideStrategy = MarabouCore.DivideStrategy.EarliestReLU

# Run the solver
start_time = time.time()
vals, state = network.solve(filename="", options=options)
end_time = time.time()

print("Took: ", end_time - start_time)
print("Upper bound for epsilon: ", max_interval / 2.0)
print("Property: ", property)
print("Network: ", networkFile)
print("Dims to optimize: ", dims_to_optimize)

if (not state.hasTimedOut()):
    # Compute the optimum output from the optimum input
    inputList = [vals[i] for i in range(0, num_inputs)]
    marabouOptimizerResult = network.evaluateWithMarabou([inputList])[0]
    deltas = [inputList[i] - centroids[property-1][i] for i in range(len(inputList))]
    print("Deltas from nominal: ", [inputList[i] - centroids[property-1][i] for i in range(len(inputList))])
    print("Optimal output: ", marabouOptimizerResult)
    print("Optimal Objective: ", min([deltas[i] for i in range(len(deltas)) if i in dims_to_optimize]))
else:
    print("Timed out")
