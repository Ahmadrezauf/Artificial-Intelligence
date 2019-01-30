priorWeights = [[7,8,9],[1,2,3] , [1, 2, 3]]

#inputFeatures = [[43, 45, 78], [23, 87, 91], [12, 13, 90], [12, 25, 68], [96, 42, 48], [21, 15, 7], [42, 58, 68]]
#inputFeatures = [[1,2,3],[1,1,1],[2,2,2],[3,2,1],[3,3,3] ,
#                 [89,90,91],[90,90,90],[88,87,93],[91,92,94] ,
#                 [51,52,49],[43,45,46],[54,53,52]]

inputFeatures = [[1,2,3],[1,1,1],[2,2,2],[3,2,1],[3,3,3] ,
                 [-8,9,9],[-9,9,9],[-8,8,9],[-9,9,9] ,
                 [5,5,-4],[4,4,-4],[5,5,-5]]

inputClasses = [0,0,0,0,0 ,
                1,1,1,1,
                2,2,2]

testFeatures = [[1,1,1],[2,2,2],[3,3,3],[-9,9,9]]

from Perceptron import  Perceptron
print("Perceptron------------------")
output = Perceptron(priorWeights , inputFeatures, inputClasses, testFeatures , 1000)
print(output)

from MIRA import MIRA
print("MIRA------------------")
output = MIRA(priorWeights , inputFeatures, inputClasses, testFeatures , 1000)
print(output)

from KernelizedPerceptron import KernelizedPerceptron
print("Kernelized Perceptron------------------")
priorWeights = [[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1]]
output = KernelizedPerceptron(priorWeights , inputFeatures, inputClasses, testFeatures , 1000)
print(output)

import Utility
