priorWeights = [[1, 2,3] , [4, 5, 6], [7, 8, 9]]

#inputFeatures = [[43, 45, 78], [23, 87, 91], [12, 13, 90], [12, 25, 68], [96, 42, 48], [21, 15, 7], [42, 58, 68]]
inputFeatures = [[1,1,1],[2,2,2],[3,3,3] , [10,10,10],[11,11,11],[12,12,12] ,[91,91,91],[90,90,90],[89,89,89]]
inputClasses = [0,0,0,1,1,1,2,2,2]

testFeatures = [[1,2,3],[89,90,91],[10,11,12]]

from Perceptron import  Perceptron

output = Perceptron(priorWeights , inputFeatures, inputClasses, testFeatures)
print(output)
