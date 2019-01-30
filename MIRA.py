from Utility import Similarity_innerProd
from Utility import sumOfTwoVectors
from Utility import minusOfTwoVectors
from Utility import multiplyScalar
import math
def MIRA (priorWeights, inputFeatures, inputClasses , testFeatures, numberOfLearningLoops) :
    # Margin Infused Relaxed Algorithm
    # priorWeights is a list of lists
    # inputfeatures is a list of lists
    # classes is a list
    # This function returns the class of the test features

    for cnt in range(numberOfLearningLoops) :
        counter = 0
        for inp in inputFeatures : # For example an image
            img = inp
            maxPred = -1*math.inf
            innerCounter = 0
            number = -1
            for wei in priorWeights :
                pred = Similarity_innerProd(img , wei)
                if pred > maxPred :
                    maxPred = pred
                    number = innerCounter

                innerCounter += 1

            # if not correct, should update weights
            ## Putting a bound on the step size ===> needed?
            if number != inputClasses[counter] :
                stepSize = 0

                # calculating tau
                tmp = minusOfTwoVectors(priorWeights[number] , priorWeights[inputClasses[counter]])
                tmp = (Similarity_innerProd(tmp , img) + 1)/(2*Similarity_innerProd(img , img))


                # Minus wrong class
                priorWeights[number] = minusOfTwoVectors(priorWeights[number], multiplyScalar(tmp , img))

                # Plus right class
                priorWeights[inputClasses[counter]] = sumOfTwoVectors(priorWeights[inputClasses[counter]], multiplyScalar(tmp , img))

            counter += 1

    outputClasses = []
    for test in testFeatures :
        img = test
        maxPred = -1*math.inf
        innerCounter = 0
        number = -1
        for wei in priorWeights :
            pred = Similarity_innerProd(img , wei)
            if pred > maxPred :
                maxPred = pred
                number = innerCounter

            innerCounter += 1
        outputClasses.append(number)
    return outputClasses