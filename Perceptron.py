from Utility import Similarity_innerProd
from Utility import sumOfTwoVectors
from Utility import minusOfTwoVectors

def Perceptron (priorWeights, inputFeatures, inputClasses , testFeatures) :
    # priorWeights is a list of lists
    # inputfeatures is a list of lists
    # classes is a list
    # This function returns the class of the test features

    counter = 0
    for inp in inputFeatures : # For example an image
        img = inp
        maxPred = -1
        innerCounter = 0
        number = -1
        for wei in priorWeights :
            pred = Similarity_innerProd(img , wei)
            if pred > maxPred :
                number = innerCounter

            innerCounter += 1

        # if not correct, should update weights
        if number != inputClasses[counter] :

            # Minus wrong class
            priorWeights[number] = minusOfTwoVectors(priorWeights[number] , img)

            # Plus right class
            priorWeights[inputClasses[counter]] = sumOfTwoVectors(priorWeights[inputClasses[counter]] , img)

        counter += 1

    outputClasses = []
    for test in testFeatures :
        img = test
        maxPred = -1
        innerCounter = 0
        number = -1
        for wei in priorWeights :
            pred = Similarity_innerProd(img , wei)
            if pred > maxPred :
                number = innerCounter

            innerCounter += 1
        outputClasses.append(number)
