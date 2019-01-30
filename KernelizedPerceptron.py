from Utility import Similarity_Kernel_x2
import math
from Utility import Similarity_innerProd
def KernelizedPerceptron (alphas , inputFeatures, inputClasses, testFeatures, numberOfLearningLoops) :
    # alpha is a list of lists
    # inputfeatures is a list of lists
    # classes is a list
    # This function returns the class of the test features

    for cnt in range(numberOfLearningLoops) :
        counter = 0
        for inp in inputFeatures :
            img = inp
            maxPred = -1*math.inf
            innerCounter = 0
            number = -1
            for wei in alphas:
                # Wei is a vector of alpha_i,y
                tmpAlpha = wei

                sim_jy = 0
                inInnerCounter = 0
                for j in tmpAlpha :
                    sim_jy += (j * Similarity_innerProd(img , inputFeatures[inInnerCounter]))
                    inInnerCounter += 1

                if sim_jy > maxPred:
                    maxPred = sim_jy
                    number = innerCounter

                innerCounter += 1

            # if not correct, should update weights
            ## Putting a bound on the step size ===> needed?
            if number != inputClasses[counter]:
                alphas[number][counter] -= 1
                alphas[counter][counter] += 1

            counter += 1

    outputClasses = []
    print(alphas)
    for test in testFeatures:
        img = test
        maxPred = -1 * math.inf
        number = -1
        innerCounter = 0
        for wei in alphas:
            tmpAlpha = wei

            sim_jy = 0
            inInnerCounter = 0
            for j in tmpAlpha:
                sim_jy += (j * Similarity_innerProd(img, inputFeatures[inInnerCounter]))
                inInnerCounter += 1

            if sim_jy > maxPred:
                maxPred = sim_jy
                number = innerCounter

        innerCounter += 1

        outputClasses.append(number)
    return outputClasses