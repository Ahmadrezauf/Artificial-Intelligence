from Utility import Similarity_Kernel_x2
def KernelizedPerceptron (alphas , inputFeatures, inputClasses, testFeatures) :
    # alpha is a list of lists
    # inputfeatures is a list of lists
    # classes is a list
    # This function returns the class of the test features

    xi_s = []

    counter = 0
    for inp in inputFeatures :
        img = inp
        maxPred = -1
        innerCounter = 0
        number = -1
        for wei in alphas:
            # Wei is a vector of alpha_i,y
            tmpAlpha = wei

            sim_jy = 0
            inInnerCounter = 0
            for j in tmpAlpha :
                sim_jy += (j * Similarity_Kernel_x2(img , inputFeatures[inInnerCounter]))
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
    for test in testFeatures:
        img = test
        maxPred = -1
        number = -1
        for wei in alphas:
            tmpAlpha = wei

            sim_jy = 0
            inInnerCounter = 0
            for j in tmpAlpha:
                sim_jy += (j * Similarity_Kernel_x2(img, inputFeatures[inInnerCounter]))
                inInnerCounter += 1

            if sim_jy > maxPred:
                maxPred = sim_jy
                number = innerCounter

        innerCounter += 1

        outputClasses.append(number)