def Similarity_innerProd (vec1 , vec2) :
    out = 0
    for i in range(len(vec1)) :
        out += (vec1[i] * vec2[i])

    return out


def findAccuracy (prediction, reality) :
    sim = 0

    for i in range(len(prediction)) :
        if prediction == reality :
            sim += 1

    return  sim / len(reality)

def minusOfTwoVectors (vec1 , vec2) :
    out = vec1
    for i in range(len(vec2)) :
        out[i] -= vec2[i]

    return out

def sumOfTwoVectors (vec1 , vec2) :
    out = vec1
    for i in range(len(vec2)):
        out[i] += vec2[i]

    return out

def multiplyScalar (scalar , vec) :
    out = vec
    for i in range(len(vec)) :
        out[i] = vec[i] * scalar

    return out

def Similarity_Kernel_x2 (vec1 , vec2) :
    out = 0
    for i in range(len(vec1)):
        out += ((vec1[i]**2) * (vec2[i]**2))

    return out

