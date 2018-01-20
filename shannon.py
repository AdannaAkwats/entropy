import math

# Shannon entropy H(x) = -sum(P(x)log(P(x)))
def shannon(*probs):
    # Check that probabilities given add up to one and are > 0 and < 1
    checkSum = 0
    for p in list(probs) :
        if(p > 1 or p < 0) : # probabilities are between 0 and 1
            print("Error: Probabilities are not > 1 or < 0")
            return
        checkSum = checkSum + p
    if(checkSum != 1) :
        print("Error: Probabilities do not add to one")
        return

    # If probabilities are valid
    sumProbs = 0
    for p in list(probs) :
        print(p)
        sumProbs = sumProbs + p*math.log(p,2) # 2 represents bits
    return -sumProbs


# Conditional entropy H(X|Y) = -sum( P(x,y) log(P(x,y)/ P(y)))
# n is the number of values of X and m is the number of values of y
# e.g. cond(2,3,0.5,0.5,0.3,0.5,0.2) P(X) = [0.5,0.5] P(Y = [0.3,0.5,0.2]

# Joint entropy
# H(X,Y) = -sum.sum( P(x,y)log(P(x,y)) )
# H(X,Y) >= max[H(x), H(Y)]
# H(X1,...Xn) >= max[H(X1),..., H(Xn)]
# H(X,Y) <= H(X) + H(Y)
# H(X1,...,Xn) <= H(X1) + ... + H(Xn)

# H(X|Y) = H(Y,X) - H(Y)
# Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)


# Testing ...
s = shannon(0.5, 0.2, 0.3)
print(s)
