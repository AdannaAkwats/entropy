import numpy as np
from numpy import linalg as LA
import math
import random

# Read input
import fileinput
import sys

# Function needed
from entropy import vonNeumann
from entropy import generate

# Fill matrix with nunbers from standard input
def fill_matrix_with_input(matrix, n):
    rows = {}
    row_num = 0
    for line in fileinput.input():
        rows[row_num] = line
        row_num = row_num + 1
        if(row_num == n) :
            break
    for i in range(n) :
        each_row = rows[i].split(" ");
        for j in range(n) :
             matrix[i][j] = complex(each_row[j])

# Intro questions
def intro_input() :
    print("Press N : Type new matrix")
    print("Press G : Generate random matrix")
    print("Press Q : Quit")
    print(" ")
    typ = raw_input("[N/G]: ")
    print(" ")
    return typ

# Get dimension of mstrix from input
def get_n_from_input():
    n = input("What is the dimension (n) of the matrix? (eg. 3x3 matrix dimension = 3): ")
    print(" ")
    return n

# Initialise matrix
def init_matrix(n) :
    temp = np.zeros((n,n))
    matrix = np.array(temp, dtype=np.complex128)
    return matrix


# Output...

# Generating matrix to use
print(" ")
while True:
    typ = intro_input()

    # Choices
    if typ in ['N', 'n', 'G', 'g'] :
        n = get_n_from_input()
        matrix = init_matrix(n)

        if typ == 'N'or typ == 'n':
            print "Input the %dx%d matrix, pressing enter after each row and having a space between each element." %(n,n,)
            print(" ")
            fill_matrix_with_input(matrix, n)
        elif typ == 'G'or typ == 'g':
            matrix = generate(n)
        break
    elif typ in ['Q', 'q']:
        sys.exit()
    else :
        print "ERROR!: Input is not N, G or Q"
        print(" ")

print(" ")

# Print original matrix to console
print "Original Matrix: "
print(" ")
print matrix

# For testing, remove warnings TODO
import warnings
warnings.filterwarnings("ignore")

# Get and print Von Neumann value
print(" ")
print "Von Neumann transformation of Matrix: "
print vonNeumann(matrix)

# Pause at python exe
raw_input()
