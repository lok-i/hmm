import numpy as np

A = np.array(

[
    [1,0,-1,0,0,0],
    [1,0, 0,1,0,0],
    [0,1, 0,0,1,0],
    [1,1, 0,0,0,1],

]

)

def all_sub(r, c, mat): # returns all sub matrices of order r * c in mat
    arr_of_subs = []
    if (r == len(mat)) and (c == len(mat[0])):
            arr_of_subs.append(mat)
            return arr_of_subs
    for i in range(len(mat) - r + 1):
        for j in range(len(mat[0]) - c + 1):
            temp_mat = []
            for ki in range(i, r + i):
                temp_row = []
                for kj in range(j, c + j):
                    temp_row.append(mat[ki][kj])
                temp_mat.append(temp_row)
            arr_of_subs.append(temp_mat)
    return arr_of_subs


all = all_sub(r=4,c=4,mat=A)
print(len(all))