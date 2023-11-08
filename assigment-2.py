import numpy as np
import random
import matplotlib.pyplot as plt

# Create data
T = 10 # nr of arrays
n = 30000 # size of each array

random.seed(1)

data = []
for i in range(T):
    row = list(range(n))
    random.shuffle(row)
    data.append(row)

def partition(arr, lo, hi, piv_ind):
    p = arr[piv_ind]
    if piv_ind == hi:
        hi -=1
    elif piv_ind == lo:
        lo += 1
    
    i = lo

    for j in range(lo, hi+1): 
          
        if arr[j] <= p: 
            arr[i], arr[j] = arr[j], arr[i] 
            i += 1
    
    # if the pivot is the low one, change i to index with value under p
    if piv_ind < lo:
        i -= 1
              
    arr[i], arr[piv_ind] = arr[piv_ind], arr[i]

    # Calculate scanned elements
    n = hi - lo + 1
    se = n-1

    return i, se

def YBB_partition(arr, lo, hi):
    # p is the left pivot, and q is the right pivot. 
    j = k = lo + 1
    g, p, q = hi - 1, arr[lo], arr[hi] 
    while k <= g: 

        # If elements are less than the left pivot 
        if arr[k] < p: 
            arr[k], arr[j] = arr[j], arr[k] 
            j += 1
            # print(3,arr[lo:hi+1])
        # If elements are greater than or equal  
        # to the right pivot 
        elif arr[k] >= q: 
            while arr[g] > q and k < g: 
                g -= 1
                  
            arr[k], arr[g] = arr[g], arr[k] 
            g -= 1
            # print(4,arr[lo:hi+1])
              
            if arr[k] < p: 
                arr[k], arr[j] = arr[j], arr[k] 
                j += 1
                # print(5,arr[lo:hi+1]) 
                  
        k += 1
          
    j -= 1
    g += 1
    # Bring pivots to their appropriate positions. 
    arr[lo], arr[j] = arr[j], arr[lo] 
    arr[hi], arr[g] = arr[g], arr[hi] 

    # Calculate number of scanned elements
    n = hi - lo + 1
    left_partition_size = j-lo-1
    se = n - 2 + left_partition_size

    # Returning the indices of the pivots 
    return j, g, se

def sesquickselect(arr, k, lo, hi, scanned_elements=0, v=1/4, recursion_depth=0):

    if arr[lo] > arr[hi]: 
        arr[lo], arr[hi] = arr[hi], arr[lo]

    i = k - lo # rank of k in sub-array
    n = hi - lo + 1
    alpha = i/n

    # Since k is rank (starts at 1) but will be caompared to indexes (starts at 0) we subtract one
    k_ind = k-1

    if alpha >= v and alpha <= 1-v:
        lo_p, hi_p, se = YBB_partition(arr, lo, hi)
        scanned_elements += se

        if k_ind == lo_p or k_ind == hi_p:
            return arr[k_ind], scanned_elements, recursion_depth
        elif k_ind < lo_p:
            return sesquickselect(arr, k, lo, lo_p-1, scanned_elements, v, recursion_depth=recursion_depth+1)
        elif k_ind < hi_p:
            return sesquickselect(arr, k, lo_p+1, hi_p-1, scanned_elements, v, recursion_depth=recursion_depth+1)
        else:
            return sesquickselect(arr, k, hi_p+1, hi, scanned_elements, v, recursion_depth=recursion_depth+1)


    else:
        if alpha < v:
            piv_ind = lo
        elif alpha > 1-v:
            piv_ind = hi

        p, se = partition(arr, lo, hi, piv_ind=piv_ind)
        scanned_elements += se

        if k_ind == p: 
            return arr[p], scanned_elements, recursion_depth
        elif k_ind < p:
            return sesquickselect(arr, k, lo, p-1, scanned_elements, recursion_depth=recursion_depth+1)
        else:
            return sesquickselect(arr, k, p+1, hi, scanned_elements, recursion_depth=recursion_depth+1)
    
# # Simple test
# for i in range(3):
#     print(i)
# n = 10
# a = list(range(n))
# random.shuffle(a)
# print(a)
# for k in range(1,n+1):
#     random.shuffle(a)
#     se = 0
#     print(sesquickselect(a, k, 0,len(a)-1, se))


I = [j*100 for j in range(1,301)]
I.insert(0,1)
print(I)
print(I[-1])
x = [i/n for i in I]
print(x)

# print(data[1:])
# print(data.shape)
# print(data[1,:].shape)
S = []

for i in I[:10]:
    S_i = 0
    for t in range(T):
        ith_smallest, S_i_r, recur_depth = sesquickselect(data[t], i, 0, len(data[t])-1)
        S_i += S_i_r
        print(recur_depth)
    print('S_{} = {}'.format(i,S_i))
    S.append(S_i)

plt.plot(x[:10],S)
plt.show()
plt.savefig('theoretical_se_approx.png', format='png')


def theoretical_scanned_elements(x):
    return










