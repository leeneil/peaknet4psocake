import numpy as np

def array2image(dn, arr):
    #print(arr.shape)
    arr = np.reshape(arr, (arr.shape[0], arr.shape[1], 1))
    arr = np.concatenate( (arr, arr, arr), axis=2)
    arr = arr.transpose(2,0,1)
    c = 3 #arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/25500.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im
