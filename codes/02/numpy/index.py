import numpy as np

a = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# index
print(a.shape)
print(a[1])
print(a[1,0])
# slice
print(a[:,1])
print(a[:,1:3])
