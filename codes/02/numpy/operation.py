import numpy as np

a = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b = np.ones(4, dtype=np.int32)

# element-wise operation
print(a+a)
print(a*a)
# matrix operation
print(a+b)
# broadcasting
print(a+10)
print(a*10)
