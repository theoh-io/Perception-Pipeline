
import numpy as np

a = np.array([10,2,3,4,5,6],dtype=np.int16).reshape((3,2)) * 10
print(a)

np.savetxt("test.txt", a, fmt='%.i',delimiter=' , ')