import heterocl as hcl
import numpy as np

A = hcl.placeholder((10, 10), "A")
def kernel(A):
    return hcl.compute((8, 8), lambda y, x: A[y][x] + A[y+2][x+2], "B")
s = hcl.create_scheme(A, kernel)
s.downsize(kernel.B, hcl.UInt(4))
s = hcl.create_schedule_from_scheme(s)
s.partition(A)
s[kernel.B].pipeline(kernel.B.axis[1])

f = hcl.build(s, target="vhls")
print(f)
