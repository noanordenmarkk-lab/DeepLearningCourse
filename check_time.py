import numpy as np
import pandas as pd
import time as t

start1 = t.perf_counter()
r = [ ]

for i in range(10000):
    r.append(i**2)

end1 = t.perf_counter()

print(end1-start1)

start2 = t.perf_counter()

r = [i**2 for i in range(1000)]

end2 = t.perf_counter()

print(end2-start2)

start3 = t.perf_counter()

r3 = np.arange(1000)**2

end3 = t.perf_counter()

print(end3-start3)


