import numpy as np
import pandas as pd

def dotProduct(x, y):
    r = 0
    nums = [ ]

    if(isinstance(x,np.ndarray) and isinstance(y,np.ndarray)):
        if(len(x) == len(y)):
            for i1 in x:
                r += 1
                nums.append(i1 * y[r-1])
                if(r > (len(y)-1)):
                    return sum(pd.array(nums))


print(dotProduct(np.array([1,2,3]), np.array([4,5,6])))