import numpy as np
import pandas as pd

def Fibonachi(n):
    nums = [n]
    r = 0

    for i in nums:
        nums.append(nums[r-1]+ nums[r])
        r += 1
        print(nums)
        if len(nums) > 9:
            print("Stopped")
            return 0

Fibonachi(4)