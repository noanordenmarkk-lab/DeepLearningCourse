import numpy as np
import pandas as pd

lam = 10.0


def poisson_counter(lam, target):
    counter = 0
    currvall = 1.0

    target = target * np.exp(-lam)
    
    while currvall > target:
        currvall = currvall*np.random.uniform(0,1)
        counter += 1
        if currvall < target:
            print("Counter:", counter)
            break


poisson_counter(lam, 0.5)



