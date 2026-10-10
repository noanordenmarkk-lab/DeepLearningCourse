import numpy as np
import pandas as pd

text = "Hi my name is Noa"

for i in text:
    if i in 'aeiuo'.lower():
        continue
    print(i)
