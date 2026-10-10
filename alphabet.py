import numpy as np
import pandas as pd

alphabet = (['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
for i in range(len(alphabet)):
    if(alphabet[i] == alphabet[0]):
        print(f"{alphabet[i]} is the {i + 1}st letter of the alphabet.")
    if(alphabet[i] == alphabet[1]):
        print(f"{alphabet[i]} is the {i + 1}nd letter of the alphabet.")
    if(alphabet[i] == alphabet[2]):
        print(f"{alphabet[i]} is the {i + 1}rd letter of the alphabet.")
    if(alphabet[i] != alphabet[0] and alphabet[i] != alphabet[1] and alphabet[i] != alphabet[2]):
        print(f"{alphabet[i]} is the {i + 1}th letter of the alphabet.")