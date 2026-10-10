def FunFun(x, y):
    if not isinstance(x, (int, float)):
        if y  == "1" or y == "2" or y == "3" or y == "4" or y == "5" or y == "6" or y == "7" or y == "8" or y == "9":
            int(y)
            return x + y
        elif x == "1" or x == "2" or x == "3" or x == "4" or x == "5" or x == "6" or x == "7" or x == "8" or x == "9":
            int(x)
            return x + y
        raise TypeError("x must be a number")

    z = x + y
    return z

print(FunFun("1", 2))  # Output: 3