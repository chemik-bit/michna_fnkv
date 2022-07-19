import numpy as np
test_list = [1,2,3,4,5,6,7,8,9]
splited = np.array_split(test_list, 2)
print(splited)
if len(splited[:-1]) != len(splited[0]):
    splited.pop(-1)
print(splited)