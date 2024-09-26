import numpy as cp

# CuPy 배열 생성
x = cp.array([1, 2, 3])

# 큰 반복 실행
for i in range(1000000):
    x[-1] = i


print(x[-1])