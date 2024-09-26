import cupy as cp

# CuPy 배열 생성
x = cp.array([1, 2, 3])

# 연산 전 GPU 동기화
cp.cuda.Device(0).synchronize()

# 큰 반복 실행
for i in range(1000000):
    x[-1] = i

# 연산 후 GPU 동기화
cp.cuda.Device(0).synchronize()

print(x[-1])