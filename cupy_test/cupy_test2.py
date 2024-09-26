import cupy as cp

# 더 큰 배열 생성
x = cp.ones((10000, 10000), dtype=cp.float32)

cp.cuda.Device(0).synchronize()

for i in range(1000):
    x += i

cp.cuda.Device(0).synchronize()
print(x[-1, -1])