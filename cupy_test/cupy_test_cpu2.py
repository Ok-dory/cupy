import numpy as np

# 더 큰 배열 생성
x = np.ones((10000, 10000), dtype=np.float32)

# 큰 반복 실행
for i in range(1000):
    x += i  # 모든 배열의 원소에 i를 더하는 연산

print(x[-1, -1])  # 배열의 마지막 값 출력