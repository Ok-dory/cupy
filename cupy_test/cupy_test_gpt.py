import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# 벡터 정규화 함수
def normalize(vector):
    return vector / cp.linalg.norm(vector)

# 반사 벡터 계산 함수
def reflected(vector, axis):
    return vector - 2 * cp.dot(vector, axis) * axis

# 구와의 교차점 계산 함수
def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * cp.dot(ray_direction, ray_origin - center)
    c = cp.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + cp.sqrt(delta)) / 2
        t2 = (-b - cp.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

# 가장 가까운 교차점을 찾는 함수
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = cp.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

# 레이 트레이싱 함수
def ray_tracing(x, y):
    # CuPy 배열로 pixel을 생성
    pixel = cp.array([x, y, 0], dtype=cp.float32)
    origin = camera 
    direction = normalize(pixel - origin) 
    color = cp.zeros((3), dtype=cp.float32)  # CuPy 배열로 초기화
    reflection = 1 
    for k in range(max_depth): 
        nearest_object, min_distance = nearest_intersected_object(objects, origin, direction) 
        if nearest_object is None: 
            break 
        intersection = origin + min_distance * direction 
        normal_to_surface = normalize(intersection - nearest_object['center']) 
        shifted_point = intersection + 1e-5 * normal_to_surface 
        intersection_to_light = normalize(light['position'] - shifted_point) 
        _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light) 
        intersection_to_light_distance = cp.linalg.norm(light['position'] - intersection) 
        is_shadowed = min_distance < intersection_to_light_distance 
        if is_shadowed: 
            break 
        illumination = cp.zeros((3), dtype=cp.float32)  # CuPy 배열로 초기화
        # Ambient 조명
        illumination += nearest_object['ambient'] * light['ambient'] 
        # Diffuse 조명
        illumination += nearest_object['diffuse'] * light['diffuse'] * cp.dot(intersection_to_light, normal_to_surface) 
        # Specular 조명
        intersection_to_camera = normalize(camera - intersection) 
        H = normalize(intersection_to_light + intersection_to_camera) 
        illumination += nearest_object['specular'] * light['specular'] * cp.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4) 
        # Reflection 조명
        color += reflection * illumination 
        reflection *= nearest_object['reflection'] 
        origin = shifted_point 
        direction = reflected(direction, normal_to_surface)
    return color

# 메인 함수 시작
max_depth = 3
width = 600
height = 400
camera = cp.array([0, 0, 2], dtype=cp.float32)  # 카메라 위치 CuPy 배열로 설정
light = {
    'position': cp.array([5, 5, 5], dtype=cp.float32), 
    'ambient': cp.array([1, 1, 1], dtype=cp.float32), 
    'diffuse': cp.array([1, 1, 1], dtype=cp.float32), 
    'specular': cp.array([1, 1, 1], dtype=cp.float32)
}
objects = [
    {'center': cp.array([-0.2, 0, -1], dtype=cp.float32), 'radius': 0.2, 'ambient': cp.array([0.1, 0, 0], dtype=cp.float32), 'diffuse': cp.array([0.7, 1, 0], dtype=cp.float32), 'specular': cp.array([1, 1, 1], dtype=cp.float32), 'shininess': 80, 'reflection': 0.1},
    {'center': cp.array([0.1, -0.3, 0], dtype=cp.float32), 'radius': 0.1, 'ambient': cp.array([0.1, 0, 0.1], dtype=cp.float32), 'diffuse': cp.array([0.7, 0, 0.7], dtype=cp.float32), 'specular': cp.array([1, 1, 1], dtype=cp.float32), 'shininess': 100, 'reflection': 0.5},
    {'center': cp.array([0.5, 0, -1], dtype=cp.float32), 'radius': 0.5, 'ambient': cp.array([0.1, 0, 0.1], dtype=cp.float32), 'diffuse': cp.array([0.7, 0.7, 0.7], dtype=cp.float32), 'specular': cp.array([1, 1, 1], dtype=cp.float32), 'shininess': 100, 'reflection': 0.5},
    {'center': cp.array([-0.3, 0, 0], dtype=cp.float32), 'radius': 0.15, 'ambient': cp.array([0, 0.1, 0], dtype=cp.float32), 'diffuse': cp.array([0, 0.6, 0], dtype=cp.float32), 'specular': cp.array([1, 1, 1], dtype=cp.float32), 'shininess': 100, 'reflection': 0.5},
    {'center': cp.array([0, -9000, 0], dtype=cp.float32), 'radius': 9000 - 0.7, 'ambient': cp.array([0.1, 0.1, 0.1], dtype=cp.float32), 'diffuse': cp.array([0.6, 0.6, 0.6], dtype=cp.float32), 'specular': cp.array([1, 1, 1], dtype=cp.float32), 'shininess': 100, 'reflection': 0.5}
]

# 화면 비율 계산 및 렌더링 영역 설정
ratio = float(width) / height 
screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

# CuPy 배열로 이미지 초기화
image = cp.zeros((height, width, 3), dtype=cp.float32)
Y = cp.linspace(screen[1], screen[3], height)
X = cp.linspace(screen[0], screen[2], width)
for i, y in enumerate(Y):
    for j, x in enumerate(X):
        color = ray_tracing(x, y)  # CuPy 연산
        image[i, j] = cp.clip(color, 0, 1)  # CuPy로 clip 연산

# CuPy 배열을 NumPy로 변환 후 이미지 저장
plt.imsave('image4.png', image.get())  # .get()을 호출해 NumPy 배열로 변환 후 저장