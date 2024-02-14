import numpy as np

# 创建一个2*3的矩阵
matrix1 = np.random.randint(10, size=(2, 3))
print("Matrix1:\n", matrix1)

# 创建一个3*4的矩阵
matrix2 = np.random.randint(10, size=(3, 4))
print("Matrix2:\n", matrix2)

# 创建一个4*1的矩阵
matrix3 = np.random.randint(10, size=(4, 1))
print("Matrix3:\n", matrix3)

result = matrix2 @ matrix3
print("Result:\n", result)
