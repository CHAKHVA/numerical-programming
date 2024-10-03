import numpy as np

user_matrix = [[float(input("Enter the number: ")) for _ in range(2)] for _ in range(4)]

matrices = [np.random.rand(4, 2) for _ in range(10)]

def first_norm(matrix1, matrix2):
    matrix = matrix1 - matrix2

    col_length = len(matrix[0])
    column_sums = [0] * col_length

    for row in matrix:
        for col_id in range(col_length):
            column_sums[col_id] += abs(row[col_id])

    return max(column_sums)

def second_norm(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2, ord=2)

def frobenius_norm(matrix1, matrix2):
    sum_of_squares = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            sum_of_squares += (matrix1[i][j] - matrix2[i][j]) ** 2

    return np.sqrt(sum_of_squares)

min_distance = float("inf")
max_distance = float("-inf")
for matrix in matrices:
    distance = frobenius_norm(matrix, user_matrix)
    if distance < min_distance:
        min_distance = distance
    if distance > max_distance:
        max_distance = distance

print(min_distance, max_distance)

'''
Frobenius norm measures the Euclidean distance between the two matrices, so its useful for general purpose distance calculations.
'''

