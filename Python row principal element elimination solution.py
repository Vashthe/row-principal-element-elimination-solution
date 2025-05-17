import numpy as np


def row_principal_element_elimination(A, b):
    rows = cols = len(A)
    augm_mat = np.column_stack((A, b.T)) if b is not None else A.copy()

    for col in range(cols):
        # 在子矩阵中寻找最大主元
        sub_mat = augm_mat[col:rows, col]
        print(f"子矩阵为{sub_mat}")
        print("-" * 50)

        # 找到最大元素的全局位置
        flat_index = np.argmax(np.abs(sub_mat))  # 使用绝对值找主元
        max_sub_row= np.unravel_index(flat_index, sub_mat.shape)[0]
        max_global_row = col + max_sub_row

        # 行交换
        if max_global_row != col:
            augm_mat[[col, max_global_row]] = augm_mat[[max_global_row, col]]

        # 检查主元是否为0
        if augm_mat[col, col] == 0:
            raise ValueError("矩阵是奇异的，无法求解")

        # 归一化当前行
        pivot = augm_mat[col, col]
        augm_mat[col] = augm_mat[col] / pivot

        # 消去下方行
        for row in range(col + 1, rows):
            factor = augm_mat[row, col] / augm_mat[col, col]
            augm_mat[row] = augm_mat[row] - factor * augm_mat[col]

    print("前向消元后的矩阵:")
    print(augm_mat)
    print("-" * 50)

    # 回代
    solution = np.zeros(cols)
    for i in range(cols - 1, -1, -1):
        solution[i] = augm_mat[i, cols]  # 最后一列是 b
        for j in range(i + 1, cols):
            solution[i] -= augm_mat[i, j] * solution[j]

    print("解向量:")
    print(solution)

    return solution


# 测试
A = np.array([[2, -1, 3], [4, 2, 5], [1, 2, 0]], dtype=float)
b = np.array([[1, 4, 7]], dtype=float)
row_principal_element_elimination(A, b)