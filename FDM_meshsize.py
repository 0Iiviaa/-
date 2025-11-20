import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------
# 问题参数
# ----------------------------
a = 0.05
b = 0.06
V1 = 200
V2 = -100
omega = 1.85
tol = 1e-6
max_iter = 20000


# ----------------------------
# Laplace SOR 求解器（给定网格数 n）
# ----------------------------
def solve_laplace(n):
    Nx = n
    Ny = int(n * b / a)  # 保持网格比例

    dx = a / (Nx - 1)
    dy = b / (Ny - 1)

    V = np.zeros((Ny, Nx))

    # 上下边界
    V[-1, :] = V1
    V[0, :] = V2

    # 左右边界线性插值
    for j in range(Ny):
        y = j * dy
        val = V2 + (V1 - V2) * (y / b)
        V[j, 0] = V[j, -1] = val

    # SOR 迭代
    for it in range(max_iter):
        max_diff = 0
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                k = 1/dx**2 + 1/dy**2
                V_new = (V[j+1, i]/(2*dx**2) + V[j-1, i]/(2*dx**2) + V[j, i+1]/(2*dy**2) + V[j, i-1]/(2*dy**2))/k
                diff = V_new - V[j,i]
                V[j,i] += omega * diff
                max_diff = max(max_diff, abs(diff))
        if max_diff < tol:
            break

    return V


# ------------------------------------------------
# 图中逻辑：比较 n1 和 n2=2n1 的误差
# ------------------------------------------------
def compare_error(n1):
    print(f"\n=== 比较 n1={n1} 和 n2={2*n1} 的误差 ===")

    V1 = solve_laplace(n1)
    V2 = solve_laplace(2*n1)

    Ny1, Nx1 = V1.shape
    Ny2, Nx2 = V2.shape

    # 对应的点索引：粗网格 i → 细网格 2i-1（0-based: 2*i）
    err = np.zeros(Ny1)

    for j in range(Ny1):  # 纵向每一层
        j2 = 2*j  # 细网格的对应点（保持 y 对齐）
        if j2 >= Ny2:
            break

        # 最大误差在 x 方向上取绝对值最大
        max_e = 0
        for i in range(Nx1):
            i2 = 2 * i
            if i2 < Nx2:
                diff = abs(V2[j2, i2] - V1[j, i])
                max_e = max(max_e, diff)
        err[j] = max_e

    # 画图
    plt.figure(figsize=(5,4))
    plt.plot(err, marker='o')
    plt.title(f"不同网格大小误差对比 (n={n1} & {2*n1}) (251853王映雪)")
    plt.xlabel("对应 y 方向网格点")
    plt.ylabel("误差")
    plt.grid()
    plt.show()

    return err


# ----------------------------
# 调用，示例：n=10→20，20→40（与图一致）
# ----------------------------
compare_error(10)
compare_error(20)
