import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------
# 参数区（可自行调整）
# ----------------------------
a = 0.05       # 宽度 50 mm
b = 0.06       # 高度 60 mm
V1 = 200.0     # 上边界电位
V2 = -100.0    # 下边界电位
Nx = 61        # x 方向网格数
Ny = 61        # y 方向网格数
omega = 1.85   # SOR 松弛因子
tol = 1e-6     # 收敛阈值
max_iter = 20000

# ----------------------------
# 构建网格
# ----------------------------
dx = a / (Nx - 1)
dy = b / (Ny - 1)

V = np.zeros((Ny, Nx))

# ----------------------------
# 设置边界条件
# ----------------------------

# 上边界 y = b (Ny-1)
V[-1, :] = V1

# 下边界 y = 0
V[0, :] = V2

# 左右边界做 线性插值：V(x=0,y) = V(x=a,y) = V2 + (V1-V2)*(y/b)
for j in range(Ny):
    y = j * dy
    V[j, 0] = V[j, -1] = V2 + (V1 - V2) * (y / b)

# ----------------------------
# SOR 迭代求解 Laplace 方程
# ----------------------------
def solve_sor(V):
    for it in range(max_iter):
        max_diff = 0
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                k = 1/dx**2 + 1/dy**2
                V_new = (V[j+1, i]/(2*dx**2) + V[j-1, i]/(2*dx**2) + V[j, i+1]/(2*dy**2) + V[j, i-1]/(2*dy**2))/k
                diff = V_new - V[j, i]
                V[j, i] += omega * diff
                max_diff = max(max_diff, abs(diff))

        if max_diff < tol:
            print(f"收敛：迭代 {it} 次，最大误差 = {max_diff:.2e}")
            break
    else:
        print("未收敛")

    return V

V = solve_sor(V)

# ----------------------------
# 绘制等位线分布图
# ----------------------------
x = np.linspace(0, a, Nx)
y = np.linspace(0, b, Ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(6, 4))
cp = plt.contour(X, Y, V, levels=20)
plt.clabel(cp, inline=True)
plt.title("等位线分布图 (251853王映雪)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.gca().set_aspect('equal')
plt.show()

# ----------------------------
# x = a/2 处的电位分布 V(a/2, y)
# ----------------------------
mid = Nx // 2
plt.figure(figsize=(5, 4))
plt.plot(V[:, mid], y)
plt.xlabel("Voltage (V)")
plt.ylabel("y (m)")
plt.title("V(a/2, y) 分布 (251853王映雪)")
plt.grid()
plt.show()
