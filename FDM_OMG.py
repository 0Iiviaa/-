import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------
# 问题参数（可调整）
# ----------------------------
a = 0.05      # 宽度 (m) 对应题目 a=50mm
b = 0.06      # 高度 (m) 对应题目 b=60mm
V_top = 200.0
V_bottom = -100.0

Nx = 61       # x 网格点数（可改）
Ny = 61       # y 网格点数（可改）

dx = a / (Nx - 1)
dy = b / (Ny - 1)

tol = 1e-6    # 收敛阈值
max_iter = 20000

# 要测试的超松弛因子列表（参考你给的图）
omega_list = [1.0, 1.4, 1.6, 1.8, 1.9, 1.96, 1.98]

# ----------------------------
# 设置初始与边界的函数
# ----------------------------
def init_V(Nx, Ny):
    V = np.zeros((Ny, Nx))
    # 上下边界
    V[-1, :] = V_top
    V[0, :]  = V_bottom
    # 左右边界用 y 方向线性插值封闭
    dy = b / (Ny - 1)
    for j in range(Ny):
        y = j * dy
        val = V_bottom + (V_top - V_bottom) * (y / b)
        V[j, 0] = V[j, -1] = val
    return V

# ----------------------------
# SOR 求解器：返回迭代后的 V 和迭代次数 it
# ----------------------------
def solve_sor(V, omega, tol=1e-6, max_iter=20000):
    Ny, Nx = V.shape
    it = 0
    for it in range(1, max_iter+1):
        max_diff = 0.0
        # 内点迭代（不更新边界）
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                k = 1/dx**2 + 1/dy**2
                V_new = (V[j+1, i]/(2*dx**2) + V[j-1, i]/(2*dx**2) + V[j, i+1]/(2*dy**2) + V[j, i-1]/(2*dy**2))/k
                diff = V_new - V[j, i]
                V[j, i] += omega * diff
                if abs(diff) > max_diff:
                    max_diff = abs(diff)
        if max_diff < tol:
            return V, it
    # 若达到 max_iter 仍未收敛，返回 max_iter
    return V, max_iter

# ----------------------------
# 对不同 omega 执行测试
# ----------------------------
iters = []
for om in omega_list:
    V0 = init_V(Nx, Ny)
    V_final, it_needed = solve_sor(V0, om, tol=tol, max_iter=max_iter)
    iters.append(it_needed)
    print(f"omega={om:.3f}  ->  iterations = {it_needed}")

# ----------------------------
# 绘图：omega vs iterations
# ----------------------------
plt.figure(figsize=(6,4))
plt.plot(omega_list, iters, '-o', markerfacecolor='white')
plt.xlabel('超松弛因子 ω')
plt.ylabel('收敛所需迭代次数')
plt.title('超松弛因子 ω 与迭代次数关系 (251853王映雪)')
plt.grid(True)

# 标出最小点
min_idx = int(np.argmin(iters))
plt.scatter([omega_list[min_idx]], [iters[min_idx]], color='red', zorder=5)
plt.text(omega_list[min_idx], iters[min_idx],
         f'  最优 ω={omega_list[min_idx]:.2f}\n  it={iters[min_idx]}', color='red', va='bottom')

plt.show()