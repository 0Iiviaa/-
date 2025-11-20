import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# -------------------- 可调参数 --------------------
a = 0.05         # 50 mm, x 方向板长 (m)
b = 0.08         # 80 mm, y 方向板间距 (m)
V_top = 200.0    # 上板电位 (V)
V_bot = -100.0   # 下板电位 (V)

# 网格设置 (越密电场越精确，但耗时越长)
Nx = 121
Ny = 161

# SOR 参数
omega = 1.85
tol = 1e-6
max_iter = 20000

# 电子及物理常数
qe = -1.602176634e-19   # 电子电荷（符号）
qe_abs = 1.602176634e-19
me = 9.10938356e-31     # 电子质量

# 要计算的加速电压 (示例三种)
V_acc_list = [80, 200, 1000]   # 单位：V

# RK4 积分设置
dt = 1e-11             # 时间步长 (s) —— 可调整
max_steps = 200000     # 最大步数


# -------------------- 有限差分 + SOR 求电势 --------------------
dx = a / (Nx - 1)
dy = b / (Ny - 1)
x_grid = np.linspace(0, a, Nx)
y_grid = np.linspace(0, b, Ny)

def solve_potential():
    V = np.zeros((Ny, Nx))
    # 上下边界
    V[-1, :] = V_top
    V[0, :]  = V_bot
    # 左右边界：沿 y 线性插值封闭
    for j in range(Ny):
        y = j * dy
        val = V_bot + (V_top - V_bot) * (y / b)
        V[j, 0]  = val
        V[j, -1] = val

    for it in range(max_iter):
        max_diff = 0.0
        # 内点更新（五点差分）
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                k = 1/dx**2 + 1/dy**2
                V_new = (V[j+1, i]/(2*dx**2) + V[j-1, i]/(2*dx**2) + V[j, i+1]/(2*dy**2) + V[j, i-1]/(2*dy**2))/k
                diff = V_new - V[j,i]
                V[j,i] += omega * diff
                if abs(diff) > max_diff:
                    max_diff = abs(diff)
        if max_diff < tol:
            # print(f"Potential converged in {it} iters, max_diff={max_diff:.2e}")
            break
    else:
        print("Potential SOR 未收敛，达到 max_iter。")
    return V

V_field = solve_potential()

# 计算电场 (中心差分)
Ex = np.zeros_like(V_field)
Ey = np.zeros_like(V_field)
# interior: central diff
Ex[:,1:-1] = -(V_field[:,2:] - V_field[:,:-2])/(2*dx)
Ey[1:-1,:] = -(V_field[2:,:] - V_field[:-2,:])/(2*dy)
# 边界用向前/向后差分
Ex[:,0] = -(V_field[:,1] - V_field[:,0])/dx
Ex[:,-1] = -(V_field[:,-1] - V_field[:,-2])/dx
Ey[0,:] = -(V_field[1,:] - V_field[0,:])/dy
Ey[-1,:] = -(V_field[-1,:] - V_field[-2,:])/dy


# -------------------- 双线性插值函数（在任意位置取 E） --------------------
def bilinear_interp(field, x, y):
    """
    field: 2D array shape (Ny, Nx) defined on x_grid,y_grid
    x,y: scalars within [0,a] and [0,b]
    returns interpolated scalar value
    """
    # clamp to domain
    if x <= 0: i = 0; tx = 0.0
    elif x >= a: i = Nx - 2; tx = 1.0
    else:
        rx = x / dx
        i = int(np.floor(rx))
        tx = (x - x_grid[i]) / dx

    if y <= 0: j = 0; ty = 0.0
    elif y >= b: j = Ny - 2; ty = 1.0
    else:
        ry = y / dy
        j = int(np.floor(ry))
        ty = (y - y_grid[j]) / dy

    # field indexed as field[j,i] (row = y index)
    f00 = field[j  , i  ]
    f10 = field[j  , i+1]
    f01 = field[j+1, i  ]
    f11 = field[j+1, i+1]
    fxy = (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11
    return fxy

def E_at(pos):
    # pos = (x,y)
    x, y = pos
    Ex_loc = bilinear_interp(Ex, x, y)
    Ey_loc = bilinear_interp(Ey, x, y)
    return Ex_loc, Ey_loc


# -------------------- RK4 推进单步 --------------------
def rk4_step(state, dt):
    # state = [x, y, vx, vy]
    def deriv(s):
        x, y, vx, vy = s
        Ex_loc, Ey_loc = E_at((x,y))
        ax = (qe / me) * Ex_loc
        ay = (qe / me) * Ey_loc
        return np.array([vx, vy, ax, ay])

    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return new_state


# -------------------- 主循环：对不同加速电压做轨迹计算 --------------------
results = []
plt.figure(figsize=(7,5))
colors = ['r','g','b']
for idx, Vacc in enumerate(V_acc_list):
    # 初速度由加速电压给出 (能量 = e*V_acc)
    v0 = np.sqrt(2.0 * qe_abs * Vacc / me)   # m/s, 仅大小
    # 初始状态：放在 x=0, y=b/2，沿 +x 方向进入
    x0 = 0.0 + 1e-6   # 给一点小位移以免刚好在边界
    y0 = b / 2.0
    vx0 = v0
    vy0 = 0.0
    state = np.array([x0, y0, vx0, vy0])

    traj_x = [x0]; traj_y = [y0]
    hit = False
    step = 0

    while step < max_steps:
        # 记录
        x, y, vx, vy = state
        # 判断是否出界（碰到上/下偏转板）
        if y <= 0 or y >= b:
            hit = True
            break
        # 判断是否穿出右端（离开偏转区）
        if x >= a:
            break

        # RK4 一步
        state = rk4_step(state, dt)
        traj_x.append(state[0]); traj_y.append(state[1])
        step += 1

    results.append({
        'Vacc': Vacc,
        'traj_x': np.array(traj_x),
        'traj_y': np.array(traj_y),
        'steps': step,
        'hit': hit
    })

    label = f"{Vacc} V"
    plt.plot(traj_x, traj_y, '-', color=colors[idx%len(colors)], label=label)

# 画出板轮廓
plt.plot([0,a],[0,0],'k-',linewidth=3)      # 下板
plt.plot([0,a],[b,b],'k-',linewidth=3)      # 上板
plt.xlim(-0.01, a+0.01)
plt.ylim(-0.01, b+0.01)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('电子轨迹（RK4）在偏转板内，三种加速电压 (251853王映雪)')
plt.legend()
plt.gca().set_aspect('equal', 'box')
plt.grid(True)
plt.show()

# 报告每种情况的结果摘要
for r in results:
    print(f"V_acc = {r['Vacc']:.0f} V: steps = {r['steps']}, hit plates? {r['hit']}, final pos = ({r['traj_x'][-1]:.4e}, {r['traj_y'][-1]:.4e})")
