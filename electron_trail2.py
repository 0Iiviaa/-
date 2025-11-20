import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

# -------------------- 可调参数 --------------------
a = 0.05         # 50 mm, x 方向板长 (m)
b = 0.08         # 80 mm, y 方向板间距 (m)

# 三组板电压（上板，下板）
voltage_sets = [
    (200, -100),  
    (200, 200),
    (-100, 200)
]

# 入射电子加速电压
Vacc = 200   # 3 keV

# 网格设置
Nx = 121
Ny = 161

# SOR 参数
omega = 1.85
tol = 1e-6
max_iter = 20000

# 常数
qe = -1.602176634e-19
qe_abs = 1.602176634e-19
me = 9.10938356e-31

# RK4 时间步长
dt = 1e-11
max_steps = 150000


# -------------------- 数值场求解 --------------------
dx = a / (Nx - 1)
dy = b / (Ny - 1)
x_grid = np.linspace(0, a, Nx)
y_grid = np.linspace(0, b, Ny)

def solve_potential(V_top, V_bot):
    V = np.zeros((Ny, Nx))
    V[-1,:] = V_top
    V[0,:]  = V_bot
    
    for j in range(Ny):
        y = j * dy
        val = V_bot + (V_top - V_bot)*(y/b)
        V[j,0]  = val
        V[j,-1] = val

    for it in range(max_iter):
        max_diff = 0
        for j in range(1,Ny-1):
            for i in range(1,Nx-1):
                k = 1/dx**2 + 1/dy**2
                V_new = (V[j+1, i]/(2*dx**2) + V[j-1, i]/(2*dx**2) + V[j, i+1]/(2*dy**2) + V[j, i-1]/(2*dy**2))/k
                diff = V_new - V[j,i]
                V[j,i] += omega * diff
                max_diff = max(max_diff, abs(diff))
        if max_diff < tol:
            break
    return V

def compute_field(V):
    Ex = np.zeros_like(V)
    Ey = np.zeros_like(V)
    Ex[:,1:-1] = -(V[:,2:] - V[:, :-2])/(2*dx)
    Ey[1:-1,:] = -(V[2:, :] - V[:-2,:])/(2*dy)
    return Ex, Ey


# 双线性插值
def bilinear(field, x, y):
    x = np.clip(x, 0, a)
    y = np.clip(y, 0, b)

    i = min(int(x/dx), Nx-2)
    j = min(int(y/dy), Ny-2)

    tx = (x - x_grid[i])/dx
    ty = (y - y_grid[j])/dy

    f00 = field[j, i]
    f10 = field[j, i+1]
    f01 = field[j+1, i]
    f11 = field[j+1, i+1]

    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

def E_at(Ex, Ey, x, y):
    return bilinear(Ex, x, y), bilinear(Ey, x, y)


# RK4 推进
def rk4_step(state, dt, Ex, Ey):
    def f(s):
        x,y,vx,vy = s
        ex, ey = E_at(Ex, Ey, x, y)
        ax = (qe/me)*ex
        ay = (qe/me)*ey
        return np.array([vx, vy, ax, ay])

    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6)*(k1+2*k2+2*k3+k4)


# -------------------- 主程序：三种电压轨迹 --------------------
colors = ["r","g","b"]
plt.figure(figsize=(7,5))

v0 = np.sqrt(2*qe_abs*Vacc/me)

for idx, (V_top, V_bot) in enumerate(voltage_sets):
    print(f"\n=== 计算电压组 {idx+1}:  V_top={V_top}, V_bot={V_bot} ===")

    V = solve_potential(V_top, V_bot)
    Ex, Ey = compute_field(V)

    # 初始状态
    state = np.array([1e-6, b/2, v0, 0.0])
    xs = [state[0]]
    ys = [state[1]]

    hit = False
    for step in range(max_steps):
        x,y,vx,vy = state
        if y<=0 or y>=b:
            hit = True
            break
        if x>=a:
            break

        state = rk4_step(state, dt, Ex, Ey)
        xs.append(state[0])
        ys.append(state[1])

    plt.plot(xs, ys, colors[idx], label=f"({V_top},{V_bot}) V")

# 画偏转板
plt.plot([0,a],[0,0],'k',linewidth=3)
plt.plot([0,a],[b,b],'k',linewidth=3)

plt.xlim(-0.01, a+0.01)
plt.ylim(-0.01, b+0.01)
plt.gca().set_aspect('equal','box')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("不同电极电压下电子轨迹（RK4）（251853王映雪）")
plt.grid(True)
plt.legend()
plt.show()
