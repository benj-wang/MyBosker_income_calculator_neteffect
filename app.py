import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide", page_title="增强激励机制价格模型分析器", page_icon="📈")

# 设置参数范围
PARAM_RANGES = {
    "P_min": (10.0, 100.0, "最低价格 (P_min)"),
    "K": (50.0, 300.0, "初始价格溢价 (K)"),
    "q": (0.01, 1.0, "指数衰减因子 (q)"),
    "N0": (5.0, 50.0, "指数衰减参考人数 (N0)"),
    "C": (0.0, 100.0, "双曲调整因子 (C)"),
    "r": (0.1, 3.0, "双曲衰减因子 (r)"),
    "N1": (20.0, 200.0, "双曲衰减参考人数 (N1)"),
    "N": (1.0, 200.0, "实际人数 (N)"),
}

# 初始化参数 - 优化后的默认值
if 'params' not in st.session_state:
    st.session_state.params = {
        "P_min": 20.0,
        "K": 100.0,
        "q": 0.2,
        "N0": 15.0,
        "C": 50.0,
        "r": 1.5,
        "N1": 100.0,
        "N": 50.0
    }

# 标题和介绍
st.title("📈 增强激励机制价格模型分析器")
st.markdown("""
**改进的价格模型**：  
$P(N) = P_{\\text{min}} + K \\cdot e^{-q \\cdot (N/N_0)} + \\frac{C}{1 + (N/N_1)^r}$  

**收入函数**：  
$I(N) = P(N) \\times N$

**模型特点**：
1. 收入始终增长（dI/dN > 0）
2. 初期增长快速，提供强烈激励
3. 后期增长放缓但仍保持增长动力
4. 价格合理下降，反映规模效应
""")

# 侧边栏参数控制
with st.sidebar:
    st.header("🛠️ 参数设置")
    
    # 创建参数输入控件
    for param, (min_val, max_val, label) in PARAM_RANGES.items():
        # 数值输入框
        st.markdown(f"**{label}**")
        
        # 当前值
        current_value = st.session_state.params[param]
        
        # 使用统一的浮点数输入控件
        value = st.number_input(
            label, 
            min_value=float(min_val), 
            max_value=float(max_val), 
            value=float(current_value),
            step=0.1,
            format="%.1f",
            key=f"{param}_input"
        )
        
        # 更新参数值
        st.session_state.params[param] = float(value)
    
    st.markdown("---")
    
    # 计算按钮
    if st.button("🔄 重新计算", use_container_width=True):
        st.experimental_rerun()

# 计算函数 - 改进的价格模型
def calculate_price(N, params):
    """计算价格函数 P(N) = P_min + K*exp(-q*(N/N0)) + C/(1+(N/N1)^r)"""
    P_min = params["P_min"]
    K = params["K"]
    q = params["q"]
    N0 = params["N0"]
    C = params["C"]
    r = params["r"]
    N1 = params["N1"]
    
    exp_part = K * np.exp(-q * (N / N0))
    hyperbolic_part = C / (1 + (N / N1) ** r)
    
    return P_min + exp_part + hyperbolic_part

def calculate_price_derivative(N, params, h=1e-5):
    """数值计算一阶导数 P'(N)"""
    return (calculate_price(N + h, params) - calculate_price(N - h, params)) / (2 * h)

def calculate_income(N, params):
    """计算收入函数 I(N) = P(N) * N"""
    return calculate_price(N, params) * N

def calculate_income_derivative(N, params, h=1e-5):
    """数值计算收入的一阶导数 I'(N)"""
    return (calculate_income(N + h, params) - calculate_income(N - h, params)) / (2 * h)

def calculate_income_second_derivative(N, params, h=1e-5):
    """数值计算收入的二阶导数 I''(N)"""
    return (calculate_income(N + h, params) - 
            2 * calculate_income(N, params) + 
            calculate_income(N - h, params)) / (h**2)

# 获取当前参数
params = st.session_state.params
N = params["N"]

# 计算当前值
try:
    current_price = calculate_price(N, params)
    current_income = calculate_income(N, params)
    dP_dN = calculate_price_derivative(N, params)
    dI_dN = calculate_income_derivative(N, params)
    d2I_dN2 = calculate_income_second_derivative(N, params)
except Exception as e:
    st.error(f"计算错误: {str(e)}")
    # 使用默认值继续运行
    current_price = params["P_min"] + params["K"]
    current_income = (params["P_min"] + params["K"]) * N
    dP_dN = 0
    dI_dN = 0
    d2I_dN2 = 0

# 条件验证
cond1 = dP_dN < 0  # 价格随人数增加而下降
cond2 = dI_dN > 0  # 收入随人数增加而增加
cond3 = d2I_dN2 < 0  # 收入增长呈现上凸性（增长减速但仍在增长）

# 显示当前结果
st.subheader("📊 当前计算结果")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("每人价格", f"{current_price:.2f}")

with col2:
    st.metric("总收入", f"{current_income:.2f}")

with col3:
    st.metric("价格变化率 (dP/dN)", f"{dP_dN:.4f}", 
             delta_color="off", help="负值表示价格随人数增加而下降")

with col4:
    st.metric("收入增长率 (dI/dN)", f"{dI_dN:.4f}", 
             delta_color="off", help="正值表示收入随人数增加而增加")

with col5:
    st.metric("收入加速度 (d²I/dN²)", f"{d2I_dN2:.4f}", 
             delta_color="off", help="负值表示收入增长放缓但仍在增长")

# 条件验证状态
st.subheader("✅ 模型特性验证")
cond_col1, cond_col2, cond_col3 = st.columns(3)

with cond_col1:
    status = "满足" if cond1 else "不满足"
    color = "green" if cond1 else "red"
    st.markdown(f"<div style='background-color:{color};color:white;padding:10px;border-radius:5px;'>"
                f"<b>价格递减:</b> {status}</div>", 
                unsafe_allow_html=True)
    st.write("当人数增加时，每人价格应该下降")

with cond_col2:
    status = "满足" if cond2 else "不满足"
    color = "green" if cond2 else "red"
    st.markdown(f"<div style='background-color:{color};color:white;padding:10px;border-radius:5px;'>"
                f"<b>收入递增:</b> {status}</div>", 
                unsafe_allow_html=True)
    st.write("当人数增加时，总收入应该增加")

with cond_col3:
    status = "满足" if cond3 else "不满足"
    color = "green" if cond3 else "red"
    st.markdown(f"<div style='background-color:{color};color:white;padding:10px;border-radius:5px;'>"
                f"<b>合理增长减速:</b> {status}</div>", 
                unsafe_allow_html=True)
    st.write("收入增长应呈现上凸趋势（合理减速但仍保持增长）")

# 生成图表数据
try:
    N_values = np.linspace(1, 200, 400)
    P_values = [calculate_price(n, params) for n in N_values]
    I_values = [calculate_income(n, params) for n in N_values]
    dI_values = [calculate_income_derivative(n, params) for n in N_values]
    d2I_values = [calculate_income_second_derivative(n, params) for n in N_values]
except Exception as e:
    st.error(f"图表数据生成错误: {str(e)}")
    # 使用默认数据继续运行
    N_values = np.linspace(1, 200, 400)
    P_values = [params["P_min"] + params["K"] for _ in N_values]
    I_values = [(params["P_min"] + params["K"]) * n for n in N_values]
    dI_values = [0 for _ in N_values]
    d2I_values = [0 for _ in N_values]

# 创建图表
try:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 价格曲线
    ax1.plot(N_values, P_values, 'b-', linewidth=2, label='价格曲线')
    ax1.axvline(N, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(current_price, color='r', linestyle='--', alpha=0.5)
    ax1.plot(N, current_price, 'ro', markersize=8)
    ax1.set_title(f"每人价格 (当前: {current_price:.2f})")
    ax1.set_xlabel("人数 (N)")
    ax1.set_ylabel("每人价格")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加价格组成分解
    exp_part = [params["K"] * np.exp(-params["q"] * (n / params["N0"])) for n in N_values]
    hyperbolic_part = [params["C"] / (1 + (n / params["N1"]) ** params["r"]) for n in N_values]
    ax1.plot(N_values, [params["P_min"] for _ in N_values], 'g--', alpha=0.7, label='最低价格')
    ax1.plot(N_values, [params["P_min"] + e for e in exp_part], 'c--', alpha=0.7, label='指数部分')
    ax1.plot(N_values, [params["P_min"] + h for h in hyperbolic_part], 'm--', alpha=0.7, label='双曲部分')

    # 收入曲线
    ax2.plot(N_values, I_values, 'g-', linewidth=2, label='收入曲线')
    ax2.axvline(N, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(current_income, color='r', linestyle='--', alpha=0.5)
    ax2.plot(N, current_income, 'ro', markersize=8)
    ax2.set_title(f"总收入 (当前: {current_income:.2f})")
    ax2.set_xlabel("人数 (N)")
    ax2.set_ylabel("总收入")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 收入增长率曲线
    ax3.plot(N_values, dI_values, 'm-', linewidth=2, label='收入增长率')
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(N, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(dI_dN, color='r', linestyle='--', alpha=0.5)
    ax3.plot(N, dI_dN, 'ro', markersize=8)
    ax3.set_title(f"收入增长率 (dI/dN, 当前: {dI_dN:.4f})")
    ax3.set_xlabel("人数 (N)")
    ax3.set_ylabel("收入增长率")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 添加理想增长区域
    ax3.fill_between(N_values, 0, max(dI_values)*1.1, where=(dI_values > 0), color='green', alpha=0.1)
    ax3.text(150, max(dI_values)*0.8, "收入增长区", color='green', fontsize=12)

    # 收入增长率变化曲线
    ax4.plot(N_values, d2I_values, 'c-', linewidth=2, label='收入增长率变化')
    ax4.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(N, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(d2I_dN2, color='r', linestyle='--', alpha=0.5)
    ax4.plot(N, d2I_dN2, 'ro', markersize=8)
    ax4.set_title(f"收入增长率变化 (d²I/dN², 当前: {d2I_dN2:.4f})")
    ax4.set_xlabel("人数 (N)")
    ax4.set_ylabel("收入增长率变化")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 添加合理减速区域
    ax4.fill_between(N_values, min(d2I_values)*1.1, 0, where=(d2I_values < 0), color='blue', alpha=0.1)
    ax4.text(150, min(d2I_values)*0.8, "合理减速区", color='blue', fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 显示图表
    st.pyplot(fig)
except Exception as e:
    st.error(f"图表绘制错误: {str(e)}")

# 参数影响分析
st.subheader("🔍 参数影响分析")
st.markdown("""
- **最低价格 (P_min):** 决定价格下限，影响后期收入增长
- **初始价格溢价 (K):** 影响初始价格水平，值越大初期价格越高
- **指数衰减因子 (q):** 控制初期价格下降速度，值越大初期降价越快
- **指数参考人数 (N0):** 决定初期价格下降转折点
- **双曲调整因子 (C):** 影响中期价格水平，提供额外激励机制
- **双曲衰减因子 (r):** 控制后期价格下降速度
- **双曲参考人数 (N1):** 决定后期价格下降转折点
- **实际人数 (N):** 当前参与人数，直接影响价格和收入
""")

# 激励机制优化建议
st.subheader("💡 激励机制优化建议")
st.markdown("""
1. **初期强激励**：
   - 增加 `初始价格溢价 (K)` 使初期价格更高
   - 增加 `指数衰减因子 (q)` 使初期价格下降更快
   - 减小 `指数参考人数 (N0)` 使价格下降更早开始

2. **中期持续激励**：
   - 增加 `双曲调整因子 (C)` 提供中期价格支撑
   - 调整 `双曲衰减因子 (r)` 控制中期价格下降速度

3. **后期稳定增长**：
   - 设置合理的 `最低价格 (P_min)` 确保后期收入增长
   - 增加 `双曲参考人数 (N1)` 延缓后期价格下降
""")

# 预设优化方案
st.subheader("🚀 预设优化方案")
col1, col2 = st.columns(2)

with col1:
    if st.button("方案1：强初期激励"):
        st.session_state.params.update({
            "P_min": 25.0,
            "K": 150.0,
            "q": 0.3,
            "N0": 10.0,
            "C": 30.0,
            "r": 1.2,
            "N1": 120.0
        })
        st.experimental_rerun()

with col2:
    if st.button("方案2：平衡激励"):
        st.session_state.params.update({
            "P_min": 30.0,
            "K": 120.0,
            "q": 0.25,
            "N0": 15.0,
            "C": 50.0,
            "r": 1.5,
            "N1": 100.0
        })
        st.experimental_rerun()

# 使用说明
st.subheader("📖 使用说明")
st.markdown("""
1. 在左侧边栏调整参数值
2. 观察图表中的曲线变化
3. 查看"优化建议"调整激励机制
4. 尝试使用预设优化方案
5. 确保"收入增长率 (dI/dN)"始终为正
6. 确保"收入增长率变化 (d²I/dN²)"为负，表示合理减速
""")

# 页脚
st.markdown("---")
st.markdown("© 2023 增强激励机制价格模型分析器 | 基于指数-双曲混合价格函数")