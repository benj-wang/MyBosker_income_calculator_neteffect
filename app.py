import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide", page_title="双曲价格模型分析器", page_icon="📈")

# 设置更大的参数范围
PARAM_RANGES = {
    "P_min": (10, 500, "最低价格 (P_min)"),
    "K": (50, 500, "价格范围 (K)"),
    "N0": (1, 200, "参考人数 (N0)"),
    "q": (0.5, 5.0, "衰减指数 (q)"),
    "N": (1, 200, "实际人数 (N)"),
}

# 初始化参数
if 'params' not in st.session_state:
    st.session_state.params = {
        "P_min": 50,
        "K": 80,
        "N0": 20,
        "q": 2.0,
        "N": 50
    }

# 标题和介绍
st.title("📈 双曲价格模型分析器")
st.markdown("""
此工具用于分析双曲价格模型：  
$P(N) = P_{\\text{min}} + \\frac{K}{(1 + N/N_0)^q}$  
其中 $P(N)$ 是当有 $N$ 人参与时的每人价格  
总收入为 $I(N) = P(N) \\times N$
""")

# 侧边栏参数控制
with st.sidebar:
    st.header("🛠️ 参数设置")
    
    # 创建参数输入控件
    for param, (min_val, max_val, label) in PARAM_RANGES.items():
        # 数值输入框
        st.markdown(f"**{label}**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # 对于整数参数，使用步长1
            if param in ["P_min", "N0", "N"]:
                value = st.number_input(
                    label, 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=st.session_state.params[param],
                    key=f"{param}_input",
                    step=1
                )
            else:
                value = st.number_input(
                    label, 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=st.session_state.params[param],
                    key=f"{param}_input",
                    step=0.1
                )
        
        with col2:
            st.write("")  # 占位符
            st.write("")  # 占位符
            st.session_state.params[param] = value
    
    st.markdown("---")
    
    # 计算按钮
    if st.button("🔄 重新计算", use_container_width=True):
        st.experimental_rerun()

# 计算函数
def calculate_price(N, P_min, K, N0, q):
    """计算价格函数 P(N) = P_min + K/(1 + N/N0)^q"""
    return P_min + K / np.power(1 + N/N0, q)

def calculate_price_derivative(N, P_min, K, N0, q, h=1e-5):
    """数值计算一阶导数 P'(N)"""
    return (calculate_price(N + h, P_min, K, N0, q) - 
            calculate_price(N - h, P_min, K, N0, q)) / (2 * h)

def calculate_income(N, P_min, K, N0, q):
    """计算收入函数 I(N) = P(N) * N"""
    return calculate_price(N, P_min, K, N0, q) * N

def calculate_income_derivative(N, P_min, K, N0, q, h=1e-5):
    """数值计算收入的一阶导数 I'(N)"""
    return (calculate_income(N + h, P_min, K, N0, q) - 
            calculate_income(N - h, P_min, K, N0, q)) / (2 * h)

def calculate_income_second_derivative(N, P_min, K, N0, q, h=1e-5):
    """数值计算收入的二阶导数 I''(N)"""
    return (calculate_income(N + h, P_min, K, N0, q) - 
            2 * calculate_income(N, P_min, K, N0, q) + 
            calculate_income(N - h, P_min, K, N0, q)) / (h**2)

# 获取当前参数
P_min = st.session_state.params["P_min"]
K = st.session_state.params["K"]
N0 = st.session_state.params["N0"]
q = st.session_state.params["q"]
N = st.session_state.params["N"]

# 计算当前值
current_price = calculate_price(N, P_min, K, N0, q)
current_income = calculate_income(N, P_min, K, N0, q)
dP_dN = calculate_price_derivative(N, P_min, K, N0, q)
dI_dN = calculate_income_derivative(N, P_min, K, N0, q)
d2I_dN2 = calculate_income_second_derivative(N, P_min, K, N0, q)

# 条件验证
cond1 = dP_dN < 0  # 价格随人数增加而下降
cond2 = dI_dN > 0  # 收入随人数增加而增加
cond3 = d2I_dN2 > 0  # 收入增长呈现下凸性（凸函数）

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
             delta_color="off", help="正值表示收入增长呈下凸趋势")

# 条件验证状态
st.subheader("✅ 条件验证")
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
                f"<b>收入凸性:</b> {status}</div>", 
                unsafe_allow_html=True)
    st.write("收入增长应呈现下凸趋势（加速增长）")

# 生成图表数据
N_values = np.linspace(1, 200, 400)
P_values = [calculate_price(n, P_min, K, N0, q) for n in N_values]
I_values = [calculate_income(n, P_min, K, N0, q) for n in N_values]
dI_values = [calculate_income_derivative(n, P_min, K, N0, q) for n in N_values]
d2I_values = [calculate_income_second_derivative(n, P_min, K, N0, q) for n in N_values]

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 价格曲线
ax1.plot(N_values, P_values, 'b-', linewidth=2, label='价格曲线')
ax1.axvline(N, color='r', linestyle='--', alpha=0.5)
ax1.axhline(current_price, color='r', linestyle='--', alpha=0.5)
ax1.plot(N, current_price, 'ro', markersize=8)
ax1.set_title(f"每人价格 (P_min={P_min}, K={K}, N0={N0}, q={q:.2f})")
ax1.set_xlabel("人数 (N)")
ax1.set_ylabel("每人价格")
ax1.grid(True, alpha=0.3)
ax1.legend()

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

# 收入一阶导数（增长率）
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

# 收入二阶导数（加速度）
ax4.plot(N_values, d2I_values, 'c-', linewidth=2, label='收入加速度')
ax4.axhline(0, color='k', linestyle='-', alpha=0.3)
ax4.axvline(N, color='r', linestyle='--', alpha=0.5)
ax4.axhline(d2I_dN2, color='r', linestyle='--', alpha=0.5)
ax4.plot(N, d2I_dN2, 'ro', markersize=8)
ax4.set_title(f"收入加速度 (d²I/dN², 当前: {d2I_dN2:.4f})")
ax4.set_xlabel("人数 (N)")
ax4.set_ylabel("收入加速度")
ax4.grid(True, alpha=0.3)
ax4.legend()

# 调整布局
plt.tight_layout()

# 显示图表
st.pyplot(fig)

# 参数影响分析
st.subheader("🔍 参数影响分析")
st.markdown("""
- **最低价格 (P_min):** 影响价格曲线的下限。增加 P_min 会提高整体价格水平，但会降低价格弹性。
- **价格范围 (K):** 控制价格从最高点到最低点的变化幅度。增加 K 会使价格曲线更加陡峭。
- **参考人数 (N0):** 决定价格开始显著下降的转折点。较小的 N0 表示价格在较少人数时就开始下降。
- **衰减指数 (q):** 控制价格下降的速度。较大的 q 值表示价格下降更快。
- **实际人数 (N):** 直接影响当前价格和收入。随着 N 增加，价格下降但收入可能增加。
""")

# 灵敏度分析
st.subheader("📐 参数灵敏度分析")
st.markdown("调整参数并观察结果如何变化，以了解每个参数对模型的影响程度。")

# 数据表
if st.checkbox("显示详细数据表"):
    data = {
        "人数": N_values,
        "每人价格": P_values,
        "总收入": I_values,
        "收入增长率": dI_values,
        "收入加速度": d2I_values
    }
    df = pd.DataFrame(data)
    st.dataframe(df.style.format("{:.2f}"), height=300)

# 使用说明
st.subheader("📖 使用说明")
st.markdown("""
1. 在左侧边栏调整参数值
2. 观察图表中的曲线如何变化
3. 关注当前计算结果区域的关键指标
4. 检查条件验证部分是否满足业务需求
5. 使用参数灵敏度分析理解每个参数的影响
""")

# 页脚
st.markdown("---")
st.markdown("© 2023 双曲价格模型分析器 | 基于双曲价格函数 $P(N) = P_{\\text{min}} + \\frac{K}{(1 + N/N_0)^q}$")