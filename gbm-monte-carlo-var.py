import numpy as np
import matplotlib.pyplot as plt

# ---- 参数 ----
S0, mu, sigma = 100, 0.10, 0.20   # 初始价、年化收益率、年化波动率
days, n_paths = 252, 80
dt = 1 / 252

# ---- 向量化 GBM 模拟 ----
np.random.seed(42)
Z = np.random.standard_normal((n_paths, days))
drift = (mu - 0.5 * sigma**2) * dt
diffusion = sigma * np.sqrt(dt) * Z
log_returns = drift + diffusion
log_paths = np.cumsum(log_returns, axis=1)
S = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))

# ---- 价格路径图 ----
plt.figure(figsize=(10, 5))
plt.plot(S.T, lw=0.8, alpha=0.6)
plt.title("GBM Monte Carlo Simulated Price Paths")
plt.xlabel("Trading Day"); plt.ylabel("Price")
plt.show()

# ---- 末期价格分布 ----
final_prices = S[:, -1]
plt.figure(figsize=(8, 4))
plt.hist(final_prices, bins=30, color="steelblue", edgecolor="white")
plt.title("Final Price Distribution (Day %d)" % days)
plt.show()

# ---- 95% 置信度历史模拟法 VaR ----
returns = (final_prices - S0) / S0
VaR_95_pct = -np.percentile(returns, 5)          # 收益率分位数
VaR_95_value = S0 * VaR_95_pct                     # 对应货币损失

print(f"期望末期价格: {final_prices.mean():.2f}")
print(f"最高/最低价格: {final_prices.max():.2f} / {final_prices.min():.2f}")
print(f"95% VaR (收益率): {VaR_95_pct*100:.2f}%")
print(f"95% VaR (货币): {VaR_95_value:.2f}")