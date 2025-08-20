import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import sys

warnings.filterwarnings('ignore')

"""
核心修复点（导致“校正区间全都一样”的原因 & 解决办法）
1) 旧版仅用常量占位特征 X=1，树无法分裂 → 只有一个叶子 → 全样本预测相同。
   ✅ 现在将 (a) n_smoothed 本身、(b) mode 的 one-hot 作为特征，
      使模型能对不同模式、不同样本学习“校正偏移”。
2) 全局物理引导项使用 (y_pred - mean(y_pred))，在单叶场景会被抵消。
   ✅ 现在改为每个样本直接施加方向性“推力”（常数梯度偏置），
      与 event_srv_ratio / length_height_ratio / fracture_height 正相关，
      使 mode 2/3/4 方向上更容易“抬升”预测；mode 1 轻微“压低”。
3) 训练时引入 sample_weight（按全局微地震特征为不同模式样本加权），
   让与“全局约束”一致的模式样本权重更大，从而主动“靠拢”微地震。

使用方式：
python XGBoostfracture_model_fixed.py path_to_excel.xlsx
Excel 至少包含列：event_srv_ratio, length_height_ratio, fracture_height, n_smoothed, mode
"""

# -------------------------------
# 全局缓存（用于自定义损失访问）
# -------------------------------
TRAIN_CONSTRAINTS = None
TEST_CONSTRAINTS = None
GLOBAL_EVENT = None
GLOBAL_LENGTH_RATIO = None
GLOBAL_HEIGHT = None


# -------------------------------
# 1) 数据加载与校验
# -------------------------------
def load_data(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    required = ['event_srv_ratio', 'length_height_ratio', 'fracture_height', 'n_smoothed', 'mode']
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"数据缺少必要列: {miss}")

    # 关键修复：先提取全局变量（基于原始数据，包含所有mode）
    global GLOBAL_EVENT, GLOBAL_LENGTH_RATIO, GLOBAL_HEIGHT
    GLOBAL_EVENT = df['event_srv_ratio'].dropna().unique()
    GLOBAL_LENGTH_RATIO = df['length_height_ratio'].dropna().unique()
    GLOBAL_HEIGHT = df['fracture_height'].dropna().unique()
    # 校验全局变量是否为单值
    if len(GLOBAL_EVENT) != 1 or len(GLOBAL_LENGTH_RATIO) != 1 or len(GLOBAL_HEIGHT) != 1:
        raise ValueError('event_srv_ratio / length_height_ratio / fracture_height 应为整段单值')
    GLOBAL_EVENT = float(GLOBAL_EVENT[0])
    GLOBAL_LENGTH_RATIO = float(GLOBAL_LENGTH_RATIO[0])
    GLOBAL_HEIGHT = float(GLOBAL_HEIGHT[0])
    print("\n📌 全局微地震约束:")
    print(f"  event_srv_ratio = {GLOBAL_EVENT}")
    print(f"  length_height_ratio = {GLOBAL_LENGTH_RATIO}")
    print(f"  fracture_height = {GLOBAL_HEIGHT}")

    # 再过滤mode（只保留1-4）
    df = df[df['mode'].isin([1, 2, 3, 4])].copy()
    df['mode'] = df['mode'].astype(int)

    # n_smoothed 清洗（不变）
    if df['n_smoothed'].isna().any():
        df['n_smoothed'].fillna(df['n_smoothed'].mean(), inplace=True)
    if np.isinf(df['n_smoothed']).any():
        raise ValueError('n_smoothed 存在无穷大，请先清洗')

    return df


# -------------------------------
# 2) 构建特征与约束
# -------------------------------
def prepare_dataset(df: pd.DataFrame):
    # 统计每个模式在原始判据下的区间（机理先验）
    mode_stats = df.groupby('mode')['n_smoothed'].agg(min_mode='min', max_mode='max').reset_index()

    df = df.merge(mode_stats, on='mode', how='left')

    # 特征：
    #  - x0: 原始 n_smoothed（让模型学习“微调/校正”而非从零拟合）
    #  - x1..x4: mode 的 one-hot（让不同模式可走到不同叶子）
    mode_onehot = pd.get_dummies(df['mode'], prefix='mode')
    X = pd.concat([df[['n_smoothed']].reset_index(drop=True), mode_onehot.reset_index(drop=True)], axis=1)

    # 目标：校正后的 n_smoothed（初始用原值，训练中靠损失推动到新位置）
    y = df['n_smoothed'].to_numpy(dtype=np.float64)

    # 约束（供损失函数使用）
    constraints = df[['min_mode', 'max_mode', 'mode']].to_numpy(dtype=np.float64)

    return X.to_numpy(dtype=np.float64), y, constraints, df, mode_stats


# -------------------------------
# 3) 规范化一个“力度系数”（使不同量纲可比）
# -------------------------------
def _norm_positive(x: float) -> float:
    # 平滑正向缩放到 ~[0,1.5) 区间（对极端值不太敏感）
    return float(np.log1p(max(x, 0.0)))


# -------------------------------
# 4) 自定义损失：机理区间 + 全局引导
# -------------------------------

def physics_constrained_loss(y_pred, dtrain):
    global TRAIN_CONSTRAINTS, TEST_CONSTRAINTS, GLOBAL_EVENT, GLOBAL_LENGTH_RATIO, GLOBAL_HEIGHT

    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    y_true = dtrain.get_label().astype(np.float64).flatten()
    n = y_pred.size

    constraints = TRAIN_CONSTRAINTS if n == TRAIN_CONSTRAINTS.shape[0] else TEST_CONSTRAINTS
    constraints = np.asarray(constraints).reshape(-1, 3)
    n_min, n_max, mode = constraints[:, 0], constraints[:, 1], constraints[:, 2].astype(int)

    # 基础 MSE
    grad = 2.0 * (y_pred - y_true)
    hess = np.full(n, 2.0, dtype=np.float64)

    # 区间惩罚（偏离机理区间时强烈拉回）
    k_interval = 6.0
    mask_lower = y_pred < n_min
    mask_upper = y_pred > n_max
    grad[mask_lower] += k_interval * 2.0 * (y_pred[mask_lower] - n_min[mask_lower])
    grad[mask_upper] += k_interval * 2.0 * (y_pred[mask_upper] - n_max[mask_upper])
    hess[mask_lower] += k_interval * 2.0
    hess[mask_upper] += k_interval * 2.0

    # 全局引导（按模式给定“抬升/压低”的方向性偏置）
    #  4 ↔ event_srv_ratio 更大 → 倾向更高
    #  3 ↔ length_height_ratio 更大 → 倾向更高
    #  2 ↔ fracture_height 更大 → 倾向更高
    #  1 ↔ 倾向更低
    push_4 = +0.25 * _norm_positive(GLOBAL_EVENT)
    push_3 = +0.20 * _norm_positive(GLOBAL_LENGTH_RATIO)
    push_2 = +0.15 * _norm_positive(GLOBAL_HEIGHT)
    push_1 = -0.10  # 轻微压低

    grad[mode == 4] += push_4
    grad[mode == 3] += push_3
    grad[mode == 2] += push_2
    grad[mode == 1] += push_1

    return grad, hess


# -------------------------------
# 5) 评估指标：物理单调性（1<2<3<4）
# -------------------------------

def evaluate_physics_consistency(y_pred, dtrain):
    y_pred = np.asarray(y_pred).flatten()
    constraints = TRAIN_CONSTRAINTS if y_pred.size == TRAIN_CONSTRAINTS.shape[0] else TEST_CONSTRAINTS
    mode = constraints[:, 2].astype(int)

    means = {m: float(np.mean(y_pred[mode == m])) for m in [1, 2, 3, 4]}
    score = 0.0
    score += 0.25 if means[1] < means[2] else 0.0
    score += 0.25 if means[2] < means[3] else 0.0
    score += 0.25 if means[3] < means[4] else 0.0
    score += 0.25 * (1 - np.std(list(means.values())) / (np.mean(list(means.values())) + 1e-8))
    return 'physics_score', float(score)


# -------------------------------
# 6) 基于全局约束的 sample weight（可选加强）
# -------------------------------

def build_sample_weight(df: pd.DataFrame) -> np.ndarray:
    # 让“全局约束更匹配的模式样本”权重大一些
    w = np.ones(len(df), dtype=np.float64)
    w[df['mode'] == 4] *= 1.0 + 0.5 * _norm_positive(float(df['event_srv_ratio'].iloc[0]))
    w[df['mode'] == 3] *= 1.0 + 0.4 * _norm_positive(float(df['length_height_ratio'].iloc[0]))
    w[df['mode'] == 2] *= 1.0 + 0.3 * _norm_positive(float(df['fracture_height'].iloc[0]))
    # mode 1 保持基线权重
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)  # 替换掉任何异常
    w[w <= 0] = 1e-6  # 保证最小值 > 0

    return w


# -------------------------------
# 7) 训练
# -------------------------------

def train_model(X, y, constraints, sample_weight):
    global TRAIN_CONSTRAINTS, TEST_CONSTRAINTS

    X_train, X_test, y_train, y_test, c_train, c_test, w_train, w_test = train_test_split(
        X, y, constraints, sample_weight, test_size=0.2, random_state=42, stratify=constraints[:, 2]
    )

    TRAIN_CONSTRAINTS = c_train.reshape(-1, 3)
    TEST_CONSTRAINTS = c_test.reshape(-1, 3)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)

    params = {
        'max_depth': 3,
        'min_child_weight': 1.0,
        'eta': 0.08,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'verbosity': 1,
        'random_state': 42
    }

    print("\n📊 开始训练模型…")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        obj=physics_constrained_loss,
        feval=evaluate_physics_consistency,
        evals=[(dtest, 'valid'), (dtrain, 'train')],
        early_stopping_rounds=30,
        verbose_eval=25
    )
    print(f"✅ 训练完成，最佳迭代: {model.best_iteration}")
    return model


# -------------------------------
# 8) 生成校正区间与阈值
# -------------------------------

def generate_corrected_intervals(model, X, df):
    dmat = xgb.DMatrix(X)
    df = df.copy()
    df['corrected_ns'] = model.predict(dmat).astype(float)

    stats = df.groupby('mode')['corrected_ns'].agg(['min', 'max', 'mean']).reset_index()
    intervals = {int(r['mode']): [float(r['min']), float(r['max'])] for _, r in stats.iterrows()}

    # 相邻模式阈值：取上/下边界的中点
    modes_sorted = sorted(intervals.keys())
    thresholds = {}
    for i in range(len(modes_sorted) - 1):
        m_curr, m_next = modes_sorted[i], modes_sorted[i + 1]
        thresholds[f't{i+1}'] = (intervals[m_curr][1] + intervals[m_next][0]) / 2.0

    return intervals, thresholds, df


# -------------------------------
# 9) 物理性检查 & 可视化
# -------------------------------

def verify_constraints(df):
    print("\n🔍 物理约束验证（期望 1<2<3<4）：")
    means = df.groupby('mode')['corrected_ns'].mean().to_dict()
    print("  各模式均值:", {int(k): float(v) for k, v in means.items()})
    seq_ok = means.get(1, 0) < means.get(2, 0) < means.get(3, 0) < means.get(4, 0)
    print("  结果:", "✅ 符合" if seq_ok else "❌ 不符合")


def visualize_results(df, thresholds):
    plt.rcParams["font.family"] = ["SimHei"]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {1: '#FF5733', 2: '#33AA55', 3: '#3366FF', 4: '#F3C233'}
    labels = {1: '突破屏障', 2: '缝高扩展', 3: '缝高受限', 4: '缝网扩展'}

    for m in [1, 2, 3, 4]:
        data = df[df['mode'] == m]['corrected_ns']
        if len(data):
            ax.hist(data, bins=12, alpha=0.65, label=labels[m], color=colors[m])

    for name, val in sorted(thresholds.items(), key=lambda x: x[1]):
        ax.axvline(val, linestyle='--', alpha=0.6)
        ax.text(val, ax.get_ylim()[1]*0.95, name, rotation=90, va='top', ha='center')

    ax.set_title('校正后 n_smoothed 分布与阈值')
    ax.set_xlabel('corrected_ns')
    ax.set_ylabel('样本数')
    ax.legend()
    plt.tight_layout()
    plt.savefig('校正结果可视化.png', dpi=300)
    print("\n📈 已保存图像: 校正结果可视化.png")
    plt.show()


# -------------------------------
# 10) 主程序
# -------------------------------

def main():
    # 直接指定Excel文件路径（调试时使用）
    excel_path = r"C:\Users\fy\Desktop\压裂施工数据+微地震参数\readresults.xlsx"

    df = load_data(excel_path)
    X, y, constraints, df, _ = prepare_dataset(df)
    sample_weight = build_sample_weight(df)

    model = train_model(X, y, constraints, sample_weight)
    intervals, thresholds, df_out = generate_corrected_intervals(model, X, df)

    print("\n📌 校正后的模式判别区间:")
    name_map = {1: '突破屏障', 2: '缝高扩展', 3: '缝高受限', 4: '缝网扩展'}
    for m in sorted(intervals.keys()):
        lo, hi = intervals[m]
        print(f"  {name_map[m]}: [{lo:.4f}, {hi:.4f}]")

    print("\n📌 模式判别阈值:")
    for k, v in sorted(thresholds.items(), key=lambda kv: kv[1]):
        print(f"  {k}: {v:.4f}")

    verify_constraints(df_out)
    visualize_results(df_out, thresholds)

    # 导出判别函数
    print("\n📝 模式判别函数：")
    print(f"""def predict_mode(n_smoothed, thresholds={ {k: float(v) for k, v in thresholds.items()} }):\n    ts = [v for _, v in sorted(thresholds.items(), key=lambda kv: kv[1])]\n    t1, t2, t3 = ts\n    if n_smoothed <= t1: return 1\n    elif n_smoothed <= t2: return 2\n    elif n_smoothed <= t3: return 3\n    else: return 4\n""")


if __name__ == '__main__':
    main()
