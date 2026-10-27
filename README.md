# FlatBall — 平面到球面的法向位移异常迁移实验

本仓库把 `exp.md` 中的实验设计实现为可配置、可复现、可扩展的代码：在平面点云上训练点级异常检测器，在平面与不同半径的球面上测试，并围绕 H1–H4 输出配对差异、置信区间、等价性检验和曲率趋势。

仓库只包含代码、配置、测试与使用说明；不包含实验结论或结果文件。所有数值结论都由运行产生。

---

## 1. 基本结构

```text
PcExp1_FlatBall/
├─ README.md
├─ pyproject.toml
├─ requirements.txt              # numpy / scipy / scikit-learn
├─ requirements-optional.txt     # torch / xgboost / pytest
├─ configs/
│  ├─ quick.json                 # 快速冒烟：N=1024，1 个 seed，仅 exp1
│  └─ full.json                  # 协议规模：N=10240，5 seeds，exp1/2/3
├─ src/flatball/
│  ├─ config.py                  # 冻结 dataclass 配置 + 校验 + 缓存指纹
│  ├─ runner.py                  # seed/实验编排、metrics.csv、summary 输出
│  ├─ cli.py / __main__.py       # flatball run | list | aggregate
│  ├─ geometry/
│  │  ├─ surfaces.py             # 平面/球面采样、FPS、测地距离、异常中心采样
│  │  ├─ neighborhoods.py        # 统一 PCA 邻域、二次曲面拟合、6 维特征
│  │  └─ anomalies.py            # 同构法向位移注入、区域规划
│  ├─ data/
│  │  ├─ conditions.py           # R × 符号 × 幅度模式 的实验条件
│  │  ├─ features.py             # FeatureExtractor 门面
│  │  ├─ generation.py           # 每个条件的 train/val/plane_test/sphere_test
│  │  ├─ dataset.py              # DatasetSplit / DatasetBundle
│  │  └─ cache.py                # 特征缓存（可关闭）
│  ├─ models/
│  │  ├─ base.py                 # PointClassifier 接口
│  │  ├─ gbdt.py                 # 模型 A：GBDT（XGBoost 可选，sklearn 回退）
│  │  └─ pointnet.py             # 模型 B：轻量 PointNet（PyTorch 可选）
│  ├─ evaluation/
│  │  ├─ metrics.py              # AUROC、FNR/FPR、阈值选择
│  │  ├─ stats.py                # 配对 t 区间、TOST、非劣性、Spearman
│  │  ├─ evaluate.py             # 模型 × 条件 × 域 → 指标记录
│  │  ├─ records.py              # metrics.csv 读写
│  │  └─ reporting.py            # summary.json / summary.md、判定标记
│  └─ experiments/
│     ├─ base.py                 # Experiment 抽象 + 数据集/模型上下文
│     ├─ exp1.py                 # H1 + H2
│     ├─ exp2.py                 # H3（符号 × 幅度）
│     └─ exp3.py                 # H4（GBDT vs PointNet）
└─ tests/                        # pytest：几何、特征、异常、统计、缓存、报告
```

设计原则：

- **配置即实验记录**：所有会改变数据的参数都在 `Config` 中，并进入缓存指纹；运行目录保存实际使用的 `config.json`。
- **生成与评估分离**：`geometry` 只负责几何与异常，`data` 负责数据集，`models` 只实现 `fit/predict_scores`，`experiments` 只负责编排。
- **可扩展点收敛**：新增曲面继承 `Surface`；新增模型实现 `PointClassifier` 并注册；新增实验实现 `Experiment` 并注册。

---

## 2. 安装

要求 Python ≥ 3.10。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e .
# 可选：PointNet（exp3）与 XGBoost 后端
pip install -e ".[deep,xgb]"
# 开发/测试
pip install -e ".[dev]"
```

- 不安装 PyTorch 时，`exp1`、`exp2` 和 GBDT 路径仍可正常运行；涉及 `exp3` 时会给出明确错误。
- 未安装 XGBoost 时，`gbdt_backend="auto"` 自动使用 sklearn 的 `HistGradientBoostingClassifier`。

---

## 3. 运行

### 3.1 查看可运行内容

```bash
flatball list --config configs/quick.json
# 或
python -m flatball list --config configs/quick.json
```

会列出实验、模型可用性、半径梯度以及每个条件的 `R/r/μ0/rR`。

### 3.2 快速冒烟（不是科学结果）

`configs/quick.json` 使用 `N=1024`、100+100 训练样本、1 个 seed，只跑 `exp1`，用于验证环境和代码路径：

```bash
flatball run --config configs/quick.json --experiment exp1 --seeds 0
```

输出类似：

```text
{"run_dir": "results/run_20261026_220000", "experiments": ["exp1"]}
summary: results/run_20261026_220000/summary.md
```

### 3.3 协议规模

`configs/full.json` 对应 `exp.md` 的规模：`N=10240`、1000 训练样本、200 验证样本、200 平面测试 + 每个 R 200 球面测试、5 seeds、`exp1/exp2/exp3`。

```bash
pip install -e ".[deep]"
flatball run --config configs/full.json
```

这是一次计算量和存储量都很可观的运行（全量特征缓存可能达到数十 GB）。建议先跑 `quick.json`，再按实验拆分：

```bash
# 只跑 H1/H2
flatball run --config configs/full.json --experiment exp1

# 只跑 H3（16 个条件）
flatball run --config configs/full.json --experiment exp2

# 只跑 H4（需要 PyTorch）
flatball run --config configs/full.json --experiment exp3 --seeds 0 1 2 3 4
```

### 3.4 从已有结果重建汇总

如果运行中断但 `metrics.csv` 已写入，可以只重建汇总：

```bash
flatball aggregate --run-dir results/run_20261026_220000
```

### 3.5 测试

```bash
pytest
```

---

## 4. 输出

每次运行创建独立目录：

```text
results/run_<timestamp>/
├─ config.json       # 本次实际使用的完整配置
├─ metrics.csv       # 长表：experiment × seed × condition × model × domain
├─ summary.json      # 机器可读的聚合、配对差异、趋势、判定标记
├─ summary.md        # 人读表格
└─ run.log           # 运行日志
```

`metrics.csv` 每行一个条件/模型/域/seed，字段包括：

- `auc`, `fnr`, `fpr`, `threshold`（阈值在**平面验证集**上按 Youden 指数选择）
- `anomaly_fraction`, `mean_regions`
- `radius`, `r`, `mu0`, `sign`, `amplitude_mode`, `radius_mode`
- `n_points`

`summary.json` 中的 `verdicts` 是**按预设判据计算的标记**，不是自动论文结论；判据同时写入摘要，便于复核。

特征缓存默认在 `cache/<fingerprint>/<condition>/seed_XXXX/`。更改几何、特征、异常或划分参数会自动失效；`--no-cache` 可禁用。缓存保存特征、标签与样本元数据，不保存原始点云。

---

## 5. 配置说明

配置为 JSON（也接受 TOML，需 Python ≥ 3.11）。所有字段都有默认值，但建议显式写出。

### `geometry`

| 字段 | 含义 |
|---|---|
| `n_points` | 每个点云的点数；协议为 10240 |
| `fps_pool_factor` | 球面先采 Fibonacci 池，再做 FPS 的比例；1.0 表示直接用 Fibonacci |
| `plane_area_factor` | 平面面积 = `factor * π * R²`；默认 4 表示与球面面积相等、密度匹配 |

### `features`

| 字段 | 含义 |
|---|---|
| `neighborhood_ratio` | 邻域测地半径 = `ratio * R`；两个域一致 |
| `min_neighbors` | 边界点不足时的最小邻居数 |
| `neighbor_slack` | 由期望邻居数放大得到 kNN 查询的 k |
| `pca_ridge` | 二次拟合的正则项 |
| `chunk_size` | 分块大小，控制内存 |

### `anomaly`

| 字段 | 含义 |
|---|---|
| `mu0` | 法向位移幅度（绝对值）；符号由实验条件决定 |
| `r_ratio` | 异常半径；`radius_mode="relative"` 时 `r = r_ratio * R` |
| `radius_mode` | `relative`（默认，自相似）或 `fixed`（`r = r_ratio * R_ref`，受局部性校验） |
| `sigma_ratio` | `σ = sigma_ratio * |μ0|` |
| `target_fraction` | 异常点占比目标区间，默认 `[0.08, 0.12]` |
| `max_regions` | 每个样本的区域预算；**占比通过区域数量控制，不通过 r 控制** |
| `allow_partial` | 无法达到目标占比时是否放行；默认 `false`（直接报错） |

### `splits`

`train_normal/train_anomaly/val_normal/val_anomaly/test_normal/test_anomaly`。默认：

- 训练 500 正常 + 500 异常；
- 验证 100 + 100（仅平面，用于早停/阈值）；
- 每个测试域 100 正常 + 100 异常。

### `model`

- `gbdt_backend`：`auto | sklearn | xgboost`
- `gbdt_max_iter`、`gbdt_learning_rate`、`gbdt_max_leaf_nodes`
- `pointnet_epochs`、`pointnet_batch_samples`、`pointnet_lr`、`pointnet_patience`
- `device`：`cpu` 或 `cuda`

这些参数被视为固定的“测量仪器”设置，代码不做超参数搜索。

### `evaluation`

| 字段 | 含义 |
|---|---|
| `confidence` | 配对置信区间水平 |
| `equivalence_margin` | 等价/非劣性边界 δ，默认 0.05 |
| `flat_tolerance` | H3 中“R 梯度平坦”的描述阈值 |
| `alpha` | 显著性水平 |
| `threshold` | `youden` 或 `fixed`；阈值只在平面验证集上选择 |

### `experiment`

- `seeds`、`radii`
- `reference_radius`：相对幅度的参考半径，默认 50
- `amplitude_power`：`relative` 幅度模式下 `μ0 ∝ (r / r_ref)^power`
- `max_locality_ratio`：`r/R` 上限，默认 0.2；违反时报错
- `experiments`：要运行的实验列表
- `output_dir`、`cache_dir`、`cache`、`log_level`

---

## 6. 实现与 `exp.md` 的对应关系

### 6.1 两个域严格同构

- 球面：Fibonacci 球面 + FPS 到 `N`，半径 `R`。
- 平面：面积 `4πR²` 的正方形块，`N` 点，点密度与同尺度球面一致。
- 异常：同一个公式、同一 `μ0`、同一 `r`、同一 `σ`、同一法向位移语义；平面用 `(0,0,1)`，球面用径向法向。
- 特征：两个域使用同一 `FeatureExtractor`、同一测地邻域半径比例；不使用真值法向、不使用全局坐标。

### 6.2 异常占比与区域数量

这是实现中最需要注意的几何约束：

- `r/R = 0.1` 时，一个球冠只覆盖约 `0.25%` 表面积；
- 因此要达到 8%–12% 异常点，需要约 40 个区域，而不是 1–3 个；
- 若坚持 1–3 个区域且保持 `r/R ≤ 0.15`，占比无法达到 8%（除非改变 `r`，而那会破坏局部性控制）。

因此代码把**区域数量**作为达到目标占比的调节量（默认上限 96），而 `r` 始终由 `r_ratio` 决定。若目标占比无法达到，默认抛错而不是静默生成低占比样本；`allow_partial=true` 可显式放行，但实际占比会写入 `metrics.csv`/`summary`。

### 6.3 曲率与特征

共享的 PCA 邻域估计器：

1. 在测地半径内取邻居（球面用弦长阈值换算）；
2. 加权 PCA 得到法向与切平面；
3. 在切平面内拟合二次高度函数 `z = a u² + b uv + c v² + d u + e v + f`；
4. 用高斯曲率 `K = (4ac − b²) / (1 + d² + e²)² / h²` 作为主曲率标量。

选择高斯曲率的原因：其符号与 PCA 法向的任意朝向无关，且球面基线为 `1/R²`，与 `exp.md` 中 H3 的推导一致。

6 个点级特征（顺序固定）：

1. `|K|`
2. `K − median(K_neighborhood)`
3. 上述偏差的符号
4. 点法向与邻域平均法向的夹角
5. 法向沿切向梯度的 Frobenius 模
6. 邻域曲率方差

模型 A/B 都只看到这 6 维特征。坐标、球心、半径、物体身份均不可见。

### 6.4 模型

- **模型 A**：GBDT 点级二分类；训练时按类别频率加权；固定默认参数。XGBoost 已安装时优先使用，否则 sklearn `HistGradientBoostingClassifier`。
- **模型 B**：两层共享 MLP（6→64→128）+ 全局 max pooling + 逐点分类头；Adam `lr=1e-3`，按平面验证集 AUROC 早停。输入仍是 6 维手工特征，不是原始坐标。

### 6.5 条件与假设

- **exp1**：`μ0 > 0`、`r = 0.1R`、绝对幅度；训练平面，测试平面与 `R ∈ {50,20,10,5}`。每个 `R` 使用匹配的平面基线（相同 `r`、相同点密度、各自独立采样），因此配对差异是“同尺度球面 − 同尺度平面”。
- **exp2**：2×2 设计，`sign ∈ {+1,−1}` × `amplitude_mode ∈ {absolute, relative}`，每个单元重复半径梯度。
  - `absolute`：`μ0` 固定；
  - `relative`：`μ0 ∝ (r/r_ref)^amplitude_power`。
- **exp3**：exp1 的数据与划分，模型 A vs 模型 B；比较的是跨域 AUROC 下降率及其配对差异。

### 6.6 统计口径

- 所有不确定性都在 **seed 之间**计算，不把点当作独立样本。
- 配对差异：`d_s = AUROC_sphere,s − AUROC_plane,s`，报告均值、SD、配对 t 的 95% CI。
- H1：不报“p≥0.05 即支持”；报告 CI 下界是否大于 `−δ`（默认 `δ=0.05`），并同时给出非劣性单侧 p 和 TOST p。
- H2：每个 seed 在半径梯度上计算 `Spearman(AUROC, 1/R²)`，再对 seed 间 rho 做单侧检验；摘要同时给出逐 seed 值。
- FNR/FPR：阈值来自平面验证集；测试时分别报告平面与球面，便于检查 H3b 预测的“失败以假阴性为主”模式。
- exp3 的 H4 判定使用**绝对 AUROC 下降**的 B−A 配对差异；`summary.json` 同时给出相对下降率 `(plane−sphere)/plane` 及其差异，便于两种口径复核。

`summary.json` 中的 `verdicts`：

- `H1.supported`：每个 R 的配对 CI 下界都 `> −margin`；
- `H2.supported`：平均 rho `< 0` 且单侧 `p < alpha`；
- `H3a_criterion_met` / `H3b_criterion_met`：描述性标记，分别对应“绝对模式 R 梯度平坦”与“相对模式随曲率单调下降且 Spearman 显著”；
- `H4_revised_supported`：模型 B 的下降不显著更小（B−A 的 CI 上界 `≥ 0`）。

这些是规则化标记，不替代对效应量和实验设计的解释。

---

## 7. 扩展方式

### 新增曲面

继承 `flatball.geometry.surfaces.Surface`，实现：

```python
geodesic_distances(points, centers)
chord_radius(geodesic_radius)
expected_neighbors(geodesic_radius, n_points=None)
sample_centers(count, margin, rng)
curvature_scale()
```

并在 `DatasetBuilder._surface` 中增加分支。特征、异常注入、评估都不需要改。

### 新增模型

实现 `flatball.models.base.PointClassifier`：

```python
class MyModel(PointClassifier):
    name = "my_model"
    def fit(self, train, val): ...
    def predict_scores(self, split): ...  # (n_samples, n_points)
```

在 `models/__init__.py` 的 `_FACTORIES` 和 `MODEL_NAMES` 注册。

### 新增实验

继承 `Experiment`，实现 `conditions()` 与 `summarize()`，然后在 `experiments/__init__.py` 的 `EXPERIMENTS` 注册。`Experiment.run()` 已处理“条件 → 数据集 → 模型 → 评估记录”的通用流程。

### 新增配置项

在对应 frozen dataclass 中加字段，并在 `__post_init__` 里校验；`Config.from_dict` 会自动接收。若新参数会影响特征/标签，把它加入 `Config.dataset_fingerprint()` 的 payload（目前包含 `geometry/features/anomaly/splits`）。

---

## 8. 已知边界

- `quick.json` 只用于验证代码路径，样本数和 seed 数都不足以支持统计结论。
- 实验测量的是“法向位移异常族”的可检测性迁移，不是真实工业缺陷，也不是缺陷身份的因果识别。
- 未加入随机旋转增强：输入不含坐标，旋转增强不会改变本设计的可检测性；若后续要测原始坐标模型，应另行加入显式旋转对照。
- `exp2` 的 H3a/H3b 标记依赖 `radius_mode` 与 `amplitude_mode` 的组合；请以 `summary.json` 中记录的实际条件和 `config.json` 为准，不要把默认配置的解释外推到其他参数集。
- 全量运行耗时与缓存体积较大；建议先用小 `n_points`/少 seed 验证流程，再逐步放大。
