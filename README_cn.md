# Quick Start

> 目标：用最少步骤在本地跑起推理服务（Docker / 非 Docker 均可），并通过 HTTP API 完成一次端到端预测调用。

## Option A：Run Locally (Python)

1. 创建虚拟环境并安装依赖

<pre class="overflow-visible! px-0!" data-start="313" data-end="450"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python -m venv .venv
</span><span># Windows: .venv\Scripts\activate</span><span>
</span><span># Linux/Mac: source .venv/bin/activate</span><span>
pip install -r requirements.txt
</span></span></code></div></div></pre>

2. 启动服务（不同入口对应不同模型组合/策略）

<pre class="overflow-visible! px-0!" data-start="478" data-end="543"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python run.py
</span><span># 或</span><span>
python run_2.py
</span><span># 或</span><span>
python run_3.py
</span></span></code></div></div></pre>

默认会启动一个本地 HTTP 服务（端口以 `config.py` 为准），提供 `/predict` 推理接口。

---

## Option B：Run with Docker

1. 构建镜像

<pre class="overflow-visible! px-0!" data-start="647" data-end="705"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>docker build -t stroke-predict-engine:latest .
</span></span></code></div></div></pre>

2. 启动容器并映射端口（把 `<PORT>` 替换成你的服务端口，比如 5000/8000）

<pre class="overflow-visible! px-0!" data-start="756" data-end="829"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>docker run --</span><span>rm</span><span> -p <PORT>:<PORT> stroke-predict-engine:latest
</span></span></code></div></div></pre>

---

# API Example

> 推理服务通过 REST API 接收病人特征 JSON，返回 stroke 预测结果（0/1）及可选概率。

## Request (POST /predict)

（字段名以你的服务实现为准；下方为典型示例，覆盖你 README 中提到的特征）

<pre class="overflow-visible! px-0!" data-start="978" data-end="1404"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>curl -X POST http://localhost:<PORT>/predict \
  -H </span><span>"Content-Type: application/json"</span><span> \
  -d '{
    "age": 67,
    "race": 1,
    "sex": 1,
    "sbp": 160,
    "dbp": 95,
    "blood_sugar": 140,
    "htn": 1,
    "dm": 0,
    "hld": 1,
    "smoking": 0,
    "hx_of_stroke": 0,
    "hx_of_afib": 0,
    "hx_of_psych_illness": 0,
    "hx_of_esrd": 0,
    "hx_of_seizure": 0,
    "nihss": 3,
    "facial_droop": 1
  }'
</span></span></code></div></div></pre>

## Response (Example)

<pre class="overflow-visible! px-0!" data-start="1429" data-end="1601"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"prediction"</span><span>:</span><span></span><span>1</span><span>,</span><span>
  </span><span>"probability"</span><span>:</span><span></span><span>0.87</span><span>,</span><span>
  </span><span>"meta"</span><span>:</span><span></span><span>{</span><span>
    </span><span>"model"</span><span>:</span><span></span><span>"xgb | rf | nn | voting"</span><span>,</span><span>
    </span><span>"cluster_id"</span><span>:</span><span></span><span>"optional"</span><span>,</span><span>
    </span><span>"is_boundary"</span><span>:</span><span></span><span>"optional"</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

# Project Structure

> 该项目结构同时支持“实验训练与分析”和“推理部署与服务化”，两条线相互独立但共享统一的特征处理与模型产物。

<pre class="overflow-visible! px-0!" data-start="1686" data-end="2509"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>.
├── Dataset - FABS (init).xlsx
├── data_clean.xlsx / data_clean_2.xlsx / data_clean_3.xlsx
├── models/
│   ├── RF_model_for_stroke_prediction.pkl
│   ├── XGB_model_for_stroke_prediction.json
│   └── ML_model_for_stroke_prediction.h5
├── featureset_multimodels/
│   ├── trained_models/
│   └── model_trainer_with_reports*.py
├── model_feature_analysis/
│   └── feature_importance_analyzer.py
├── nearest_neighbor_analyzer/
│   └── nearest_neighbor_analyzer.py
├── processing/
│   ├── interval_divider.py
│   ├── grid_processor.py
│   ├── grid_merger.py
│   ├── gravitational_kmeans.py
│   └── voting_models_1.py
├── app/
│   ├── routes/
│   ├── services/
│   ├── models/
│   ├── templates/
│   ├── static/
│   └── config.py
├── run.py / run_2.py / run_3.py
├── requirements.txt
├── Dockerfile
└── .dockerignore
</span></span></code></div></div></pre>

---

# 0. 项目目标与业务背景（Ambulance / Pre-hospital Stroke Triage）

**目标** ：做一个能部署到救护车（或边缘设备/车载电脑）上的轻量 AI 推理服务：

* 输入：救护车上**易获取**的病人特征（比如年龄、性别、种族、血压 SBP/DBP、血糖等）
* 输出：病人是否 stroke 的预测概率 / 分类（0/1）
* 用途： **现场分诊** 、提前通知医院、优先级排序、资源调度

你这句话很关键：

> “features 都是救护车上比较容易得到的 feature”
>
> 这决定了整个项目不是追求极限精度，而是追求 **可部署、可实时、可解释、可稳定** 。

---

# 1. 数据集与问题定义（FAB / FABS Stroke Dataset）

### 1.1 数据概况

* 数据：FAB / FABS stroke dataset（目录里也有 `Dataset - FABS (init).xlsx`）
* 样本量：约 **800**
* 特征数：约 **15 个 feature + 1 个 label**
* label：0/1（你这里的定义是：0=non-stroke，1=stroke）

> 你提到“不是 balance 的，majority 是 1，minority 是 0”。
>
> 这个会直接影响：accuracy 可能虚高，但 precision/recall/ROC/AUC 可能不理想，尤其在 minority 类上波动明显。

### 1.2 典型特征（你提到的）

* demographic：age, race, sex
* vitals：SBP, DBP
* labs：blood sugar（救护车也能测）
* 以及你之前一直强调的：**可在 pre-hospital 获取**的特征集合

---

# 2. 数据清洗与 Feature Engineering（从 Excel 到“可训练表”）

你的目录里有多份清洗版本：

`data_clean.xlsx / data_clean_2.xlsx / data_clean_3.xlsx` ——这很符合“多轮清洗 + 迭代试验”的工程现实。

这一阶段核心动作可以总结为：

### 2.1 清洗（Data Cleaning）

* 去除行列空白、异常字符、不可解析单元格
* 统一类别编码（race/sex 等）
* 缺失值处理（删行/填补/保留缺失标记——按你当时策略）
* 移除明显无关列 / ID 列 / 备注列等

### 2.2 特征选择（Feature Selection）

* 保留对 stroke 预测最有用且**现场可拿到**的列（SBP/DBP 等）
* 这个过程通常会结合：
  * 先验医学常识（哪些指标与 stroke 相关）
  * 模型驱动的重要性分析（你目录里有 `model_feature_analysis/feature_importance_analyzer.py`）

---

# 3. Baseline：三模型直接训练（RF / XGBoost / Neural Network）

你选择三种模型是很合理的“互补组合”：

* **Random Forest (RF)** ：稳、对噪声鲁棒、对小数据友好、解释性更容易（feature importance）
* **XGBoost (XGB)** ：非线性强、泛化强、常在 tabular 任务上很强
* **Neural Network (NN)** ：表达能力强，但对数据量/特征工程/正则化更敏感

目录里也能看到三种产物落地：

* `RF_model_for_stroke_prediction.pkl`
* `XGB_model_for_stroke_prediction.json`
* `ML_model_for_stroke_prediction.h5`

### 3.1 训练与评估指标（你提到的 + 工程上必备）

你当时看的 metrics 很可能包括（你也允许我自由补充）：

* Accuracy（整体对不对）
* Precision（预测为 stroke 的里面有多少是真的 stroke）
* Recall / Sensitivity（真正 stroke 的有多少被抓住）
* F1（precision/recall 平衡）
* ROC-AUC（阈值无关的区分能力）
* Confusion Matrix（TP/FP/TN/FN）
* 可能还会有：PR-AUC（在类别不平衡时更敏感）

### 3.2 Baseline 痛点（你提到的结论）

> “尤其 accuracy 和 precision 表现不是很好”

这在不平衡数据 + 特征可分性不足时很常见，原因通常是：

* 类别不平衡导致模型倾向 majority（你这里 majority 是 1）
* feature space 的分布可能是“多团块 + 局部可分”，而不是一个全局统一边界
* 全局训练会把本来分得开的局部结构“搅浑”

这就自然引到你第 4 步的思路： **先分布结构化，再训练** 。

---

# 4. 关键创新：Grid + Clustering + 边界扩展 + 分区训练（把 tabular 做成“空间结构”）

这一部分是项目的核心亮点，你的目录结构也直接对应：

* `processing/grid_processor.py`
* `processing/grid_merger.py`
* `processing/interval_divider.py`
* `processing/gravitational_kmeans.py`
* `processing/voting_models_1.py`
* 以及 `nearest_neighbor_analyzer/nearest_neighbor_analyzer.py`（用于边界点/邻域分析很合理）

我按你描述，把算法逻辑“讲清楚”：

## 4.1 你当时的假设（非常重要）

1. **相似样本会聚在一起** （局部密度高）
2. 多数情况下分布更接近“凸/球形团块”，而不是复杂凹形（这会让 k-means 类方法更可用）
3. 通过把空间切成 grid，可以先粗定位 cluster 的数量与初始中心（相当于“自动找 k 与初始中心”）

## 4.2 Grid 找初始 cluster（“先粗后细”的结构发现）

你做的不是直接 k-means，而是：

* 先对连续变量做  **interval 划分** （对应 `interval_divider.py`）
* 形成多维 grid cell（对应 `grid_processor.py`）
* 统计每个 cell 的样本密度、label 比例（0/1 分布）
* 基于一组 grid 条件（你说“反复实验得到”）筛出：
  * 高密度的候选中心 cell
  * 候选 cluster 个数与位置（初始中心）

这一步的价值是：

**把 k-means 最难的“k 和初始中心”从拍脑袋，变成数据驱动。**

## 4.3 边界扩展（Boundary Growing / Region Expansion）

你接着做了“从中心往外扩”的过程：

* 初始化 cluster = 若干中心 cell
* 然后 **逐步扩大边界** （比如扩大到邻近 cell、扩大某个半径/阈值）
* 每扩大一步：
  * cluster 内样本变化
  * cluster 外残余样本变化（边界点/噪声点）

**补充：boundary 附近点的距离计算与归属判断（类似 k-means assignment）**

在每一次边界扩展过程中，你会对“靠近边界、尚未稳定归属”的样本点做一次类似 k-means 的分配步骤：

1. **定义距离度量** （通常是特征空间距离；可理解为 k-means 的欧式距离/加权距离）
2. 对每个 boundary 附近点 `x`，计算它到各个 cluster 当前中心（或代表点）的距离：
   * `d(x, c1), d(x, c2), ..., d(x, ck)`
3. **归属判断** ：

* 若最小距离对应的 cluster 满足本轮扩展阈值（比如距离 < r，或落入邻近 cell 集合），则把该点吸收到对应 cluster
* 否则该点继续保留在  **boundary / remaining set** （作为“模糊区域/未归类点”）

这样做的目的，是把“边界扩展”从纯几何扩圈，变成 **“按距离竞争归属、逐步稳定分区”** 的过程；边界点不会被硬塞进某个 cluster，而是经过距离与阈值规则反复确认。

你每次扩展都会做很工程化的评估闭环：

### 4.4 每一步都“重新训练 + 分区评估”

你不是只看聚类好不好，而是看它 **是否提升下游 ML** ：

* 对每个 0/1 cluster（或每个 cluster 内部）训练/评估模型表现
* 对剩余边界点（未归类/模糊区域）单独训练/评估
* 最后把这些分区模型组合起来看整体指标

更具体地说：每一轮扩展完成一次“距离分配 + cluster 更新”后，你都会重新得到：

* 本轮的 clusters（每个 cluster 的样本集合）
* 本轮的 remaining / boundary set（仍然难以归属的点）

然后你会对它们做一次完整的训练-评估：

* **cluster 内部模型表现** （可能按 cluster 分别训练/评估）
* **boundary / remaining 的模型表现** （单独训练/评估或作为 fallback）
* **合并后的整体表现** （最终以整体指标作对比）

这通常意味着你实现了类似两种结构（任选其一或混合）：

**结构 A：Cluster-specific models**

* 每个 cluster 一个模型（RF/XGB/NN 或固定一种）
* 推理时：先判断样本落在哪个 cluster → 调用对应模型

**结构 B：Cluster model + Boundary model**

* cluster 内用 cluster-model
* cluster 外/边界用一个 fallback model（比如全局 XGB 或 NN）

你还提到了 `voting_models_1.py`，这说明你可能还有：

**结构 C：Ensemble voting**

* RF / XGB / NN 在某些区域投票或加权融合
* 或者 cluster 内部用投票提升稳定性

## 4.5 阈值选择（你说的“找到最好的 threshold”）

你最终的“threshold”可以理解为：

* grid 的划分粒度 / 密度阈值 / 边界扩展半径 / cluster 合并阈值

  这些超参数共同决定最终的 cluster 分配。

你用“效果最好的阈值”作为最终方案，结果：

* Accuracy 从 **~88% 提升到 92–93%**

这一点非常像你在做一种“结构化分布对齐”：

> 先把数据从一个全局混杂空间，拆成局部更可分的区域，再在区域上训练模型 → 指标自然上升。

**补充：在选定最终 threshold 后，并不是直接定稿，而是做三路“结构增强”对比（threshold / gravity-kmeans / credit-weighted）**

当你通过边界扩展与分区训练选出了一个候选的最佳 threshold（得到 clusters + boundary set）后，你会进一步做“是否还能更好”的探索，并把同一份数据在同一阈值基础上跑三套方案做对比：

### 方案 A：Threshold-based clustering（原始阈值方案）

* 输入：threshold 下得到的 clusters + boundary set
* 做法：按当前 clusters 分区训练 + boundary fallback（并可配合 voting）
* 输出：整体 ML 指标（Accuracy/Precision/ROC-AUC 等）

### 方案 B：Threshold + Gravity-KMeans（在阈值基础上重新聚类）

* 输入：方案 A 的 clusters（以及可能的中心/代表点）
* 做法：引入 **gravity-kmeans** 思路（对应 `processing/gravitational_kmeans.py`）
  * 直观理解：cluster 的“吸引力”不只取决于几何距离，还会结合局部密度/样本分布，使中心更新与归属分配更稳定。
* 输出：新的 clusters + 新的 boundary/remaining set，再做同样的分区训练与整体评估

### 方案 C：Threshold + Gravity-KMeans + Credit 加成（对 boundary 点做 label-driven credit）

* 输入：方案 B 得到的 clusters + boundary/remaining set
* 做法：对 boundary 附近点引入一个 **credit（信用/加成权重）** 概念：
  * boundary 点会根据其原始 label（0/1）或其风险方向，赋予不同的 credit 大小
  * credit 会影响该点被某个 cluster 吸收的“有效距离/吸引力”（等价于：同样距离下，有些点更容易被纳入，或更难被错误吸收）
* 输出：credit-weighted 的重新聚类结果 + boundary set，再做同样的分区训练与整体评估

### 三者统一比较与最终选型

你最终会把三套方案的结果放在一起比较：

* A：threshold clustering（clusters + boundary）
* B：gravity-kmeans re-clustering（clusters + boundary）
* C：credit-weighted gravity-kmeans clustering（clusters + boundary）

比较维度包括：

* 整体指标（Accuracy / Precision / Recall / ROC-AUC / PR-AUC）
* minority/难例区域的稳定性（尤其 boundary 部分的误判是否下降）
* 分区后模型是否更稳（避免某个 cluster 被“坏边界点”污染）

最终选择 **效果最好的一套** 作为最后的 clustering + 分区建模方案。

---

# 5. 模型产物固化（Artifacts）与目录对应关系

你的目录里这部分很清晰：

* `models/`：三模型文件（.pkl / .json / .h5）
* `featureset_multimodels/trained_models/`：可能是不同 feature set 或不同 threshold 的模型版本
* `featureset_multimodels/model_trainer_with_reports*.py`：多轮训练 + 报告输出
* `model_feature_analysis/`：特征重要性与性能分析脚本
* `nearest_neighbor_analyzer/`：用于边界点/邻域关系的分析与调参

这套结构本质上已经是：

**实验层（训练+分析）** 和 **部署层（模型文件+API）** 分离了。

---

# 6. 服务化与 Docker 化（把模型变成可访问的推理服务）

你项目根目录有：

* `app/`（典型的服务目录）
  * `routes/`：API 路由（例如 `/predict`）
  * `services/`：推理服务（加载模型、特征处理、调用预测）
  * `models/model.py`：模型加载/封装
  * `templates/ static/`：如果当时做过简单网页 demo 也合理
  * `config.py`：端口/模型路径/环境变量
* `run.py / run_2.py / run_3.py`：不同启动方式/不同模型组合的启动入口
* `requirements.txt`：依赖
* `Dockerfile / .dockerignore`：容器化

服务化的典型推理流程是：

1. API 接收 JSON 输入（病人特征）
2. 校验字段、类型转换、缺失处理
3. 做同样的 preprocessing（必须与训练一致：编码/标准化/interval/grid 等）
4. 决策：样本属于哪个 cluster / 是否边界点
5. 调用对应模型（或投票融合）
6. 返回预测结果（class + probability + 可选解释字段）

---

# 7. Azure 部署全流程（你要我“回忆并补全”的部分）

你说的“本地 docker 容器化 → 弄到 azure 挂起来 → 外部访问”，最标准、也最符合你当时做法的链路一般是：

## 7.1 本地构建镜像

* 写好 `Dockerfile`
* `docker build` 生成本地镜像（暴露端口，比如 5000/8000）

## 7.2 推送到 Azure Container Registry (ACR)

典型步骤（概念层面，不写死命令细节以免与你当时命名不一致）：

1. 在 Azure 创建 ACR（有一个 registry 名字，比如 `xxxacr`）
2. 本地登录 ACR
3. 给镜像打 tag：`<acr_login_server>/<image_name>:<tag>`
4. `docker push` 推到 ACR

> 你在描述里提到 “containersintance 和 registration”，基本就是  **ACR（注册表）+ ACI（实例）** 。

## 7.3 用 Azure Container Instance (ACI) 启动容器并开放端口

* 从 ACR 拉取镜像
* 指定：
  * CPU / Memory（小模型推理一般够用）
  * 环境变量（如果你用 `config.py`/`.env`）
  * 端口暴露（对外开放 HTTP）
  * DNS name label（生成公网 FQDN）

部署完会得到一个公网访问地址（FQDN:port）。

## 7.4 外部访问测试（Postman）

你当时最后一步就是：

* Postman 构造 JSON body（包含 age/race/sex/SBP/DBP/...）
* POST 到 `http://<aci_fqdn>:<port>/predict`
* 返回：
  * prediction: 0/1
  * probability（如果输出了）
  * （可选）模型类型 / cluster id / voting 结果

---

# 8. 把“最核心的算法 + 最核心的工程流程”浓缩成一条主线

如果你要对外讲（GitHub README / 面试 / 汇报），建议用这一条主线：

1. **Pre-hospital Stroke Prediction** ：面向救护车的实时预测
2. **Clean & Select Features** ：只保留现场可获得且有效的特征
3. **Baseline Models** ：RF / XGB / NN 全局训练评估（accuracy/precision/ROC-AUC/CM）
4. **Structure-aware Improvement（核心贡献）** ：

* 用 **grid** 自动发现 cluster 数量与中心
* 用  **边界扩展 + 距离归属判断** （类似 k-means assignment）稳定 cluster 与 boundary
* 在选定阈值后，进一步做  **三路对比** ：
  * threshold clustering
  * threshold + gravity-kmeans
  * threshold + gravity-kmeans + credit-weighted boundary clustering
* 选最优方案，指标从 ~88% 提升到 92–93%

5. **Deployment-ready** ：模型固化为 pkl/json/h5 → 封装 REST API → Docker
6. **Azure Production-like** ：推到 ACR → ACI 拉起服务 → Postman 端到端验证

---

# 9. Azure 云端部署与端到端验证（基于真实资源与截图）

本项目在完成模型训练与 Docker 容器化之后，进一步在 **Microsoft Azure** 上完成了真实的云端部署与端到端验证。以下内容  **全部基于 Azure Portal 中的实际资源与截图信息** ，用于证明该系统不仅停留在本地实验阶段，而是完成了完整的云端推理闭环。

---

## 9.1 Azure 资源组织方式（Resource Group 级别）

项目所有云资源统一放置在同一个 Resource Group 中：

* **Resource Group** ：`StrokePredictionResourceGroup`
* Tag：
  * `StrokePredictionTag : StrokePredictionValue`
* Deployments：**16 Succeeded**

这表明该项目在 Azure 上并非一次性部署，而是经历了多轮创建、调整与验证。

---

## 9.2 Azure Container Registry（ACR）：模型推理镜像管理

为统一管理推理服务的 Docker 镜像，项目创建并使用了专用的 Azure Container Registry：

* **ACR 名称** ：`strokepredictregistry`
* **Login Server** ：

<pre class="overflow-visible! px-0!" data-start="11868" data-end="11912"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>strokepredictregistry.azurecr.io
</span></span></code></div></div></pre>

* **Region** ：East US
* **Pricing Tier** ：Basic
* **Creation Time** ：2024-11-24 11:04 PM (EST)

推理服务镜像以如下形式存储于 ACR 中：

<pre class="overflow-visible! px-0!" data-start="12029" data-end="12102"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>strokepredictregistry.azurecr.io/stroke-predict-engine:latest
</span></span></code></div></div></pre>

该镜像被后续多个 Azure 计算资源复用，作为统一的模型推理 artifact。

---

## 9.3 Azure Container Instance（ACI）：直接容器化推理服务

项目使用 Azure Container Instance 直接从 ACR 拉起推理容器，用于快速验证云端推理能力：

* **Container Instance 名称** ：`strokepredictcontainer`
* **Region** ：East US
* **OS** ：Linux
* **Container Count** ：1
* **Current State** ：Stopped

  （用于验证完成后停止，以控制成本）
* **公网 FQDN（已分配）** ：

<pre class="overflow-visible! px-0!" data-start="12443" data-end="12515"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>strokepredictengine-dzhbfh3epg2dhgu.eastus.azurecontainer.io
</span></span></code></div></div></pre>

该 ACI 实例具备完整的 CPU / Memory / Network 使用指标，说明容器曾实际运行并对外提供服务，而非仅创建未使用的空资源。

---

## 9.4 Azure App Service（Container-based）：另一种托管方式验证

除 ACI 外，项目还使用 **Azure App Service for Containers** 对同一推理镜像进行了部署验证，用于对比不同 Azure 托管模式：

* **App Service 名称** ：`strokepredictengine`
* **OS** ：Linux
* **Region** ：Canada Central
* **App Service Plan** ：`ASP-StrokePredictionResourceGroup-91e0`
* Tier：Basic (B1)
* **Publishing Model** ：Container
* **Container Image Source** ：Azure Container Registry
* **Image** ：

<pre class="overflow-visible! px-0!" data-start="13005" data-end="13078"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>strokepredictregistry.azurecr.io/stroke-predict-engine:latest
</span></span></code></div></div></pre>

该配置表明推理服务不仅可以以“裸容器实例”的形式运行，也可以通过 App Service 进行 Web/API 托管，体现了对不同云端部署形态的工程探索。

---

## 9.5 多 Region 部署特征（工程实验痕迹）

从 Azure 资源信息可以看出：

* **ACR / ACI** ：East US
* **App Service / App Service Plan** ：Canada Central

这说明该项目在实际部署过程中，曾基于资源可用性、实验目的或成本因素，在不同 Azure Region 上进行过部署尝试，而非严格绑定单一区域。

---

## 9.6 云端端到端推理验证（Postman）

在云端容器成功运行后，项目通过 **Postman** 对推理服务进行了端到端验证：

* 使用公网 FQDN + 暴露端口访问推理接口
* 请求方式：HTTP POST
* 输入：包含以下字段的 JSON（示例）
  * age
  * race
  * sex
  * SBP
  * DBP
  * blood sugar
  * 等救护车可获取特征
* 输出：
  * stroke prediction（0 / 1）
  * probability（若启用）
  * （可选）cluster / voting 相关信息

该过程验证了从 **云端容器 → 模型加载 → 特征处理 → 推理返回** 的完整链路。

---

## 9.7 Azure 订阅与资源背景

* **Subscription** ：Microsoft Azure Sponsorship
* Subscription 状态：Active
* 资源统计（在该订阅下）：
  * Container Registries：1
  * Container Instances：1
  * App Services：1
  * App Service Plans：1

这进一步说明该项目是在真实 Azure 订阅环境中完成的完整工程实验。

---

## 9.8 本节小结

通过 Azure Portal 中的真实资源与运行痕迹可以确认：

* 模型推理服务已完成 **Docker 化**
* 镜像已推送至 **Azure Container Registry**
* 推理服务已在 **Azure Container Instance** 中成功运行并对外暴露
* 同一镜像也通过 **Azure App Service（Container-based）** 完成部署验证
* 外部客户端（Postman）成功完成端到端调用

因此，该项目不仅是算法与本地工程实验，而是一个  **完成了真实云端部署与验证的端到端 Stroke Prediction 系统原型** 。

---

# 10. Feature Importance 与 Feature 组合分析（基于模型、SHAP 与可部署性约束）

在完成基础模型训练与结构化 clustering 之前，本项目对 **feature 的重要性及其组合效果**进行了系统分析。该分析并非依赖单一方法，而是结合  **多模型特征重要性、SHAP 可解释性分析、特征消融实验以及实际部署可获取性约束** ，逐步筛选并确认核心特征集合。

---

## 10.1 Feature 输入定义与工程约束（来自真实业务场景）

在进行特征重要性分析前，项目首先明确了  **用户输入规范与现实约束** ：

* 年龄（Age）：1–150
* 种族（Race）：1=C，2=AA，3=Other（实际使用中统一按 Other 处理）
* 性别（Sex）：1=Male，2=Female
* 二值病史特征（0/1）：
  * HTN, DM, HLD, Smoking
  * HxOfStroke, HxOfAfib, HxOfPsychIllness
  * HxOfESRD, HxOfSeizure
  * FacialDroop (weakness)
* 连续生理指标：
  * SBP / DBP（mmHg）
  * BloodSugar（mg/dL）
* NIHSS（点数）

同时，在真实 workflow 讨论中明确指出：

* **NIHSS 与 FacialDroop** 往往来自 **teleneurologist 的离线评估**
* 在救护车现场 **并非总是实时可得**
* 因此必须评估：
  > *“如果移除这些特征，对模型性能影响有多大？”*
  >

这直接引出了后续的  **两套 feature set 对比实验** 。

---

## 10.2 多模型 Feature Importance 分析（DT / RF / LightGBM）

### 10.2.1 使用的模型与统一超参数设置

为了保证可比性，特征重要性分析在以下模型上进行：

* Decision Tree (DT)
* Random Forest (RF)
* LightGBM (GBM)

统一的关键参数包括：

* `max_depth = 6`
* `min_samples_leaf = 20`
* `n_estimators = 100`（RF / LightGBM）

---

### 10.2.2 跨模型一致性分析（Consistency Check）

在 **完整 feature set** 下，分析结果显示：

* **FacialDroop (weakness)、SBP、Age、BloodSugar**
  * 在 DT、RF、LightGBM 中 **一致性地排名靠前**
* 不同模型的侧重点不同：
  * DT / RF：更强调**个别强特征**
  * LightGBM：importance 分布更均匀，多特征共同贡献

 **关键结论** ：

> 只有在 **多模型中同时重要** 的特征，才被视为“稳定可信”的候选核心特征。

---

## 10.3 Feature Correlation 与冗余关系分析

在 feature importance 之外，项目还通过 **Feature Correlation Heatmap** 分析：

* 检查特征之间的线性/非线性相关性
* 避免引入高度冗余、但解释价值有限的特征组合
* 为后续 clustering 与 grid 划分提供更“干净”的特征空间

该步骤用于辅助判断：

* 哪些特征是“独立风险信号”
* 哪些特征只是其他变量的 proxy

---

## 10.4 SHAP 分析：从“重要”到“可解释”

为了进一步理解特征在模型决策中的实际作用方式，项目引入了  **SHAP（SHapley Additive exPlanations）分析** ，并在 DT / RF / LightGBM 上进行对比。

### 10.4.1 SHAP 分析目标

SHAP 在本项目中的作用并非仅用于可视化，而是用于回答：

* 该特征是  **稳定地推高 / 推低 stroke 风险** ，还是行为混乱？
* 不同模型中，该特征的  **贡献方向是否一致** ？
* 某些特征是否只在极少数样本上起作用？

---

### 10.4.2 SHAP 分布宽度（Distribution Width）解读

通过 SHAP 分布图，可以观察到：

* **SBP / DBP** ：
* SHAP 值分布稳定
* 高值 → 正向贡献（风险升高）
* 低值 → 负向贡献（风险降低）
* 某些特征：
  * SHAP 分布宽、正负混杂
  * 说明其影响不稳定或高度依赖上下文

 **关键结论** ：

> SBP / DBP 不仅“重要”，而且是  **行为一致、医学合理、可解释的风险特征** 。

---

## 10.5 Cross-Validated Feature Importance（稳定性验证）

项目还对 **Random Forest 与 LightGBM** 进行了  **Cross-Validated Feature Importance 对比** ：

* 在不同数据切分下重复训练
* 比较 feature importance 排名是否稳定

结果显示：

* SBP、Age、BloodSugar 等特征在多次验证中保持稳定
* 一些边缘特征排名波动较大，更容易受数据切分影响

这一步进一步增强了对核心特征选择的信心。

---

## 10.6 特征消融实验：移除 NIHSS 与 FacialDroop 的影响评估

基于真实 workflow 约束，项目进行了  **关键特征消融实验** ：

### 10.6.1 实验设计

* 对比两套 feature set：
  1. **完整特征集** （包含 NIHSS + FacialDroop）
  2. **受限特征集** （移除 NIHSS + FacialDroop）
* 在 DT / RF / LightGBM 上分别评估：
  * Accuracy
  * Precision
  * Recall
  * F1
  * AUC

---

### 10.6.2 实验结果与结论

* 移除 NIHSS / FacialDroop 后：
  * 所有模型性能 **均有下降**
  * AUC、Accuracy 明显降低
* 但即便如此：
  * **SBP、DBP、Age、BloodSugar** 仍然是最重要的剩余特征
  * 模型仍保持一定的判别能力

 **关键工程结论** ：

> NIHSS / FacialDroop 是强特征，但在 pre-hospital 场景不可稳定获取；
>
> SBP / DBP 等特征则在“可获取性 + 重要性 + 稳定性”三方面取得最佳平衡。

---

## 10.7 本节小结：Feature 选择的最终决策逻辑

本项目的 feature 选择并非基于单一指标，而是遵循如下决策链路：

1. 多模型 feature importance（DT / RF / LightGBM）
2. SHAP 全局与局部解释（稳定性 + 方向一致性）
3. Feature correlation 与冗余分析
4. Cross-validation 稳定性验证
5. 特征消融与组合对比实验
6. 真实救护车场景下的可获取性与 workflow 约束

最终，**SBP、DBP、Age、BloodSugar 等特征**被确定为：

> **稳定、可解释、可部署、适合结构化 clustering 与云端推理的核心特征集合** 。
