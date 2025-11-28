## 科研项目需求文档 (Context Document)

**项目名称：** 基于Transition Matching (DTM) 的TTS模型推理加速适配
**项目目标：** 将现有的预训练 DiT-TTS 模型适配到 DTM 框架，通过训练轻量级 MLP Head 实现推理加速（降低 NFE），目标是在保持音质的前提下将推理步数从 32 降低至 4-8 步。

#### 1\. 现有模型架构 (Base Model Context)

  * **Backbone:** Diffusion Transformer (DiT)。
      * 配置：22 Layers, 16 Heads, Hidden Size = 1024。
      * 参数量：335.8M。
      * **状态：已在大规模 In-the-wild 数据集上充分训练，权重将完全冻结 (Frozen)。**
  * **输入数据:** Mel Spectrogram (80 bins)。
  * **条件注入:** Cross-Attention (Text) + AdaLN (Timestep/Speaker)。
  * **现有推理:** Euler Solver, 32 Steps, CFG = 2.0。
  * 注意：f5_tts中的scripts、runtime、eval在这一阶段完全无需考虑

### 理论分析
### 核心数学假设 (The Physics)
* **全局路径 (Global Path):** 数据从纯噪声 $X_0$ 到真实数据 $X_T$ 的变化遵循线性插值（Conditional Optimal Transport）：
    $$X_t = (1 - \frac{t}{T})X_0 + \frac{t}{T}X_T$$
    其中 $t$ 是离散的大时间步，$t \in \{0, 1, ..., T\}$。
* **预测目标 (Target):** 模型需要预测从起点到终点的总位移向量：
    $$Y = X_T - X_0$$

### 架构设计：解耦机制 (Decoupling)**
系统分为两部分：
* **Backbone ($f^\theta$, Frozen):** 你的 DiT 模型。
    * **输入:** 全局状态 $X_t$ 和全局时间 $t$。
    * **输出:** 特征序列 $h_t$。
    * **频率:** 低频运行（仅在整数 $t$ 时运行）。
* **Flow Head ($g^\theta$, Trainable):** 一个轻量级 MLP。
    * **输入:** Backbone 特征 $h_t$ + 当前 Flow 状态 $y_s$ + 微观时间 $s$。
    * **输出:** 速度场 $u$（即对 $Y$ 的预测）。
    * **频率:** 高频运行（在 ODE Solver 内部运行）。

### 训练流程 (Algorithm 3 实现要点)**
对于每一个 Batch：
1.  **采样:**
    * 随机采样真实数据 $X_T$ (Mel Spectrogram)。
    * 随机采样噪声 $X_0 \sim \mathcal{N}(0, I)$。
    * 随机采样离散时间步 $t \sim \text{Uniform}(\{1, ..., T-1\})$。
2.  **准备 Backbone 输入:**
    * 计算 $X_t = (1 - \frac{t}{T})X_0 + \frac{t}{T}X_T$。
    * **[Forward 1]**: 将 $X_t$ 输入 **冻结的** Backbone，提取特征 $h_t$。
3.  **准备 Head 输入 (Conditional Flow Matching):**
    * 计算目标 Target $Y = X_T - X_0$。
    * 随机采样微观时间 $s \sim \text{Uniform}([0, 1])$。
    * 随机采样微观噪声 $Y_{noise} \sim \mathcal{N}(0, I)$。
    * 计算 Head 的输入状态 $Y_s = (1-s)Y_{noise} + sY$。
4.  **计算 Loss:**
    * **[Forward 2]**: 将 $h_t, Y_s, s$ 输入可训练的 MLP Head。
    * 得到预测输出 $\hat{v}$。
    * 计算 MSE Loss: $\|\hat{v} - (Y - Y_{noise})\|^2$。
    * *注意：TTS 任务需应用 Padding Mask，忽略 Padding 部分的 Loss。*

### 推理流程 (Algorithm 4 实现要点)**
生成过程如下：
1.  初始化 $X_0 \sim \mathcal{N}(0, I)$。
2.  循环 $t$ 从 $0$ 到 $T-1$：
    * **Backbone Step:** 运行一次 Backbone $f^\theta(X_t, t)$ 得到特征 $h_t$。
    * **Head Step (ODE Solve):** 使用 ODE Solver (如 Euler) 从 $s=0$ 积分到 $s=1$。
        * 求解对象是 $Y$。
        * 初始状态 $Y_{s=0} \sim \mathcal{N}(0, I)$。
        * 动力学方程：$dY_s/ds = g^\theta(Y_s, h_t, s)$。
        * 解出 $Y_{final} = Y_{s=1}$。
    * **Update Step:** 更新全局状态 $X_{t+1} = X_t + \frac{1}{T}Y_{final}$。
3.  返回 $X_T$。

### 特定参数配置 (基于F5-TTS 模型)
* **Backbone Dim:** 1024 (Frozen)。
* **Head Config:**
    * Input Proj: 1024 -> 512。
    * Main Body: 6 Layers, 512 Hidden Dim。
    * Components: AdaLN (用于注入时间 $s$) + 4x FFN Expansion。
* **Data:** Mel Spectrogram (80 dims)。

### 4\. 工程约束 (Constraints)

* **资源限制:** 单卡 RTX 4090 (24GB)。
* **代码风格:** PyTorch。代码必须是**非侵入式 (Non-intrusive)** 的，即不修改原 DiT 代码，而是作为一个独立的 `nn.Module` 挂载。
* **数据流:** 必须支持 Variable Length Sequence (因为是 TTS)，注意 Mask 的处理。
* 如果参数配置与原始F5-TTS代码不符，以原始F5-TTS代码为准