# 2026 人工智能实训 —— 多模态参考人脸图像恢复

> **工程内核**：冻结扩散主干 + 可训练文本分支 / 多模态参考图条件；当前以老照片人像修复与文本条件编辑为落点。
> **开源仓库（2026-11-18 起公开）**：[fty020702/2026-ai-practicum-multimodal-face-restoration](https://github.com/fty020702/2026-ai-practicum-multimodal-face-restoration)。只提交代码、配置、文档与精选成果快照；17 GB 权重 / 2.7 GB 数据 / 1.6 GB 实验产物留在本机。
> **许可**：[`LICENSE`](LICENSE) —— **CC BY-NC-SA 4.0**（署名-非商业性使用-相同方式共享 4.0 国际）。**仅限学术交流 / 学习 / 引用 / 修改，禁止商业使用**；衍生作品须同协议共享并署名。GitHub 侧栏可能显示 "Other"，因为 CC BY-NC-SA 不是 OSI 认证协议，自动识别库不覆盖；以 `LICENSE` 文件内的 SPDX 标识 `CC-BY-NC-SA-4.0` 为准。
> 目录语义、复现路径与成果索引见 [`docs/仓库结构与复现.md`](docs/仓库结构与复现.md)，精选对比图见 `results/figures/`。
> 课程官方要求与本项目达成情况对照见 [`docs/课程目标与达成情况.md`](docs/课程目标与达成情况.md)（依据 `AI系统实训_课程安排.pdf`）。
>
> **引用 / 分享**：请注明项目名「2026 人工智能实训 —— 多模态参考人脸图像恢复」、作者 `fty020702` 与本仓库链接；转发或改编时保留本许可证与负面结论的边界声明（见下）。

把 [DeepSeek 分享对话](https://chat.deepseek.com/share/mkk5xoznfgkuuga1vq) 的最终方案落地成一个**可运行、可验证、可迁移到真实 4B 基座**的工程。

原始设想：已有模型的「文本分支」被参考图特征占据（FDI / Qwen2VL-Flux 式），在此基础上**新增用户文本输入**，冻结扩散主干与图像分支，只训练文本通路。

---

## 状态一览（务必先读）

| 部分 | 状态 | 说明 |
|---|---|---|
| 核心架构（冻结 / 零初始化 / 逐层旁路 / 梯度隔离） | **[RESULT] 已本地跑通** | `src/smoke_test.py` 两种模式 T1–T6 全 PASS |
| toy 训练 + 四组消融 | **[RESULT] 已本地跑通** | checkpoint、指标、对比图在 `outputs/` |
| ChatAI 环境升级 | **[RESULT] 已完成** | `diffusers 0.37.1`，`Flux2KleinPipeline` 可导入，`pip check` 干净 |
| FLUX.2 结构契约（模块名 / 注入位置 / 文本嵌入构造） | **[FACT] 已核实** | `scripts/check_flux2_contract.py` 全 PASS，无需下载权重 |
| FLUX.2 去噪 forward 契约（参考图进图像流 / 三维宽度 / timestep） | **[FACT] 已核实** | `scripts/check_flux2_train_contract.py` 18/18 PASS，见 `docs/方案设计.md` §11–§12 |
| 按 `backend.name` 分发到 toy / FLUX.2 | **[RESULT] 已接通** | `src/system.py`；toy 全部 `[RESULT]` 数值回归后逐位不变 |
| FLUX.2 BF16 权重下载 + 真实权重结构探测 | **[FACT] 已完成** | 21 文件 / 15.98 GB 落盘；`inspect_backbone.py` 实测 runtime dims，`config_reconciliation.ok=true` |
| 真实数据集（老照片修复，对标 CVPR 2020 / InstructIR） | **[RESULT] 已落地** | 真实 DIV2K 照片 + 真实 caption + 真实老照片（无 GT）；见 `docs/真实数据集与论文对标.md` |
| 真实基座 CPU 端到端验证 | **[RESULT] T0–T3 通过；T4=0 定位为单步 warm-up artefact；文本通路在真实 4B 主干上因果连通** | 汇总分级现已随 GPU 训练升级为 **`GPU-TRAINING-VERIFIED`**（见下方两行）。① T0 latent 的 `32 vs 128` 是 `AutoencoderKLFlux2` 的 128 宽 BN 统计与 `latent_channels=32` 两个空间混用，按 diffusers `L898–909` 改成「先反 BN、再 unpatchify」；`test_latent_roundtrip.py` 秒级回归 float32 逆变换 max abs err = **0.0**（带负向对照精确复现原报错）。② 真实 BF16 权重 CPU 跑通 T0 / T1（恒等 maxdiff=0.0）/ T2（梯度零泄漏、loss=0.523438）/ T3（仅文本分支 10 个参数更新）。③ 唯一失败项 T4 `d_text=0.0` **不是架构缺陷**：零初始化的 projector 末层 + 每个 bypass `o_proj` 使第 1 步只有 `o_proj` 有梯度；`verify_text_warmup.py` 用 tiny float32 stub 秒级复现「step1=0 / step2 起>0」。④ `verify_real_text_effect.py` 在**真实 4B 权重**上不做 backward、仅 4 次 forward：零初始化对照 `d_text=0.0`，把文本分支确定性激活成非零（**WEIGHT-PERTURBED, NOT TRAINED**，如实标注）后 `d_text_rms=0.2496` / maxabs=0.8203 / relative=**23.5%** → 用户文本通路因果连通。此处的 `FORWARD-READY` 是**历史阶段结论**，已被下方 GPU 训练结果取代 |
| FLUX.2 GPU 端到端训练 | **[RESULT] 已跑通（64×64 smoke）** | `train.py --override configs/flux2_klein.yaml --device cuda`：20 步 / 173.4 s、100 步 / 678.7 s；`preflight_flux2(cuda)` 已放行（实测可分配 9 GB bf16）。**真正训练出来的权重** `d_text`：20 步 **0.019874** → 100 步 **0.046817**（相对输出 3.82%）。verdict 升为 **`GPU-TRAINING-VERIFIED`**。训练数据仍是 64×64 合成图，loss 波动大未收敛，恢复质量（PSNR/SSIM/LPIPS/user study）**仍未测** |
| 文本是否真被训练利用（旧 FLUX.2 / 合成数据） | **[RESULT] 目前基本没有** | `scripts/diagnose_text_usage.py` 固定图像与噪声、只置换文本：换上**错误** prompt 时 loss 仅变 **0.003906**（正好 1 个 bf16 ULP，相对 **1.08%**）；而「有文本 vs 无文本」能降 loss **5.39%**。说明文本分支学到的是**通用有益偏置**，不是指令特定的条件控制。根因：退化类型 ↔ prompt 严格 6→6 对应，且纯 velocity loss 没有任何文本对比项。详见 `docs/方案设计.md` §8.1 与 `outputs/logs/text_usage.json` |
| **一图多目标 / 文本消歧（2026-11-12 新增）** | **[RESULT] 已跑通 A/B，方案被验证** | 同一张退化图配 **K=4 个不同高清目标**（自然色 / 黑白 / sepia / 保守修复），文本是唯一能选目标的信息。严格对照（同样 4 条文本但只有 1 个 GT）下，wrong-text loss penalty 占理论 gap **8.95%**；多目标为 **60.61%**（**6.78 倍**），样式选择性 5.36% → **11.08%**。**纯 velocity MSE 就能迫使模型读文本，无需加对比损失**。详见 `docs/一图多目标数据集调研.md`、`outputs/logs/ambiguity_summary.json` |
| **官方 500MB 数据集 + 本地训练（2026-11-13）** | **[RESULT] 已落盘并跑通** | 下载 `timbrooks/instructpix2pix-clip-filtered` 单个 parquet 分片 **499,831,414 B / 1,195 行 / 512² JPEG**，按官方 triplet 格式（`original_image→degraded`、`edited_image→clean`、`edit_prompt→prompt`）接进 `make_dataset()`；用 800 行 / 748 条真实编辑指令在 CUDA 上训 1200 步（28 s，loss 1.453→1.031，4 个 checkpoint）。**证伪检验为负**：`wrong_penalty_relative=0.017%`，`text_needed_by_loss=false` —— 格式对但没有一图多目标就没有歧义下界。随后用该分片构造 200 组 ×4 目标 A 类视图，严格 A/B 得 `penalty/gap` 对照 −1.0% vs **36.73%**，样式选择性 3.83% vs **7.92%**，`TEXT-NECESSITY-SUPPORTED`。详见 `outputs/logs/ip2p_500m_summary.json` |
| **老照片（生成的）+ 文本 → 多个不同高清目标（2026-11-13）** | **[RESULT] 已落盘、已训练；严格 A/B 未过阈值** | 自建 **800 组 × K=4 = 3200 样本 / 463.3 MB / 512² JPEG**：一张共享的合成老化老照片（fading / sepia / film_grain / scratches / dust_spots / vignette）对应 4 个不同高清修复目标（自然色 / 黑白 / sepia / 保守修复），目标间像素 RMS **52.28**，每个目标 3 条独立文本改写。CUDA 上各训 1200 步（A 类 / `collapse_targets` 严格对照）。**诚实结论：方向对但量级不足** —— 32 组探针 `penalty/gap` 对照 **0.03%** vs A 类 **7.28%**（8 组探针 0.80% vs **9.30%**），阈值 10%，verdict **`NOT-YET-SUPPORTED`**，未复现 IP2P A 类的 36.73%。根因推测：300 组 ×3 改写 = 3600 条 distinct prompt，1200 步只够每个样本约 1 次；且 4 个目标主要是全局色调差异，64×64 下采样后被压缩。示例 `outputs/samples/old_photo_multitarget_example.png`，汇总 `outputs/logs/old_photo_multitarget_summary.json` |
| **反向方案调研：冻结文本、只训参考图分支（2026-11-13）** | **[FACT] 有成熟先例，本机可直接用** | IP-Adapter（22M 参数、解耦 cross-attn）/ ControlNet 就是该范式。本地 `diffusers 0.37.1` 实测：`FluxPipeline` / `StableDiffusionXLPipeline` **自带** `load_ip_adapter()` + `set_ip_adapter_scale()`；而 `Flux2KleinPipeline` **没有** `load_ip_adapter()`，因为官方原生 `image=[...]` 多参考图（VAE latent 沿序列维拼进图像流，Qwen3 `prompt_embeds` 独立）已经做完了这件事。结论：**路走得通，但在我们选定的基座上不必造**；「原生文本能力不变」只在权重层面成立，行为层面有明确干扰证据。详见 `docs/冻结文生图_只训参考图分支_先例调研.md` |
| **真实 FLUX.2 出图实测（2026-11-13）** | **[RESULT] 已跑通，肉眼可见文本控制** | 本地 15.98 GB BF16 权重 + `enable_model_cpu_offload()`，4070Ti 12 GB，512² / 20 步 / guidance 4.0，**零训练**（原生 Qwen3 与 DiT 全冻结），参考图走官方 `image=`。组 A：同一张老化肖像 + 4 条修复风格文本 → 99 s 出 4 张，划痕/褪色去除且自然色/黑白/sepia/保守修复明显不同。组 B：用户点名的编辑指令 → 90.5 s 出 4 张，`blue_sky` 真的变蓝天白云、`beard` 真的长出浓密黑胡子、`sky_and_beard` 两者同时生效；同一 seed、同一步数，唯一变量是文本。图见 `outputs/samples/flux2_old_photo/grid.png` 与 `flux2_old_photo_edit/grid.png`，脚本 `scripts/run_flux2_old_photo_demo.py`。**边界**：这是原生模型能力，不是我们训练的文本旁路；PSNR/SSIM/LPIPS 已于 2026-11-17 补齐且为**负面结果**（见下两行与 `docs/方案设计.md` §9）。详见 `docs/冻结文生图_只训参考图分支_先例调研.md` §9 |
| **系统架构图（2026-11-17）** | **[FACT] 已交付** | 依据课程 PDF 第 4 页「报告含系统架构图」：Mermaid 可编辑源 `docs/系统架构图.md`（5 张图：总体分层 / 双模型协同与冻结边界 / 一图多目标数据流 / 推理演示流 / 评测链）+ 200 dpi 报告位图 `results/figures/system_architecture.png`；重绘 `scripts/draw_system_architecture.py` |
| **恢复质量性能分析（2026-11-17）** | **[RESULT] 已实测，结果为负，如实保留** | `scripts/eval_restoration_quality.py` 对真实 4B 出图算 PSNR / SSIM / LPIPS(AlexNet)：均值 **10.68 dB / 0.3737 / 0.7059**，而退化输入本身是 **15.14 dB / 0.5588**（ΔPSNR **−4.46 dB**）；own-GT 最近仅 **2/4** → `NO-TEXT-TARGET-ALIGNMENT`；编辑组两两 LPIPS 均值 **0.5031** → `TEXT-CHANGES-OUTPUT`。**零训练生成式重建没有提升相对 GT 的保真度**，不得包装成质量提升。图 `results/figures/restoration_quality.png`，JSON `results/reports/restoration_quality.json` |
| **可交互 Web Demo（2026-11-17）** | **[RESULT] 已端到端跑通** | `app/demo_server.py`：Python 标准库 `http.server` + 原生 HTML/JS，**零新增依赖**；上传老照片 + 文本 → 真实 BF16 FLUX.2 出图；8 条预设（含改蓝天 / 加胡子）、参数滑块、GPU 状态、6 个一键鲁棒性 / 边界用例。`scripts/verify_demo_http.py --generate` 实测：HTTP 层 7 个用例全 PASS，真实生成 384² / 8 步 **21.5 s** 返回有效 PNG。见 `docs/Demo使用说明.md` |

> **[RESULT] 「文本起作用了」要分三层，别混为一谈**（2026-11-12 实测）
> 1. **能改变输出** ✓：训练后 `d_text=0.0468`，且随步数增长（20 步 0.0199 → 100 步 0.0468）。
> 2. **被训练目标需要** ✗：换上错误文本几乎不涨 loss（1.08%，正好 1 个 bf16 ULP）；「有文本」有用，「文本对不对」几乎无所谓。
> 3. **按指令做条件控制** ✗：未验证。`d_content ≈ d_task` 只说明 projector 对不同 embedding 敏感，不说明内容约束被理解或执行。
>
> 上面三行是 **2026-11-12 早些时候在 FLUX.2 + 单目标合成数据上的结论**，仍然有效，但**已经被新数据方案部分解决**：
>
> **[RESULT] 2026-11-12 晚间更新（一图多目标）**：把「同一张退化图 + 多条不同文本」升级成
> 「同一张退化图 + **K 个不同高清目标** + 每个目标一条唯一文本」后，在 toy 上的严格 A/B 里，
> 换错文本带来的 loss 惩罚达到理论无文本下界的 **60.61%**（单目标对照只有 **8.95%**，差 **6.78 倍**）。
> 也就是说：**不用改 loss，光靠数据构造就能让纯 MSE 迫使模型使用具体指令**。
> 关键额外条件是训练时把 `t` 采样压到高噪声端（`t_min=0.9`）——否则 `x_t=(1-t)x0+t·x1`
> 会把目标身份泄漏给模型。第一轮用标准 `U(0,1)` 只有 0.6% of gap，方向对但量级不足。
> 证据：`outputs/logs/ambiguity_round1.json`、`ambiguity_round2.json`、`ambiguity_summary.json`，
> 图见 `outputs/samples/ambiguity_ab.png`。**注意**：这是 toy 主干 + 哈希文本编码器的结论，
> 证明的是数据设计与优化目标，不是语言理解或恢复质量；真实 FLUX.2 上的多目标训练尚未跑。

> **中心假设修正 `[FACT]`**：FLUX.2 [klein] 的参考图是**拼进图像流**，不是占据文本分支；`encoder_hidden_states` 是空闲的 Qwen3 文本流。本项目把 `prompt_embeds` 冻结成固定中性前导，用户文本**只**走新增可训练分支，这样 `d_text` 才能归因。详见 `docs/方案设计.md` §2.7。

> **下载路线修正 `[FACT]`（2026-11-07）**：FP8 仓库 `black-forest-labs/FLUX.2-klein-base-4b-fp8` **不是可加载的 diffusers pipeline** —— 它只有一个 4.09 GB 的 ComfyUI 原生 scaled-FP8 单文件（键为 `double_blocks.*` / `single_blocks.*` + 逐层 scale），没有 `model_index.json` / `scheduler` / `text_encoder` / `vae` / `tokenizer`；`Flux2KleinPipeline` 既无 `from_single_file` 也无 FP8 支持，`from_pretrained(FP8 repo)` 必然失败。主线是 BF16 完整 pipeline `black-forest-labs/FLUX.2-klein-base-4B`（排除重复 ComfyUI 单文件后 **15.98 GB**），按官方 `torch_dtype=torch.bfloat16` 加载。证据与实测见 `scripts/download_models.py` 顶部注释、`docs/方案设计.md` §2.1。

> **[RESULT] 2026-11-13 老照片多目标：一个诚实的负面结果。** 数据集格式完全符合「老照片 + 文本 → 不同高清图」，数据层面 `Var_k(y_k)=0.005665 > 0` 也确实让文本在理论上成为必要信息；但 1200 步训练后，换错文本带来的 loss 惩罚只达到理论下界的 **7.28%**（严格对照 0.03%），没过 10% 判据。`verdict = NOT-YET-SUPPORTED`。不要把它写成「验证成功」，也不要拿 IP2P A 类的 36.73% 来代替它。下一步应固定单条改写文本、把 `t_min` 提到 0.95、降 `num_pairs` 到 200 并加长步数，再重跑同一套 A/B。证据：`outputs/logs/ambiguity_old_photo_ab_g32.json`、`ambiguity_old_photo_ab.json`、`old_photo_multitarget_summary.json`。
>
> **[FIX] 离线文本缓存竞态（2026-11-13）**：两个训练进程首次同时构建同一个 `OfflineTextCache` 时，一方会读到写了一半的 `.pt` 并崩在 `PytorchStreamReader failed finding central directory`。`src/data/text_features.py` 已改为 `os.replace()` 原子发布 + 读取失败自愈重算；全量校验 2936 个缓存文件无损坏，对照训练重跑通过。

> **不要拿 toy 的 PSNR 当恢复质量指标。** toy 主干随机初始化且全程冻结，本身不具备恢复能力；PSNR ≈ 7 dB 只用于 A/B/C/D 组内相对比较。真实基座才携带扩散先验。
>
> **环境按用户决定在 `ChatAI` 内原地升级，没有新建 conda 环境。** 升级记录见 `docs/环境升级.md`，回滚快照 `docs/升级前包快照_20261107.txt`。
>
> **[FIX] 不装 gradio（2026-11-17）**：装 `gradio==4.44.1` 会把 `pydantic 1.10.13→2.14.3`、`rich 13.7.1→15.0.0`、`typer 0.9.0→0.29.1`，`pip check` 立刻报 `fschat / spacy / streamlit / weasel` 四个冲突。已卸载 gradio、恢复三个原版本，`pip check` 回到 `No broken requirements found`，`torch / diffusers / transformers / accelerate` 版本不动。Demo 改为零新增依赖的 `app/demo_server.py`，旧入口 `app/gradio_demo.py` 保留为转发 shim。详见 `docs/Demo使用说明.md` §1。

---

## 环境

硬件实测：**`RTX 4070 Ti`，12282 MiB**。2026-11-07 曾长期只有 3.4–6.9 GB 空闲、`preflight_flux2()` 硬拦 GPU 训练；**2026-11-12 已放开**：`nvidia-smi` 空闲 7.3–10.5 GB，`torch.cuda.mem_get_info()` 报 11.59 GB，实测可成功分配 9 GB bf16 张量，`preflight_flux2(cuda)` 放行，真实 4B GPU 训练**已跑通**（见状态表）。系统内存 34.2 GB；G 盘约 27–37 GB 空闲。训练时显存峰值 **~11.8 GB / 100% 利用率**，已贴近 12 GB 上限；`nvidia-smi` 与 `torch` 的空闲数字口径不一致，跑前仍以 `nvidia-smi` 实时数字为准。

统一用 `D:/Anaconda/envs/ChatAI/python.exe`（Python 3.11.8 / torch 2.5.1+cu121 / CUDA 可用）。

已升级：`diffusers 0.37.1` · `transformers 4.57.6` · `accelerate 1.0.1` · `peft 0.19.1` · `huggingface_hub 0.36.2` · `sentence-transformers 4.1.0`。

**别追最新版**：`diffusers ≥ 0.42` 会调用 `torch.accelerator`，torch 2.5.1 没有该 API，直接 import 失败；`transformers ≥ 5` 会破坏 `sentence-transformers`。详见 `docs/环境升级.md`。

---

## 快速开始（本地已验证路径）

```bash
set P=D:/Anaconda/envs/ChatAI/python.exe
set R=G:/研究生生涯相关/课程/人工智能系统实训

# 0. 结构 / 去噪契约自检（不需要权重，秒级）
%P% %R%/scripts/check_flux2_contract.py
%P% %R%/scripts/check_flux2_train_contract.py

# 1. 架构验证（约 10 秒 -> outputs/logs/smoke_test.json）
%P% %R%/src/smoke_test.py

# 2. 训练两种注入模式（各 200 步，约 3 秒）
%P% %R%/src/train.py --steps 200 --mode projector_only        --out-dir outputs/checkpoints/projector_only
%P% %R%/src/train.py --steps 200 --mode projector_plus_bypass --out-dir outputs/checkpoints/bypass

# 3. 四组消融 + 对比图
%P% %R%/scripts/run_ablation.py

# 4. 单张推理
%P% %R%/src/inference.py --ckpt %R%/outputs/checkpoints/bypass/cond_step000200.pt --prompt "restore and denoise this image"
```

诊断工具：`scripts/diagnose_text_path.py`（逐层打印文本信号差异，区分实现 bug 与真实结论）。

---

## 真实数据集：聚焦老照片修复（2026-11-07 新增）

**不再拿 64×64 程序化合成图当训练数据。** 按论文做法改成三段式：真实 DIV2K 高清照片作 clean GT → 论文式物理老化退化生成配对 → 真实老照片（**无 GT**）单独进评测集。完整对标（Wan et al. CVPR 2020 / InstructIR ECCV 2024 / Time-Travel Rephotography TOG 2020 / CVPR 2023 多参考 / LL-Bench）见 `docs/真实数据集与论文对标.md`。

```bash
set P=D:/Anaconda/envs/ChatAI/python.exe
set R=G:/研究生生涯相关/课程/人工智能系统实训

# 1) 下载真实图像并构建配对（已下载文件自动跳过；--offline 用 data/raw/ 里自己的照片）
%P% %R%/scripts/build_old_photo_data.py --gt 24 --captions 48 --real-old 14 --size 256

# 2) 可视化真实 clean / degraded / prompt，以及无 GT 的真实老照片评测集
%P% %R%/scripts/show_old_photo_example.py

# 3) 训练（toy 主干验证架构；真实基座另加 --override configs/flux2_klein.yaml）
%P% %R%/src/train.py --override %R%/configs/old_photo.yaml --steps 300

# 4) 文本响应消融：同一张真实图，只改用户文本
%P% %R%/scripts/run_ablation.py --override %R%/configs/old_photo.yaml
```

关键设计：`make_dataset()` 按 `data.source` 分发，`configs/default.yaml` 无该字段 → 原有 toy 全部 `[RESULT]` 数值逐位不变。新数据集修掉了原 toy 的 **confound**（退化类型与 prompt 用 `i % 6` 一一绑定）：随机缺陷组合、instruction 是噪声子集（随机丢弃 + 注入干扰指令）、真实 caption 池解耦、每张图 3 条不同文本。真实老照片**不进入任何 (clean, degraded) 训练对**。

但这个数据还有一个更深的洞：**每张图只有 1 个 clean GT**，所以换文本不改变目标，忽略文本仍是最优解。

## 一图多目标：让文本成为唯一消歧信息（2026-11-12 新增）

用户方案：文本要提供图像没有的信息；**同一张低清图配多个高清目标**，不同目标对应不同文本。论文/数据集对标见 `docs/一图多目标数据集调研.md`（InstructPix2Pix / RFIR / PRISM / InstructRestore / UltraEdit / SEED-Data-Edit / ImgEdit / CIN25 / Text2Colors-PAT / 老照片修复）。

本项目落地：24 张真实 DIV2K 照片 → **1 张共享老化退化图 x** + **K=4 个不同高清目标 y_k**（natural_color / monochrome / sepia_tone / minimal_repair），共 96 训练样本、288 条不同 prompt。目标间像素 RMS 平均 **48.6/255**，token 空间 `Var_k(y_k)=0.00547 > 0` —— 这是无文本模型 MSE 的**理论下界**，不需要训练就能算。

```bash
set P=D:/Anaconda/envs/ChatAI/python.exe
set R=G:/研究生生涯相关/课程/人工智能系统实训

# 1) 构建（从已落盘的真实 DIV2K GT；--strong 放大目标间差异）
%P% %R%/scripts/build_multi_target_data.py --strong
%P% %R%/scripts/show_multi_target_example.py

# 2) 严格 A/B：唯一变量是「GT 是否随文本变化」（collapse_targets）
%P% %R%/src/train.py --override %R%/configs/multi_target_strong.yaml --device cpu --out-dir outputs/checkpoints/amb_multi_strong
%P% %R%/src/train.py --override %R%/configs/single_target_strong.yaml --device cpu --out-dir outputs/checkpoints/amb_single_strong

# 3) 量化：wrong-text penalty / 理论 gap、样式选择性
%P% %R%/scripts/diagnose_ambiguity.py --groups 8 --checkpoint outputs/checkpoints/amb_single_strong/cond_step001200.pt --checkpoint outputs/checkpoints/amb_multi_strong/cond_step001200.pt --out ambiguity_round2.json
%P% %R%/scripts/show_ambiguity_ab.py
```

真实 FLUX.2 路径用合并好的 `configs/multi_target_flux2.yaml`（`train.py` 只接受一个 `--override`）。**必须先重跑离线文本编码**：现有 Qwen3 缓存只有旧数据集的 10 条 prompt，新数据集有 288 条，CPU 上约 3 小时 / 10 条，需按量评估时间预算；在此之前不要声称真实基座的多目标训练已验证。

## 官方 500MB 数据集 → 本地训练（2026-11-13）

用户要求「格式符合要求即可，不必全部训练；拿最符合期望的数据集做一个 500M 的出来，开始训练」。

**选型**：`timbrooks/instructpix2pix-clip-filtered`（官方 CLIP-filtered，313,010 对 / 436GB，但 HuggingFace 拆成 262 个 parquet 分片，**每个 ~500MB**）。只下第 0 个分片 = **499,831,414 B**，不需要碰全量。这是调研里最可复现的「一输入 + 文本指令 → 目标图」triplet 范式。

```bash
set P=D:/Anaconda/envs/ChatAI/python.exe
set R=G:/研究生生涯相关/课程/人工智能系统实训

# 1) 下载 500MB 官方分片 + 核实结构
%P% %R%/scripts/fetch_ip2p_shard.py --shard 0
%P% %R%/scripts/inspect_ip2p_shard.py
%P% %R%/scripts/show_ip2p_example.py

# 2) 按官方 triplet 格式直接训练（800 行子集，CUDA 1200 步）
%P% %R%/src/train.py --override %R%/configs/ip2p_500m.yaml --device cuda --steps 1200 --out-dir outputs/checkpoints/ip2p_500m
%P% %R%/scripts/diagnose_ip2p_text.py

# 3) 用同一分片构造 A 类视图并做严格 A/B
%P% %R%/scripts/build_ip2p_multitarget.py --groups 200
%P% %R%/src/train.py --override %R%/configs/ip2p_multitarget.yaml --device cuda --steps 1200 --out-dir outputs/checkpoints/ip2p_multitarget
%P% %R%/src/train.py --override %R%/configs/ip2p_multitarget_control.yaml --device cuda --steps 1200 --out-dir outputs/checkpoints/ip2p_multitarget_control
%P% %R%/scripts/diagnose_ambiguity.py --override %R%/configs/ip2p_multitarget.yaml --groups 8 --checkpoint outputs/checkpoints/ip2p_multitarget_control/cond_step001200.pt --checkpoint outputs/checkpoints/ip2p_multitarget/cond_step001200.pt --out ambiguity_ip2p_ab.json
%P% %R%/scripts/show_ip2p_summary.py
```

**必须如实标注的事实**：
- `[FACT]` 官方图像是 SD + Prompt-to-Prompt **生成**的，不是真实拍摄照片。
- `[FACT]` `scripts/inspect_ip2p_shard.py` 实测 `a_class_native=false`：1,195 个输入图全部互不相同，每个输入只有一个目标，**不是**一图多目标。格式符合项目 batch 契约，但没有 `Var_k(y_k)` 歧义下界。
- `[RESULT]` 按原格式训练后，换错 `edit_prompt` 只让 loss 变 `0.000176`（相对 0.017%），`text_needed_by_loss=false`。**格式对 ≠ 文本被需要**。
- `[RESULT]` 补上 A 类视图（200 组 ×4 目标，目标间 RMS 52.67）后，严格对照 `penalty/gap` −1.0% → **36.73%**，样式选择性 3.83% → **7.92%**，复现 §8.2 结论。
- `[FACT]` 训练走 toy 主干 + 64×64；真实 FLUX.2 需要为 748 条新 `edit_prompt` 重跑 Qwen3 离线缓存（CPU 约 3h/10 条）。按「不必全部训练」可取 10–20 条子集做真实基座试点，但**目前没有跑**。
- `data/ip2p_500m/` 合计 574MB（499.83MB 官方分片 + 74MB 衍生图）。

---

## 目录结构

```
configs/default.yaml             toy 全套配置（本地已验证）
configs/flux2_klein.yaml         真实基座覆盖配置（dims 已按真实权重实测：128/3072/7680）
configs/old_photo.yaml           真实老照片数据集配置（data.source=old_photo）
configs/multi_target.yaml        ONE 退化图 -> K=4 个不同 GT（data.source=multi_target）
configs/single_target_control.yaml  严格对照：同 4 条文本但只有 1 个 GT
configs/multi_target_strong.yaml / single_target_strong.yaml  强信号 A/B（t_min=0.9, lr 5e-4）
configs/multi_target_flux2.yaml  真实基座 + 多目标（合并 override，需重跑 Qwen3 缓存）
src/condition/                   唯一可训练的文本分支（projector + 零初始化旁路）
src/backends/toy.py              双流 MMDiT 风格 toy 主干
src/backends/flux2_klein.py      FLUX.2 Klein Base 对接层（结构+权重 [FACT] / GPU 训练 [UNVERIFIED]）
src/system.py                    后端分发 + FLUX.2→ToyMMDiT 接口适配 + 硬 preflight
src/data/old_photo.py            论文式 11 种老化缺陷（结构化 + 非结构化）
src/data/old_photo_dataset.py    真实图配对 + 去 confound 的多文本采样
src/data/multi_target_old_photo.py  共享退化 + K 个目标风格 / paraphrase 生成
src/data/multi_target_dataset.py    一图多目标 Dataset（get_group / all_prompts / collapse 对照）
src/data/dataset.py              make_dataset() 工厂 + 原 toy RestorationDataset
src/data/                        合成退化 / 真实老照片退化与配对 / 文本特征与离线缓存
src/train.py  inference.py  evaluate.py  smoke_test.py
scripts/check_flux2_contract.py       结构契约自检（含 hook 注入位置模拟，全 PASS）
scripts/check_flux2_train_contract.py 去噪 forward 契约自检（18/18 PASS，含中心假设修正）
scripts/download_models.py       下载 BF16 完整 pipeline（direct 引擎，Range 断点续传）
scripts/watch_download.py        实时下载进度面板（--once / --interval N）
scripts/inspect_backbone.py       加载真实权重，实测 runtime dims + 配置对齐检查
scripts/precompute_prompts.py     CPU 离线编码全部 prompt（Qwen3-4B 永不进显存；缓存已齐则秒级跳过）
scripts/test_latent_roundtrip.py  快速 VAE-only T0 回归（秒级；验证 BN 反归一化顺序 + 负向对照）
scripts/verify_real_forward.py    真实权重 CPU 端到端：latent 往返 / 恒等 / 梯度隔离 / 文本响应
scripts/verify_text_warmup.py     tiny stub 秒级复现 T4=0 的 zero-init 单步 warm-up 动力学
scripts/verify_real_text_effect.py 真实 4B 主干上的文本通路因果性（4 次 forward，无 backward）
scripts/verify_trained_text_effect.py 用 GENUINELY TRAINED 权重测 d_text（替代上面的扰动证据）
scripts/diagnose_text_usage.py    文本是否真被损失需要：d_content / d_loss_wrong / confound 诊断
scripts/show_train_data_used.py   可视化「本次训练到底喂了什么数据」
scripts/watch_gpu_train.py         GPU 训练实时面板（--log / --out-dir / --interval）
scripts/watch_real_verify.py      验证链实时终端面板（--once / --interval N / --log）
scripts/verify_real_backend.py    静态 preflight（版本 / 权重 / dims / 缓存 / 显存）
scripts/build_old_photo_data.py  下载真实 DIV2K/caption/老照片 + 构建配对 manifest
scripts/show_old_photo_example.py 真实数据可视化（含无 GT 的老照片评测集）
scripts/build_multi_target_data.py   从真实 GT 构建一图多目标 manifest（--strong）
scripts/show_multi_target_example.py 一图多目标对照图（共享 x + 4 个 y + 各自文本）
scripts/diagnose_ambiguity.py    Var_k(y_k) 下界 + own/wrong/off loss + 样式选择性
scripts/show_ambiguity_ab.py     两轮 A/B 汇总图与 ambiguity_summary.json
scripts/draw_system_architecture.py  重绘报告用系统架构图（PNG 200 dpi）
scripts/eval_restoration_quality.py  PSNR / SSIM / LPIPS + 目标选择性 + 文本响应
scripts/verify_demo_http.py      Demo 的 HTTP 层可复现验证（--generate 跑真实出图）
scripts/run_flux2_old_photo_demo.py  真实 4B 出图（修复风格 / 内容编辑两组）
scripts/                         数据、探测、自检、消融、诊断、评测
app/demo_server.py               零新增依赖的交互 Web Demo（stdlib HTTP）
app/gradio_demo.py               已废弃的兼容转发 shim（不再需要 gradio）
outputs/                         checkpoint / 日志 / 样例图
docs/方案设计.md                  方案论证、修正、显存预算、失败判据、§11 结构契约、§12 三维对齐
docs/环境升级.md                  ChatAI 原地升级记录与回滚
docs/实验记录模板.md               每次实验的复现记录模板
docs/真实数据集与论文对标.md        论文数据构造/规模/指标对标 + 本项目真实数据方案
docs/一图多目标数据集调研.md        一输入多目标范式对标 + 歧义 gap 推导 + 两轮 A/B 结果
docs/仓库结构与复现.md             入库/不入库规则、逐条复现命令、成果索引
docs/课程目标与达成情况.md          官方课程要求逐条对照与 10 天优先级
docs/系统架构图.md                 报告用架构图（Mermaid 源 + 位图）
docs/Demo使用说明.md               交互 Demo 启动、接口、边界用例与负面声明
results/figures/                   11 张精选图（含 system_architecture / restoration_quality）
results/reports/                   10 份关键 JSON 报告
```

---

## 面向课程要求的三项交付（2026-11-17 新增）

依据 `AI系统实训_课程安排.pdf` 第 4 页：系统架构图、性能分析、基本可用 Demo（含鲁棒性 / 边界情况）。逐条达成对照见 `docs/课程目标与达成情况.md`。

```bash
set P=D:/Anaconda/envs/ChatAI/python.exe
set R=G:/研究生生涯相关/课程/人工智能系统实训

# 1) 系统架构图（无需权重 / GPU）
%P% %R%/scripts/draw_system_architecture.py        # -> results/figures/system_architecture.png

# 2) 恢复质量性能分析（读已落盘的真实出图，不重新推理）
%P% %R%/scripts/eval_restoration_quality.py        # -> results/reports/restoration_quality.json

# 3) 交互 Demo（终端 1：启动服务；终端 2：验证）
%P% %R%/app/demo_server.py --port 7860             # 打开 http://127.0.0.1:7860/
%P% %R%/app/demo_server.py --selftest              # 不加载模型，5 个边界用例
%P% %R%/scripts/verify_demo_http.py                # HTTP 层 7 个用例
%P% %R%/scripts/verify_demo_http.py --generate     # 端到端真实出图
```

> **诚实边界**：性能分析给出的是**负面结果**（零训练生成式重建相对数据集 GT 的
> PSNR/LPIPS 劣于退化输入本身）；Demo 展示的是**原生 FLUX.2 零训练能力**，
> 不是自训文本旁路。一图多目标 verdict 仍为 `NOT-YET-SUPPORTED`。

---

## 迁移到真实基座（权重已落盘 / CPU 验证阶段）

统一加 `--override configs/flux2_klein.yaml` 才会走真实基座；不加就是已验证的 toy 路径。

```bash
set P=D:/Anaconda/envs/ChatAI/python.exe
set R=G:/研究生生涯相关/课程/人工智能系统实训

# 1. 下载 BF16 完整 diffusers pipeline（21 文件 / 15.98 GB，已完成）
%P% %R%/scripts/download_models.py --dry-run       # 看计划（复用 manifest，零网络依赖）
%P% %R%/scripts/download_models.py                 # 断点续传，写 outputs/logs/download_state.json
%P% %R%/scripts/watch_download.py --interval 5     # 实时面板；--once 出一次快照

# 2. CPU 侧真实权重验证（不需要 GPU）
%P% %R%/scripts/inspect_backbone.py                # 实测 dims + config_reconciliation
%P% %R%/scripts/precompute_prompts.py             # CPU 离线编码全部 prompt -> data/text_cache/
%P% %R%/scripts/verify_real_forward.py            # 真实权重 T0–T4（T4 单步预期=0，见下方说明）
%P% %R%/scripts/verify_text_warmup.py             # 秒级解释 T4=0 的 zero-init warm-up 动力学
%P% %R%/scripts/verify_real_text_effect.py        # 真实 4B 主干文本通路因果性（约 10 min，无 backward）
%P% %R%/scripts/verify_real_backend.py            # 汇总分级，现应 GPU-TRAINING-VERIFIED

# 3. GPU 训练（2026-11-12 已跑通；显存需 >= 9 GB，跑前看 nvidia-smi）
%P% %R%/src/train.py --override configs/flux2_klein.yaml --device cuda --steps 100 --out-dir outputs/checkpoints/flux2_klein_100
%P% %R%/scripts/watch_gpu_train.py --interval 5 --log gpu_train_100.log --out-dir outputs/checkpoints/flux2_klein_100
%P% %R%/scripts/verify_trained_text_effect.py --checkpoint outputs/checkpoints/flux2_klein_100/cond_step000100.pt
%P% %R%/scripts/diagnose_text_usage.py           # 文本是否真被利用（confound 诊断）
%P% %R%/scripts/show_train_data_used.py          # 本次训练到底喂了什么数据
%P% %R%/scripts/run_ablation.py
```

### 真实权重实测 `[FACT]`（2026-11-07 14:54，CPU 加载 BF16 完整 pipeline）

`scripts/inspect_backbone.py` 加载真实权重后测得，已回写 `configs/flux2_klein.yaml`：

| 项 | 实测值 | 说明 |
|---|---|---|
| `in_channels` | **128** | packed VAE latent token 宽度，`ref_bias_proj` 的输出维度 |
| `inner_dim` | **3072** | 24 heads × 128，double block hidden；旁路 q/o 投影宽度 |
| `joint_attention_dim` | **7680** | 3 × Qwen3 hidden 2560（层 9/18/27），即 `text_in_dim` |
| `transformer_blocks` | **5** | 只在这 5 个 double block 挂旁路 |
| `single_transformer_blocks` | **20** | 接收 `cat([txt,img])` 拼接流，不挂旁路 |
| double block 输出 | `(txt, img)` | 故 `IMG_INDEX = 1` |

早期占位值 `hidden_dim / cond_dim = 6144` 是错的，已按实测改为 `3072`；`inspect_backbone.py` 现在自动做 `config_reconciliation`，维度不一致直接报 `CONFIG_MISMATCH`，避免后面以晦涩的 shape mismatch 失败。

另一个真实权重路径上必然踩到的坑：冻结主干以 **bf16** 运行，hook 收到的 hidden states 是 bf16，而 `TextConditionModule` 默认 float32 —— 直接相加会混合 dtype 报错。修法：`build_flux2_system()` 显式 `.to(dtype=torch.bfloat16)`，`set_text()` 把 float32 的离线缓存转成模块自身 dtype；toy 全程 float32，回归数值逐位不变。

**下载引擎说明**：默认 `direct` 是顺序、Range 断点续传的自写下载器，直接写 `<target>.incomplete` 后用 `os.replace()` 收尾。不用 `snapshot_download` 是因为它会为每个文件创建并**删除** `.lock` / `tmp_*`，本工作区沙箱对批量删除有 `SAFE_DELETE_BULK_CONFIRM_REQUIRED` 拦截，实测会卡在 `1/21`。`--engine hub` 保留为可选回退；`--fp8-singlefile` 只下 4.09 GB 研究素材，**不可加载**。

`src/system.py::preflight_flux2()` 会在权重不全、CUDA 目标显存不足时**硬失败并给出精确下一步**，不会让工程在半加载状态下静默跑出错的结果。显存不足现在记为 warning（不阻断 CPU 契约验证），GPU 硬拦在 preflight 里。

### 为什么 `verify_real_forward.json` 是 `FAILED`，汇总却不判失败

`verify_real_forward.py` 的 T4 在**一步训练之后**测 `d_text`，而本项目为了保证 T1「零初始化恒等」，把 projector 末层和每个 bypass 的 `o_proj` 都零初始化了：

- 第 1 步只有 `o_proj.weight/bias` 拿到梯度（真实运行实测正好 **10 个 tensor** = 5 个 double block × 2）；projector 输出仍精确为 0，K/V 与 prompt 无关 → 两个 prompt 的输出逐位相同 → `d_text=0.0`。
- `scripts/verify_text_warmup.py` 用 tiny float32 stub 秒级复现：step1 只有 2 个参数有梯度、`d_delta=0`；step2 起 projector 打开、`d_delta>0` 并逐帧增长。所以这是**单步 warm-up artefact，不是架构缺陷**，也不是 bf16 精度吞掉了信号。
- `scripts/verify_real_text_effect.py` 直接问因果问题（**不做 backward、只 4 次真实 transformer forward，约 10 分钟**）：零初始化对照 `d_text=0.0` 精确复现 T4；把两个零初始化张量确定性地激活成非零后，真实 4B 主干输出 `d_text_rms=0.2496`（相对输出 **23.5%**）。

> **诚实边界（2026-11-12 更新）**：激活用的是 `WEIGHT-PERTURBED, NOT TRAINED` 权重，证明的是「文本通路与冻结主干因果连通」，**不是**「训练收敛后文本生效」，更不是恢复质量结论。`verify_real_forward.json` 的历史 `FAILED` 故意不改写；`verify_real_backend.py` 把 T0–T3 与 T4 分开，分级先报 `FORWARD-READY`。随后**真实 GPU 训练已跑通**，`scripts/verify_trained_text_effect.py` 用 GENUINELY TRAINED 权重量到 `d_text=0.019874`（20 步）→ `0.046817`（100 步），verdict 升为 **`GPU-TRAINING-VERIFIED`**。
>
> **但更重要的发现**：`scripts/diagnose_text_usage.py` 显示，换错误文本只让 loss 变 **1 个 bf16 ULP（相对 1.08%）**，而「有文本 vs 无文本」降 loss **5.39%** —— 文本分支目前学的是**通用偏置**，训练目标并没有迫使模型使用**具体指令**。根因是退化类型 ↔ prompt 严格 6→6 对应 + 纯 velocity loss 无文本对比项。PSNR/SSIM/LPIPS、≥256 分辨率训练、指令跟随评测仍全部未做。

`FLUX.2` 已核实的关键点（会静默出错的那类，全在 `docs/方案设计.md` §11–§12）：
- double block 返回 `(txt, img)`，旁路必须挂 `img_index=1`；single block 吃拼接流，不挂旁路；
- 参考图拼进**图像流**，文本流是空闲的 Qwen3 `prompt_embeds`（本项目把它冻结成固定中性前导，见 §2.7）；
- 三个宽度 `in_channels / inner_dim / joint_attention_dim` 不能混用；
- `timestep` 在 transformer 内部 `*1000`，我们的 `t∈[0,1]` 直接传入。

---

## 仓库结构（2026-11-13 建立私人仓库）

| ✅ 入库（约 7 MB） | ❌ 不入库（本机保留，可重建） |
|---|---|
| `src/` `scripts/` `configs/` `docs/` `app/` | `models/` 17 GB（FLUX.2 BF16 权重 + LPIPS 骨干缓存 234 MB） |
| `README.md` `requirements.txt` `.gitignore` `AI系统实训_课程安排.pdf` | `data/` 2.7 GB（数据集 / 分片 / 文本缓存） |
| `results/figures/` 11 张精选对比图 | `outputs/` 1.6 GB（checkpoint / 全量日志 / 5226 张原图） |
| `results/reports/` 10 份关键 JSON 报告 | `*.pt` `*.safetensors` `*.parquet` `__pycache__` `*.log` |

完整目录语义、逐条复现命令、成果索引与诚实边界见 **[`docs/仓库结构与复现.md`](docs/仓库结构与复现.md)**。

最快看懂成果的三张图：
`results/figures/flux2_edit_grid.png`（同一张老照片，改蓝天 / 长胡子，文本是唯一变量）、
`results/figures/flux2_restoration_grid.png`（4 种修复风格）、
`results/figures/old_photo_multitarget_summary.png`（一图多目标严格 A/B，**未过阈值**）。
