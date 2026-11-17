# 交互 Demo 使用与验证说明

> **课程依据**：`AI系统实训_课程安排.pdf` 第 4 页「Demo 达到基本可用（可展示鲁棒性 / 边界情况）」
> **落盘日期**：2026-11-17
> **入口**：`app/demo_server.py`（Python 标准库 HTTP 服务，零新增依赖）
> **兼容入口**：`app/gradio_demo.py` 已改为转发 shim，旧命令仍可用

---

## 1. 为什么不是 Gradio

原骨架 `app/gradio_demo.py` 标注 `[UNVERIFIED]`（gradio 未安装）。2026-11-17 做过一次安装预演与实际安装：

```
pip install gradio==4.44.1
→ pydantic 1.10.13 → 2.14.3
→ rich     13.7.1  → 15.0.0
→ typer    0.9.0   → 0.29.1
```

`pip check` 随即报 4 个冲突：`fschat 0.2.35` 需要 `pydantic<2`、`spacy 3.7.2` 与 `weasel 0.3.4` 需要 `typer<0.10`、`streamlit 1.30.0` 需要 `rich<14`。用户明确要求**不新建环境、只在 ChatAI 上原地升级**，破坏共享环境里其它包不可接受。

**处理**：卸载 gradio，把 `pydantic==1.10.13 / rich==13.7.1 / typer==0.9.0` 恢复原版本，`pip check` 回到 `No broken requirements found`，`torch 2.5.1+cu121 / diffusers 0.37.1 / transformers 4.57.6 / accelerate 1.0.1` 全部保持不动。Demo 改用 `http.server` + 原生 HTML/JS，**零新增依赖**。

---

## 2. 启动

```bash
cd "G:/研究生生涯相关/课程/人工智能系统实训"
D:/Anaconda/envs/ChatAI/python.exe app/demo_server.py --port 7860
# 浏览器打开 http://127.0.0.1:7860/
```

前置条件：本机 `models/FLUX.2-klein-4B-Base/` 的 BF16 完整 pipeline 已下载（17 GB，不入库）。
首次点击「生成」时才加载权重（约 30–60 s），之后复用；`enable_model_cpu_offload(gpu_id=0)` 让 12 GB 4070 Ti 可跑。

**无模型也能验证**：`--selftest` 与 `/api/validate` 不加载任何权重，秒级完成，适合中期汇报现场演示鲁棒性。

---

## 3. 页面功能

| 区域 | 内容 |
|---|---|
| ① 输入 | 上传老照片 / 6 张内置样例；8 条预设文本（4 条修复风格 + 4 条内容编辑，含「背景改蓝天」「人脸加胡子」）；steps / size / guidance / seed 可调 |
| ② 输出 | 生成图、耗时、实际 guidance、归一化告警、系统状态 JSON（权重是否在位、pipeline 状态、GPU 空闲显存） |
| ③ 边界 | 6 个一键用例，直接演示输入校验与自动归一化，**不消耗 GPU** |

### 边界用例与预期行为

| 用例 | 系统响应 |
|---|---|
| 空图 | HTTP 400，`未提供图像：请先上传或选择一张老照片` |
| 空文本 | 自动回退到中性修复指令，带 warning，不报错 |
| 超长文本（>1000 字） | 截断到 1000 字，带 warning |
| 损坏图像文件 | HTTP 400，`UnidentifiedImageError` |
| 越界参数 `steps=999 / size=4096 / guidance=99 / seed=-1` | 分别裁剪到 `50 / 1024 / 8.0 / 0`，size 再对齐到 16 的倍数，带 warning |
| 灰度 / 非 RGB 输入 | 自动 `convert("RGB")`，带 warning |
| CUDA OOM（真实生成时） | 自动 `empty_cache()` 后以 `guidance=1.0` 重试，带 warning |
| 权重缺失 | `/api/status` 显示 `weights_present=false`，生成返回明确错误 |

---

## 4. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单文件 HTML/JS 页面 |
| GET | `/api/status` | 权重 / pipeline / GPU / 最近一次生成 / 8 条预设 |
| GET | `/api/examples` | 6 张内置老照片缩略图（base64） |
| POST | `/api/validate` | 只做校验 + 归一化，**不加载模型** |
| POST | `/api/generate` | 校验 + 真实 FLUX.2 生成，返回 PNG base64 |

请求体（两个 POST 共用）：

```json
{
  "image": "data:image/jpeg;base64,...",
  "prompt": "Restore this old photograph ...",
  "steps": 20, "size": 512, "guidance": 4.0, "seed": 20261113
}
```

---

## 5. 可复现验证

```bash
P=D:/Anaconda/envs/ChatAI/python.exe
$P app/demo_server.py --selftest                 # 5 个边界用例 + 1 个合法输入
# 另开一个终端，服务已启动后：
$P scripts/verify_demo_http.py                  # HTTP 层：首页 / 状态 / 样例 / 7 个边界用例
$P scripts/verify_demo_http.py --generate       # 端到端：8 步 / 384² 真实出图
```

`verify_demo_http.py` 写出 `outputs/logs/demo_http_verify.json`；带 `--generate` 时另存
`outputs/samples/demo_http_generated.png`。

---

## 6. 边界声明（必须写进报告）

- 本 Demo 用的是**冻结的原生 FLUX.2 [klein] 4B Base**（原生 Qwen3 文本编码 + 原生 `image=[ref]`），
  **零训练**，不是本项目自训文本旁路的成果。
- 它证明的是「文本 + 参考图 → 不同输出」这一交互链路可用；**不能**用来宣称一图多目标
  `TEXT-NECESSITY-SUPPORTED`。严格 A/B 的最新 verdict 仍是 `NOT-YET-SUPPORTED`
  （32 组探针 `penalty/gap = 7.28% < 10%`）。
- `scripts/eval_restoration_quality.py` 的实测显示：零训练生成结果相对数据集 GT 的
  PSNR / LPIPS **甚至低于未修复的退化输入**，属于诚实的负面结果，不得包装成质量提升。
- 出图是生成式重建，不保证人脸身份逐像素一致，也不提供 PSNR 意义上的「真值恢复」。
