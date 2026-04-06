# simple-LLMRF

`simple-LLMRF` 是一个面向 GGUF 稠密模型推理过程拆解的实验项目，目标不是做通用推理框架，而是围绕一个确定模型，把从 `GGUF` 加载到多 token 生成的核心链路完整打通，并保留足够清晰的代码结构，便于教学、调试和答辩展示。

当前版本的收敛目标已经明确为：

- 只支持 `models/internlm2-1_8-F16.gguf`
- 完成该模型的 CPU / CUDA 推理闭环
- 保留推理链路中的关键中间环节，方便观察和验证

## 项目状态

当前项目已经完成从模型加载到文本生成的功能闭环，包含：

- GGUF 文件加载、metadata 解析、tensor 索引与按名访问
- `Tokenizer` 编码 / 解码
- prompt 准备、position 构造、token embedding
- `internlm2` 配置解析与张量布局映射
- runtime 与 KV cache 分配
- 多层 decoder block forward
- final norm 与 lm head logits 计算
- `greedy sampling` 与基础 `stochastic sampling`
- 多 token 自回归生成
- `cpu` 与 `cuda[:n]` 两种执行设备入口
- 原生 CUDA 后端的关键路径接入

项目现阶段更适合被理解为：

- 一个面向 `internlm2-1_8-F16.gguf` 的定向推理实现
- 一个用于展示大模型推理主链路的教学 / 实验工程

而不是：

- 面向多模型、多量化格式的通用推理框架

## 已实现能力

- 使用 `mmap` 读取 GGUF 文件
- 解析 GGUF v2/v3 文件头、metadata、tensor infos
- 使用强类型 `MetadataValue` 表示标量、字符串和数组 metadata
- 为 GGUF 张量提供按名字索引、按行读取、F16/F32 转 float32 的访问接口
- 从 GGUF 中构建 tokenizer，并完成基础编码/解码流程
- 提供 `internlm2` 专用模型配置解析、张量命名布局和 KV cache/runtime
- 实现 `prompt -> token ids -> positions -> token embedding`
- 实现完整 decoder block 主链：
  `attn_norm -> qkv -> RoPE -> KV cache write -> causal attention -> attn residual -> ffn_norm -> SwiGLU -> ffn residual`
- 实现 `output_norm + lm_head(last token) -> logits`
- 实现多层 rollout 与多 token 自回归生成
- 实现 `greedy sampling`
- 实现基础 `stochastic sampling`，支持 `temperature` 与 `seed`
- 提供 `Device` / `TensorBuffer` / `OperatorContext` 多设备抽象
- 提供 CPU backend
- 提供 CUDA backend，包括：
  - tensor device allocation
  - embedding lookup
  - rms norm
  - add / add_inplace / multiply / silu
  - linear projection
  - rope
  - KV cache write
  - causal attention
  - lm head logits projection
- 提供 OpenMP 多核 CPU 执行
- 提供 CLI 用于查看模型摘要、分词结果、embedding 预览、block rollout 和生成结果
- 提供最小测试夹具与自动化测试

## 当前支持范围

当前版本只保证以下组合：

- 模型：`models/internlm2-1_8-F16.gguf`
- 架构：`internlm2`
- 推理任务：单轮 prompt prefill + 自回归生成
- 设备：`cpu`、`cuda`

当前不承诺：

- 通用 GGUF 多模型兼容
- 量化格式全面支持
- 高级采样策略完备支持
- 生产级性能优化

## 工程结构

```text
include/
  sllmrf.h
  sllmrf/
    device.h
    gguf.h
    internlm2.h
    operators.h
    tensor.h
    tokenizer.h
src/
  device.cpp
  gguf.cpp
  internlm2.cpp
  internlm2_cuda.cu
  internlm2_cuda.h
  operators.cpp
  operators_cuda.cu
  operators_cuda.h
  tensor.cpp
  tokenizer.cpp
app/
  main.cpp
test/
  test.cpp
```

## 构建

推荐直接使用脚本：

```bash
./compile.sh
```

该脚本会：

- 开启 CUDA 构建选项
- 使用全部 CPU 核心并行编译
- 使用全部 CPU 核心并行运行测试

也可以手动执行：

```bash
cmake -S . -B build -DSLLMRF_ENABLE_CUDA=ON
cmake --build build --parallel "$(getconf _NPROCESSORS_ONLN)"
OMP_NUM_THREADS="$(getconf _NPROCESSORS_ONLN)" ctest --test-dir build --output-on-failure --parallel "$(getconf _NPROCESSORS_ONLN)"
```

## 运行方式

如果仓库下存在 `models/internlm2-1_8-F16.gguf`，可以直接运行：

```bash
./build/app/sllmrf_cli --prompt "Hello world"
```

显式指定模型路径：

```bash
./build/app/sllmrf_cli \
  --model ./models/internlm2-1_8-F16.gguf \
  --prompt "Hello world"
```

使用 CPU 生成：

```bash
./build/app/sllmrf_cli \
  --prompt "Hello world" \
  --generate 8 \
  --device cpu
```

使用 CUDA 生成：

```bash
./build/app/sllmrf_cli \
  --prompt "Hello world" \
  --generate 8 \
  --device cuda
```

使用随机采样：

```bash
./build/app/sllmrf_cli \
  --prompt "Hello world" \
  --generate 8 \
  --device cpu \
  --stochastic \
  --temperature 0.8 \
  --seed 42
```

查看执行计划：

```bash
./build/app/sllmrf_cli \
  --prompt "Hello world" \
  --show-plan
```

查看单层 block 输出：

```bash
./build/app/sllmrf_cli \
  --prompt "Hello world" \
  --run-one-block \
  --embedding-preview 12
```

## CLI 功能

CLI 当前支持以下信息展示：

- GGUF 文件摘要
- 模型配置摘要
- execution device
- prompt 的 token ids / token pieces / decoded text / positions
- embedding shape 与 embedding preview
- 单层 block forward 预览
- 多层 rollout 预览
- logits preview
- greedy / stochastic 生成结果
- `generated text` 与 `full text`

可通过 `--help` 查看参数：

```bash
./build/app/sllmrf_cli --help
```

## 验收建议

项目收尾阶段建议至少完成一次下面的验收：

```bash
./compile.sh
./build/test/sllmrf-test
./build/app/sllmrf_cli --prompt "Hello world" --generate 8 --device cpu
./build/app/sllmrf_cli --prompt "Hello world" --generate 8 --device cuda
```

建议重点确认：

- CPU 路径可稳定生成
- CUDA 路径可稳定生成
- 单层 block forward 不报 shape / device mismatch
- 多 token 生成不崩溃

## 已知限制

- 当前项目只面向 `internlm2-1_8-F16.gguf`
- 采样策略目前只有 `greedy` 和基础 `stochastic`
- `top-k`、`top-p`、repetition penalty 等高级采样尚未实现
- CUDA attention 已有原生 kernel，但仍属于“正确性优先、持续优化中”的实现
- 当前更强调推理链路可观察、可解释，而不是极限性能

## 调试与实验记录

项目中的阶段性实验记录保存在：

```text
debug.log
```

其中包含：

- tokenizer 修复记录
- block forward 开发记录
- 多 token 生成记录
- device / CUDA backend 接入记录
- 原生 CUDA kernel 迭代记录
- 实机报错修复记录

## 答辩展示点

这个项目比较适合从以下角度展示：

- GGUF 文件是如何被加载和索引的
- tokenizer 如何把文本变成 token ids
- embedding / decoder block / final norm / lm head / sampling 如何连成完整链路
- KV cache 在自回归生成中的作用
- 为什么同一 prompt 在 stochastic sampling 下会产生不同输出
- 多设备抽象如何支撑 CPU / CUDA 推理

## 测试环境

![neofetch命令展示](docs/测试环境.png)
