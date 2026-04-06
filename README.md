# simple-LLMRF

`simple-LLMRF` 是一个面向 GGUF 稠密模型的轻量级推理框架实验项目。当前阶段的目标不是“立刻跑完整推理”，而是先把推理链路里最关键、最适合教学和研究的底层组件搭起来，让后续的张量系统、计算图和算子库有稳定基座。

这次重构后的工程重点放在两件事上：

- 把 GGUF 文件读取、元数据解析、张量索引等能力做成清晰的基础库
- 把 `tokenizer -> prompt 准备 -> embedding -> runtime/KV cache` 的推理前半段骨架搭起来，并贴合 `models/internlm2-1_8-F16.gguf`

## 当前能力

- 使用 `mmap` 读取 GGUF 文件
- 解析 GGUF v2/v3 的文件头、metadata、tensor infos
- 使用强类型 `MetadataValue` 表示标量、字符串和数组 metadata
- 为 GGUF 张量提供按名字索引、按行读取、F16/F32 转 float32 的访问接口
- 从 GGUF 中构建 tokenizer，并完成基础编码/解码流程
- 提供 `internlm2` 专用模型配置解析、张量命名布局和 KV cache/runtime 骨架
- 已实现 `prompt -> token ids -> positions -> token embedding` 的最小真实通路
- 已实现第一批 CPU 算子骨架：`RMSNorm`、逐元素加法、逐元素乘法、`SiLU`
- 已接入 OpenMP，多层 rollout 的热点 CPU 路径支持多核并行
- 已实现 `internlm2` prefill execution plan builder，可枚举完整的 block 级执行节点
- 已实现 `output_norm + lm_head(last token)` 接口，可为后续生成逻辑提供 logits 头
- 已实现多层 decoder block rollout，单层 block 包含：`attn_norm -> qkv -> RoPE -> causal attention -> attn residual -> ffn_norm -> SwiGLU -> ffn residual`
- 已实现 `greedy sampling` 与 `stochastic sampling` 的最小自回归生成闭环，可连续输出多个 token
- 提供 CLI 用于查看模型摘要、prompt 分词结果和 embedding 预览
- 提供不依赖真实大模型文件的最小测试夹具

## 工程结构

```text
include/
  sllmrf.h              # 聚合头文件
  sllmrf/
    gguf.h              # GGUF 解析相关 API
    internlm2.h         # internlm2 模型骨架与运行时接口
    operators.h         # CPU 算子骨架
    tensor.h            # 基础 TensorBuffer
    tokenizer.h         # tokenizer API
src/
  gguf.cpp
  internlm2.cpp
  operators.cpp
  tensor.cpp
  tokenizer.cpp
app/
  main.cpp              # CLI 工具
test/
  test.cpp              # 解析/tokenizer/internlm2 骨架测试
```

## 构建

```bash
./compile.sh
```

或者手动执行：

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## 使用示例

如果仓库下存在 `models/internlm2-1_8-F16.gguf`，可以直接运行：

```bash
./build/app/sllmrf_cli --prompt "It's good day for me"
```

也可以显式指定模型路径：

```bash
./build/app/sllmrf_cli \
  --model ./models/internlm2-1_8-F16.gguf \
  --prompt "Hello world"
```

输出会包含：

- GGUF 文件摘要
- internlm2 的关键配置
- 当前推理 pipeline 描述
- 可选完整 execution plan 输出（`--show-plan`）
- prompt 对应的 token ids
- token 片段及重建后的文本
- embedding shape、runtime token 使用量、KV cache 层数
- embedding 数值预览
- 可选单层 block forward 预览（`--run-one-block`）
- 可选 greedy 生成（`--generate <n>`）
- 可选随机采样生成（`--generate <n> --stochastic --temperature 0.8 --seed 42`）
- 生成阶段会同时输出 `generated text` 和 `full text`

## 下一步路线

- 为 internlm2 实现 RMSNorm、线性层、RoPE、attention、SwiGLU MLP
- 对已完成的多层 block forward 和 logits 头做 buffer/算子优化
- 继续补 top-k/top-p 等更完整的随机 sampling，并扩展文本生成体验

## 测试环境

![neofetch命令展示](docs/测试环境.png)
