#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

#include "sllmrf.h"

namespace {

[[nodiscard]] sllmrf::ops::OperatorContext parse_execution_device(std::string_view value) {
    if (value == "cpu") {
        return sllmrf::ops::OperatorContext::cpu();
    }
    if (value == "cuda") {
        return sllmrf::ops::OperatorContext::cuda(0);
    }
    constexpr std::string_view prefix = "cuda:";
    if (value.rfind(prefix, 0) == 0U) {
        return sllmrf::ops::OperatorContext::cuda(
            static_cast<uint32_t>(std::stoul(std::string(value.substr(prefix.size())))));
    }

    throw std::invalid_argument("unsupported device string");
}

[[nodiscard]] std::string shell_quote_path(const std::filesystem::path &path) {
    const auto value = path.string();
    std::string quoted = "'";
    for (char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    quoted += "'";
    return quoted;
}

void print_mermaid_render_hint(
    std::string_view graph_label,
    const std::filesystem::path &graph_path,
    const std::filesystem::path &puppeteer_config) {
    const auto render_script = puppeteer_config.parent_path() / "render-mermaid-svg.sh";
    std::cout << graph_label << " mermaid written: " << graph_path << '\n';
    std::cout << "render to SVG under build/compute_graphs_svg:\n";
    std::cout << "  " << shell_quote_path(render_script)
              << " " << shell_quote_path(graph_path) << '\n';
}

void print_mermaid_directory_render_hint(
    const std::filesystem::path &graph_dir,
    const std::filesystem::path &puppeteer_config) {
    const auto render_script = puppeteer_config.parent_path() / "render-mermaid-svg.sh";
    std::cout << "render all module graphs to SVG under build/compute_graphs_svg:\n";
    std::cout << "  " << shell_quote_path(render_script)
              << " " << shell_quote_path(graph_dir) << '\n';
}

bool write_text_file(const std::filesystem::path &path, std::string_view content) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        return false;
    }
    stream << content;
    return static_cast<bool>(stream);
}

[[nodiscard]] uint32_t resolve_runtime_sequence_length(
    const sllmrf::Internlm2Config &config,
    std::size_t prompt_token_count,
    uint32_t max_new_tokens) {
    const auto requested = static_cast<uint64_t>(prompt_token_count) + static_cast<uint64_t>(max_new_tokens);
    const auto bounded = std::min<uint64_t>(
        config.context_length,
        std::max<uint64_t>(1U, requested));
    return static_cast<uint32_t>(bounded);
}

void print_usage(std::string_view program) {
    std::cout
        << "Usage: " << program
        << " [--model <path>] [--prompt <text>] [--no-bos] [--eos] [--embedding-preview <n>] [--run-one-block]"
        << " [--run-all-blocks] [--layers <n>] [--logits-preview <n>] [--generate <n>]"
        << " [--stochastic] [--temperature <value>] [--seed <value>] [--device <cpu|cuda[:n]>]"
        << " [--export-flow-graph <path>] [--export-dataflow-graph <path>]"
        << " [--export-module-graphs <dir>]\n"
        << "       " << program << " --help\n";
}

}  // namespace

int main(int argc, char **argv) {
    std::filesystem::path model_path =
        std::filesystem::path(SLLMRF_PROJECT_ROOT) / "models" / "internlm2-1_8-F16.gguf";
    const auto mermaid_puppeteer_config =
        std::filesystem::path(SLLMRF_PROJECT_ROOT) / "tools" / "mermaid-puppeteer.json";
    std::string prompt;
    bool add_bos = true;
    bool add_eos = false;
    std::size_t embedding_preview = 8U;
    bool show_plan = false;
    bool run_one_block = false;
    bool run_all_blocks = false;
    uint32_t requested_layers = 0U;
    std::size_t logits_preview = 8U;
    uint32_t generate_tokens = 0U;
    bool use_stochastic_sampling = false;
    float temperature = 1.0F;
    uint64_t seed = 0U;
    bool use_seed = false;
    auto execution_context = sllmrf::ops::OperatorContext::cpu();
    std::filesystem::path flow_graph_path;
    std::filesystem::path dataflow_graph_path;
    std::filesystem::path module_graph_dir;

    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--model") {
            if (index + 1 >= argc) {
                std::cerr << "--model requires a path\n";
                return 1;
            }
            model_path = argv[++index];
            continue;
        }
        if (arg == "--prompt") {
            if (index + 1 >= argc) {
                std::cerr << "--prompt requires a value\n";
                return 1;
            }
            prompt = argv[++index];
            continue;
        }
        if (arg == "--no-bos") {
            add_bos = false;
            continue;
        }
        if (arg == "--eos") {
            add_eos = true;
            continue;
        }
        if (arg == "--embedding-preview") {
            if (index + 1 >= argc) {
                std::cerr << "--embedding-preview requires a value\n";
                return 1;
            }
            embedding_preview = static_cast<std::size_t>(std::stoull(argv[++index]));
            continue;
        }
        if (arg == "--show-plan") {
            show_plan = true;
            continue;
        }
        if (arg == "--run-one-block") {
            run_one_block = true;
            continue;
        }
        if (arg == "--run-all-blocks") {
            run_all_blocks = true;
            continue;
        }
        if (arg == "--layers") {
            if (index + 1 >= argc) {
                std::cerr << "--layers requires a value\n";
                return 1;
            }
            requested_layers = static_cast<uint32_t>(std::stoul(argv[++index]));
            continue;
        }
        if (arg == "--logits-preview") {
            if (index + 1 >= argc) {
                std::cerr << "--logits-preview requires a value\n";
                return 1;
            }
            logits_preview = static_cast<std::size_t>(std::stoull(argv[++index]));
            continue;
        }
        if (arg == "--generate") {
            if (index + 1 >= argc) {
                std::cerr << "--generate requires a value\n";
                return 1;
            }
            generate_tokens = static_cast<uint32_t>(std::stoul(argv[++index]));
            continue;
        }
        if (arg == "--stochastic") {
            use_stochastic_sampling = true;
            continue;
        }
        if (arg == "--temperature") {
            if (index + 1 >= argc) {
                std::cerr << "--temperature requires a value\n";
                return 1;
            }
            temperature = std::stof(argv[++index]);
            continue;
        }
        if (arg == "--seed") {
            if (index + 1 >= argc) {
                std::cerr << "--seed requires a value\n";
                return 1;
            }
            seed = static_cast<uint64_t>(std::stoull(argv[++index]));
            use_seed = true;
            continue;
        }
        if (arg == "--device") {
            if (index + 1 >= argc) {
                std::cerr << "--device requires a value\n";
                return 1;
            }
            try {
                execution_context = parse_execution_device(argv[++index]);
            } catch (const std::exception &) {
                std::cerr << "unsupported device: " << argv[index] << '\n';
                return 1;
            }
            continue;
        }
        if (arg == "--export-flow-graph") {
            if (index + 1 >= argc) {
                std::cerr << "--export-flow-graph requires a path\n";
                return 1;
            }
            flow_graph_path = argv[++index];
            continue;
        }
        if (arg == "--export-dataflow-graph") {
            if (index + 1 >= argc) {
                std::cerr << "--export-dataflow-graph requires a path\n";
                return 1;
            }
            dataflow_graph_path = argv[++index];
            continue;
        }
        if (arg == "--export-module-graphs") {
            if (index + 1 >= argc) {
                std::cerr << "--export-module-graphs requires a directory\n";
                return 1;
            }
            module_graph_dir = argv[++index];
            continue;
        }

        std::cerr << "unknown argument: " << arg << '\n';
        print_usage(argv[0]);
        return 1;
    }

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "model file not found: " << model_path << '\n';
        return 1;
    }

    try {
        const auto model = sllmrf::Internlm2Model::load(model_path);
        std::cout << model.gguf().describe();
        std::cout << "model name: " << model.config().model_name << '\n';
        std::cout << "architecture: " << model.config().architecture << '\n';
        std::cout << "context length: " << model.config().context_length << '\n';
        std::cout << "block count: " << model.config().block_count << '\n';
        std::cout << "embedding length: " << model.config().embedding_length << '\n';
        std::cout << "attention heads: " << model.config().attention_head_count << '\n';
        std::cout << "kv heads: " << model.config().attention_head_count_kv << '\n';
        std::cout << "head dimension: " << model.config().head_dimension() << '\n';
        std::cout << "vocab size: " << model.tokenizer().vocab_size() << '\n';
        std::cout << "execution device: " << execution_context.device.to_string() << '\n';
        std::cout << model.describe_pipeline();
        if (show_plan) {
            std::cout << model.build_prefill_plan().describe();
        }
        if (!flow_graph_path.empty()) {
            const auto content = model.build_prefill_plan().to_flow_mermaid(
                "simple-LLMRF flow graph for fast prototype validation: " + model.config().model_name);
            if (!write_text_file(flow_graph_path, content)) {
                std::cerr << "failed to open flow graph output: " << flow_graph_path << '\n';
                return 1;
            }
            print_mermaid_render_hint("flow graph", flow_graph_path, mermaid_puppeteer_config);
        }
        if (!dataflow_graph_path.empty()) {
            const auto content = model.to_dataflow_mermaid(
                "simple-LLMRF dataflow graph for inference learners: " + model.config().model_name);
            if (!write_text_file(dataflow_graph_path, content)) {
                std::cerr << "failed to open dataflow graph output: " << dataflow_graph_path << '\n';
                return 1;
            }
            print_mermaid_render_hint("dataflow graph", dataflow_graph_path, mermaid_puppeteer_config);
        }
        if (!module_graph_dir.empty()) {
            std::filesystem::create_directories(module_graph_dir);
            std::cout << "module graph directory: " << module_graph_dir << '\n';
            for (const auto &document : model.build_module_graphs()) {
                const auto graph_path = module_graph_dir / document.file_name;
                if (!write_text_file(graph_path, document.content)) {
                    std::cerr << "failed to open module graph output: " << graph_path << '\n';
                    return 1;
                }
                print_mermaid_render_hint(document.description, graph_path, mermaid_puppeteer_config);
            }
            print_mermaid_directory_render_hint(module_graph_dir, mermaid_puppeteer_config);
        }

        if (!prompt.empty()) {
            const auto batch = model.prepare_prompt(prompt, add_bos, add_eos);
            std::cout << "prompt: " << prompt << '\n';
            std::cout << "token ids:";
            for (uint32_t token_id : batch.token_ids) {
                std::cout << ' ' << token_id;
            }
            std::cout << '\n';

            std::cout << "token pieces:";
            for (uint32_t token_id : batch.token_ids) {
                std::cout << " [" << model.tokenizer().token(token_id).text << ']';
            }
            std::cout << '\n';
            std::cout << "decoded: " << model.tokenizer().decode(batch.token_ids) << '\n';
            std::cout << "positions:";
            for (uint32_t position : batch.positions) {
                std::cout << ' ' << position;
            }
            std::cout << '\n';

            auto runtime = model.create_runtime(
                resolve_runtime_sequence_length(model.config(), batch.token_ids.size(), generate_tokens),
                execution_context);
            const auto embedding = model.run_prompt_embedding(batch, runtime);
            std::cout << "embedding shape: " << embedding.shape_string() << '\n';
            std::cout << "runtime consumed tokens: " << runtime.consumed_tokens() << '\n';
            std::cout << "kv cache layers: " << runtime.layers().size() << '\n';
            std::cout << "embedding device: " << embedding.placement_string() << '\n';

            const auto preview = std::min<std::size_t>(embedding.values().size(), embedding_preview);
            std::cout << "embedding preview:";
            for (std::size_t index = 0; index < preview; ++index) {
                std::cout << ' ' << std::fixed << std::setprecision(6) << embedding.values()[index];
            }
            std::cout << '\n';

            if (run_one_block) {
                const auto block_output = model.forward_blocks(embedding, runtime, 1U);
                std::cout << "single block output shape: " << block_output.shape_string() << '\n';
                const auto block_preview = std::min<std::size_t>(block_output.values().size(), embedding_preview);
                std::cout << "single block preview:";
                for (std::size_t index = 0; index < block_preview; ++index) {
                    std::cout << ' ' << std::fixed << std::setprecision(6) << block_output.values()[index];
                }
                std::cout << '\n';
            }

            if (run_all_blocks || requested_layers > 0U) {
                auto rollout_runtime = model.create_runtime(
                    resolve_runtime_sequence_length(model.config(), batch.token_ids.size(), generate_tokens),
                    execution_context);
                const auto rollout_hidden = model.forward_prompt(batch, rollout_runtime, requested_layers);
                const auto layers_ran =
                    requested_layers == 0U ? model.config().block_count : std::min(requested_layers, model.config().block_count);
                std::cout << "multi-layer rollout count: " << layers_ran << '\n';
                std::cout << "multi-layer output shape: " << rollout_hidden.shape_string() << '\n';

                const auto rollout_preview = std::min<std::size_t>(rollout_hidden.values().size(), embedding_preview);
                std::cout << "multi-layer preview:";
                for (std::size_t index = 0; index < rollout_preview; ++index) {
                    std::cout << ' ' << std::fixed << std::setprecision(6) << rollout_hidden.values()[index];
                }
                std::cout << '\n';

                const auto logits = model.compute_logits(rollout_hidden);
                const auto shown_logits = std::min<std::size_t>(logits.size(), logits_preview);
                std::cout << "logits preview:";
                for (std::size_t index = 0; index < shown_logits; ++index) {
                    std::cout << ' ' << std::fixed << std::setprecision(6) << logits[index];
                }
                std::cout << '\n';
            }

            if (generate_tokens > 0U) {
                const auto generation = model.generate(
                    prompt,
                    sllmrf::GenerationConfig {
                        .max_new_tokens = generate_tokens,
                        .add_bos = add_bos,
                        .stop_at_eos = true,
                        .layer_count = requested_layers,
                        .execution_context = execution_context,
                        .sampling_strategy = use_stochastic_sampling
                            ? sllmrf::SamplingStrategy::Stochastic
                            : sllmrf::SamplingStrategy::Greedy,
                        .temperature = temperature,
                        .seed = seed,
                        .use_seed = use_seed,
                    });
                std::cout << "sampling strategy: "
                          << (use_stochastic_sampling ? "stochastic" : "greedy") << '\n';
                if (use_stochastic_sampling) {
                    std::cout << "temperature: " << temperature << '\n';
                    if (use_seed) {
                        std::cout << "seed: " << seed << '\n';
                    }
                }
                std::cout << "generated token ids:";
                for (uint32_t token_id : generation.generated_token_ids) {
                    std::cout << ' ' << token_id;
                }
                std::cout << '\n';
                std::cout << "generated token pieces:";
                for (uint32_t token_id : generation.generated_token_ids) {
                    std::cout << " [" << model.tokenizer().token(token_id).text << ']';
                }
                std::cout << '\n';
                std::cout << "generated text: " << generation.generated_text << '\n';
                std::cout << "full text: " << generation.full_text << '\n';
            }
        }

        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
