#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
puppeteer_config="${script_dir}/mermaid-puppeteer.json"
default_input_root="${repo_root}/build/compute_graphs"
output_root="${repo_root}/build/compute_graphs_svg"

usage() {
    cat <<USAGE
Usage: tools/render-mermaid-svg.sh [--output-dir <dir>] [path ...]

Render Mermaid .mmd files to SVG with Mermaid CLI.

Options:
  -o, --output-dir <dir>
          Directory for generated SVG files.
          Defaults to build/compute_graphs_svg.

Arguments:
  path    A .mmd file or a directory containing .mmd files.
          Defaults to build/compute_graphs.

Examples:
  tools/render-mermaid-svg.sh
  tools/render-mermaid-svg.sh build/compute_graphs/modules
  tools/render-mermaid-svg.sh build/compute_graphs/dataflow.mmd
  tools/render-mermaid-svg.sh -o build/compute_graph_svgs build/compute_graphs
USAGE
}

declare -a targets=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --output-dir|-o)
            if [[ "$#" -lt 2 ]]; then
                echo "error: --output-dir requires a directory" >&2
                exit 1
            fi
            output_root="$2"
            shift 2
            ;;
        --)
            shift
            targets+=("$@")
            break
            ;;
        -*)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            targets+=("$1")
            shift
            ;;
    esac
done

if ! command -v mmdc >/dev/null 2>&1; then
    echo "error: Mermaid CLI 'mmdc' was not found in PATH" >&2
    echo "hint: install @mermaid-js/mermaid-cli, then run this script again" >&2
    exit 127
fi

if [[ ! -f "${puppeteer_config}" ]]; then
    echo "error: Puppeteer config not found: ${puppeteer_config}" >&2
    exit 1
fi

mkdir -p -- "${output_root}"

relative_output_path() {
    local input_path="$1"
    local base_path="$2"
    local absolute_input
    local absolute_base
    local relative_path

    absolute_input="$(realpath -m -- "${input_path}")"
    absolute_base="$(realpath -m -- "${base_path}")"

    if [[ "${absolute_input}" == "${absolute_base}/"* ]]; then
        relative_path="${absolute_input#"${absolute_base}/"}"
    elif [[ "${absolute_input}" == "${absolute_base}" ]]; then
        relative_path="$(basename -- "${absolute_input}")"
    else
        relative_path="$(basename -- "${absolute_input}")"
    fi

    printf '%s/%s\n' "${output_root}" "${relative_path%.*}.svg"
}

render_file() {
    local input_path="$1"
    local base_path="$2"
    local output_path

    output_path="$(relative_output_path "${input_path}" "${base_path}")"
    mkdir -p -- "$(dirname -- "${output_path}")"

    echo "render: ${input_path} -> ${output_path}"
    mmdc \
        -i "${input_path}" \
        -o "${output_path}" \
        -p "${puppeteer_config}" \
        -b white
}

if [[ "${#targets[@]}" -eq 0 ]]; then
    targets=("${default_input_root}")
fi

for target in "${targets[@]}"; do
    if [[ -d "${target}" ]]; then
        while IFS= read -r -d '' file; do
            render_file "${file}" "${default_input_root}"
        done < <(find "${target}" -type f -name '*.mmd' -print0 | sort -z)
    elif [[ -f "${target}" ]]; then
        if [[ "${target}" != *.mmd ]]; then
            echo "skip non-Mermaid file: ${target}" >&2
            continue
        fi
        render_file "${target}" "${default_input_root}"
    else
        echo "error: path not found: ${target}" >&2
        exit 1
    fi
done
