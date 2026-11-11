#!/usr/bin/env bash
# Magpie on-policy data generation for GLM-5.3-Flash (Intel W4A16 AutoRound checkpoint):
# instructions -> responses -> task category tags. Same skeleton and the same category
# rotation as magpie-qwen3.6-27b.sh, so the two datasets differ in the generator model and
# as little else as possible.
#
# MUST be run with the CWD set to this directory (scripts/), and inside the ddo-judge venv
# environment set up by tamia/magpie_glm53.sbatch (vLLM >= 0.30 is required for
# Glm5NextForConditionalGeneration; see ~/ddo/docs/glm53_judge_vllm.md).
#
#   bash magpie-glm5.3-flash.sh                                  # defaults below
#   STAGE=res TIMESTAMP=1755440000 bash magpie-glm5.3-flash.sh    # re-run one stage
#   STAGE=ins|res|tag|quality|all (default all); each stage builds its own vLLM engine.
#
# WHAT IS DELIBERATELY NOT CARRIED OVER FROM QWEN. min_tokens=96 and presence_penalty=1.5
# were fixes for two measured Qwen3.6 failures (ending the user turn as its first token,
# then repetition once forced past it). Whether GLM has either is exactly what the smoke
# run should show, so both default OFF; turn them on only if the smoke data calls for it.
#
# COMPARABILITY. Category templates (configs/model_configs.json) are word-for-word the
# Qwen ones; only the chat markers differ. Tagging and quality judging use one fixed
# third-party model, Gemma-4-31B-it (TAG_MODEL_PATH), never the generator grading its own
# output -- the same revision (439edf5) that judged the Qwen3.6 pilot.
set -euo pipefail

model_path=${1:-"Intel/GLM-5.3-Flash-W4A16-AutoRound"}
total_prompts=${2:-10000}

# Instruction sampling: full entropy, identical to the Qwen run.
ins_topp=${INS_TOPP:-1.0}
ins_temp=${INS_TEMP:-1.0}
# Higher than Qwen's 256. GLM is expected to end the user turn itself, so the cap should
# only ever bind on runaway generations; a tight cap would instead truncate legitimate
# long prompts (a request plus the code or text it is about). Truncation is recorded as
# finish_reason=length and handled by the sanitizer.
ins_max_tokens=${INS_MAX_TOKENS:-1024}
ins_min_tokens=${INS_MIN_TOKENS:-0}
ins_presence_penalty=${INS_PRESENCE_PENALTY:-0}
ins_repetition_penalty=${INS_REPETITION_PENALTY:-0}
ins_max_model_len=${INS_MAX_MODEL_LEN:-4096}
control_tasks=${CONTROL_TASKS:-"all+free"}

# Response sampling: GLM's own generation_config.json (temp 1.0, top_p 0.95), the
# counterpart of using Qwen's model-card preset for Qwen. No thinking: the chat template
# always opens <think> and ignores enable_thinking=False, so gen_res.py closes it with
# the model's native empty block (<think></think>) -- see assistant_no_think_suffix.
res_topp=${RES_TOPP:-0.95}
res_temp=${RES_TEMP:-1.0}
res_rep=${RES_REP:-1.0}
res_max_tokens=${RES_MAX_TOKENS:-4096}
res_max_model_len=${RES_MAX_MODEL_LEN:-8192}

tag_max_tokens=${TAG_MAX_TOKENS:-1024}
tag_max_model_len=${TAG_MAX_MODEL_LEN:-4096}
tag_model_path=${TAG_MODEL_PATH:-"google/gemma-4-31B-it"}
# Quality judge output: unitag.py --tag_mission quality (very poor .. excellent).
quality_max_tokens=${QUALITY_MAX_TOKENS:-512}

device=${DEVICE:-"0,1,2,3"}
tensor_parallel=${TENSOR_PARALLEL:-4}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}
n=${N:-100}
batch_size=${BATCH_SIZE:-1000}
ins_checkpoint_every=${INS_CHECKPOINT_EVERY:-1}
res_checkpoint_every=${RES_CHECKPOINT_EVERY:-5}
tag_checkpoint_every=${TAG_CHECKPOINT_EVERY:-5}
output_folder=${OUTPUT_FOLDER:-"${SCRATCH:-/scratch/n/niklas}/magpie-data"}
stage=${STAGE:-all}
seed=${SEED:-}

# Engine flags, per ~/ddo/docs/glm53_judge_vllm.md: --enforce_eager (graph capture has
# crashed on these nodes) and --disable_custom_all_reduce ('invalid argument' during
# capture). --language_model_only skips the vision tower of this ...ForConditionalGeneration
# checkpoint. No --gdn_prefill_backend: that kernel exists only on the Qwen3.5/3.6 hybrids.
engine_flags=(--language_model_only)
[[ "${ENFORCE_EAGER:-1}" == "1" ]] && engine_flags+=(--enforce_eager)
[[ "${DISABLE_CUSTOM_ALL_REDUCE:-1}" == "1" ]] && engine_flags+=(--disable_custom_all_reduce)

timestamp=${TIMESTAMP:-$(date +%s)}

job_name="${model_path##*/}_topp${ins_topp}_temp${ins_temp}_${timestamp}"
job_path="${output_folder}/${job_name}"
mkdir -p "$job_path"

exec > >(tee -a "$job_path/${job_name}.log") 2>&1
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] Pretty name: $job_name"
echo "[magpie.sh] Total Prompts: $total_prompts"
echo "[magpie.sh] Stage(s): $stage"
echo "[magpie.sh] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp, max_tokens=$ins_max_tokens, min_tokens=$ins_min_tokens, presence_penalty=$ins_presence_penalty, control_tasks=$control_tasks"
echo "[magpie.sh] Response Generation Config: temp=$res_temp, top_p=$res_topp, rep=$res_rep, max_tokens=$res_max_tokens (non-thinking)"
echo "[magpie.sh] Tagging Config: tagger/judge=$tag_model_path, missions=classification,quality"
echo "[magpie.sh] System Config: device=$device, n=$n/template, batch_size=$batch_size, tensor_parallel=$tensor_parallel"
echo "[magpie.sh] Engine flags: ${engine_flags[*]}"
echo "[magpie.sh] Output folder: $job_path"
echo "[magpie.sh] Timestamp: $timestamp"

ins_file="$job_path/Magpie_${model_path##*/}_${total_prompts}_${timestamp}_ins.json"
res_file="${ins_file%.json}_res.json"
tag_file="${res_file%.json}_category.json"
quality_file="${tag_file%.json}_quality.json"

run_stage() { [[ "$stage" == "all" || "$stage" == "$1" ]]; }

if run_stage ins; then
    echo "[magpie.sh] Start Generating Instructions..."
    ins_extra=()
    [[ "$ins_min_tokens" != "0" ]] && ins_extra+=(--min_tokens "$ins_min_tokens")
    [[ "$ins_presence_penalty" != "0" ]] && ins_extra+=(--presence_penalty "$ins_presence_penalty")
    [[ "$ins_repetition_penalty" != "0" ]] && ins_extra+=(--repetition_penalty "$ins_repetition_penalty")
    CUDA_VISIBLE_DEVICES=$device python ../exp/gen_ins.py \
        --device "$device" \
        --model_path "$model_path" \
        --total_prompts "$total_prompts" \
        --top_p "$ins_topp" \
        --temperature "$ins_temp" \
        --max_tokens "$ins_max_tokens" \
        --max_model_len "$ins_max_model_len" \
        --tensor_parallel_size "$tensor_parallel" \
        --gpu_memory_utilization "$gpu_memory_utilization" \
        --mix_control_tasks "$control_tasks" \
        --disable_early_stopping \
        --sanitize \
        --drop_unsanitizable \
        --n "$n" \
        --checkpoint_every "$ins_checkpoint_every" \
        --output_folder "$output_folder" \
        --job_name "$job_name" \
        --timestamp "$timestamp" \
        ${seed:+--seed "$seed"} \
        ${ins_extra[@]+"${ins_extra[@]}"} \
        "${engine_flags[@]}"
    echo "[magpie.sh] Finish Generating Instructions!"
fi

if run_stage res; then
    echo "[magpie.sh] Start Generating Responses..."
    CUDA_VISIBLE_DEVICES=$device python ../exp/gen_res.py \
        --device "$device" \
        --model_path "$model_path" \
        --batch_size "$batch_size" \
        --checkpoint_every "$res_checkpoint_every" \
        --top_p "$res_topp" \
        --temperature "$res_temp" \
        --repetition_penalty "$res_rep" \
        --max_tokens "$res_max_tokens" \
        --max_model_len "$res_max_model_len" \
        --tensor_parallel_size "$tensor_parallel" \
        --gpu_memory_utilization "$gpu_memory_utilization" \
        --input_file "$ins_file" \
        --use_tokenizer_template \
        "${engine_flags[@]}"
    echo "[magpie.sh] Finish Generating Responses!"
fi

if run_stage tag; then
    echo "[magpie.sh] Start Tagging Task Categories..."
    # --quantization None: unitag.py defaults to fp8; the Qwen pilot's Gemma judge ran
    # unquantized, so this matches it.
    CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
        --device "$device" \
        --tag_mission classification \
        --model_path "$tag_model_path" \
        --input_file "$res_file" \
        --batch_size "$batch_size" \
        --checkpoint_every "$tag_checkpoint_every" \
        --max_tokens "$tag_max_tokens" \
        --max_model_len "$tag_max_model_len" \
        --tensor_parallel_size "$tensor_parallel" \
        --gpu_memory_utilization "$gpu_memory_utilization" \
        --quantization None \
        --use_tokenizer_template \
        --disable_prefix_caching \
        --save_as json \
        --offline \
        "${engine_flags[@]}"
    echo "[magpie.sh] Finish Tagging Task Categories!"
fi

if run_stage quality; then
    echo "[magpie.sh] Start Quality Judging..."
    # Same judge and settings as scripts/tamia/quality_tag.sbatch used on the Qwen pilot.
    # Note the Qwen pilot was judged AFTER filter_pairs.py; here it runs on the raw pairs,
    # so compare the two only after filtering this file the same way.
    CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
        --device "$device" \
        --tag_mission quality \
        --model_path "$tag_model_path" \
        --input_file "$tag_file" \
        --batch_size "$batch_size" \
        --checkpoint_every "$tag_checkpoint_every" \
        --max_tokens "$quality_max_tokens" \
        --max_model_len "$tag_max_model_len" \
        --tensor_parallel_size "$tensor_parallel" \
        --gpu_memory_utilization "$gpu_memory_utilization" \
        --quantization None \
        --use_tokenizer_template \
        --disable_prefix_caching \
        --save_as json \
        --offline \
        "${engine_flags[@]}"
    echo "[magpie.sh] Finish Quality Judging!"
fi

echo "[magpie.sh] Outputs:"
for f in "$ins_file" "$res_file" "$tag_file" "$quality_file"; do
    [[ -f "$f" ]] && echo "  $f  ($(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f") records)"
done

if [[ -f "$tag_file" ]]; then
    echo "[magpie.sh] Diversity report:"
    if python tamia/diversity_report.py "$tag_file"; then
        echo "[magpie.sh] Diversity bar: PASSED"
    else
        echo "[magpie.sh] Diversity bar: NOT MET (see the report above; data still written)"
    fi
fi

echo "[magpie.sh] Done."
