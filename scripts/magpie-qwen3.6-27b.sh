#!/usr/bin/env bash
# Magpie on-policy data generation for Qwen3.6-27B: instructions -> responses -> task
# category tags. Same skeleton as the other scripts/magpie-*.sh drivers.
#
# MUST be run with the CWD set to this directory (scripts/): gen_ins.py, gen_res.py and
# unitag.py all open "../configs/model_configs.json" as a relative path.
#
#   bash magpie-qwen3.6-27b.sh                                    # defaults below
#   bash magpie-qwen3.6-27b.sh Qwen/Qwen3.6-27B 10000             # model, total_prompts
#   STAGE=res TIMESTAMP=1755440000 bash magpie-qwen3.6-27b.sh      # re-run one stage
#
# STAGE=ins|res|tag|all (default all). Each stage builds its own vLLM engine, so a
# failed stage can be resumed without repeating the earlier ones -- but pass the
# original TIMESTAMP, since it is what identifies the job's files.
#
# Diversity: --mix_control_tasks all+free rotates over every pre_query_template_* the
# model config defines PLUS the bare unsteered template, within a single run. The
# unsteered shard is deliberately kept as a control: it reproduces the un-nudged
# behaviour in the same job, so scripts/tamia/diversity_report.py can compare them.
set -euo pipefail

model_path=${1:-"Qwen/Qwen3.6-27B"}
# NOTE: --sanitize --drop_unsanitizable discards the ~14% of generations with no complete
# query in them, so the instruction file holds fewer records than total_prompts. Aim ~15%
# high if you need an exact count.
total_prompts=${2:-10000}

# Instruction sampling. Full-entropy: temp 1.0 / top_p 1.0 with top_k left disabled.
# Diversity here comes from the template rotation plus unrestricted sampling; Qwen's
# generation_config.json top_k=20 is NOT applied, because vLLM only falls back to it
# when no SamplingParams object is passed.
ins_topp=${INS_TOPP:-1.0}
ins_temp=${INS_TEMP:-1.0}
ins_max_tokens=${INS_MAX_TOKENS:-256}
# MEASURED, NOT CHOSEN. Qwen3.6 emits EOS as its FIRST token for ~91% of pre-query
# prompts (job 418055: 218/240 empty instructions, finish_reason=stop, stop_reason=None
# i.e. natural EOS; identical under both GDN prefill kernels, so it is the model, not a
# kernel bug). With a system prompt present it is ~100%: 218/220 of the steered shards
# were empty while the bare template gave 20/20 non-empty. min_tokens suppresses EOS and
# stop_token_ids until this many tokens exist, forcing the model to actually write a
# query. Set INS_MIN_TOKENS=0 to reproduce the empty-instruction failure.
#
# 96, not 16. At 16 the model emitted EOS the instant it was allowed to: 148/240
# instructions were exactly 16 tokens and 83% were cut mid-sentence (one mid-WORD). At 96
# it writes real content -- median 522 chars, 59% terminating on their own -- which the
# --sanitize pass then trims back to the leading complete query.
ins_min_tokens=${INS_MIN_TOKENS:-96}
# Forcing the model past its immediate EOS exposes a second failure: it collapses into
# repetition ("tailored to Amazon A10 Algorithm." x5). Qwen's own model card prescribes
# the remedy -- "adjust the presence_penalty parameter between 0 and 2 to reduce endless
# repetitions" -- at the cost of occasional language mixing. 0 disables it.
# 1.5 measured: heavy repetition fell from 7.1% to 0.8% (jobs 418098 vs 418106).
ins_presence_penalty=${INS_PRESENCE_PENALTY:-1.5}
ins_repetition_penalty=${INS_REPETITION_PENALTY:-0}
ins_max_model_len=${INS_MAX_MODEL_LEN:-4096}
control_tasks=${CONTROL_TASKS:-"all+free"}

# Response sampling: the model card's "Instruct (non-thinking) mode" preset.
res_topp=${RES_TOPP:-0.80}
res_temp=${RES_TEMP:-0.7}
res_topk=${RES_TOPK:-20}
res_presence_penalty=${RES_PRESENCE_PENALTY:-1.5}
res_rep=${RES_REP:-1.0}
res_max_tokens=${RES_MAX_TOKENS:-4096}
res_max_model_len=${RES_MAX_MODEL_LEN:-8192}

# Tagging: greedy, and short -- it only has to emit a small JSON object.
tag_max_tokens=${TAG_MAX_TOKENS:-1024}
tag_max_model_len=${TAG_MAX_MODEL_LEN:-4096}
tag_model_path=${TAG_MODEL_PATH:-$model_path}

# System config. Tamia allocates whole nodes, so 4 GPUs is the floor.
device=${DEVICE:-"0,1,2,3"}
tensor_parallel=${TENSOR_PARALLEL:-4}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}
# n is per pre-query template per round, so a round yields n * num_templates.
n=${N:-100}
batch_size=${BATCH_SIZE:-1000}
ins_checkpoint_every=${INS_CHECKPOINT_EVERY:-1}
res_checkpoint_every=${RES_CHECKPOINT_EVERY:-5}
tag_checkpoint_every=${TAG_CHECKPOINT_EVERY:-5}
output_folder=${OUTPUT_FOLDER:-"${SCRATCH:-/scratch/n/niklas}/magpie-data"}
stage=${STAGE:-all}
seed=${SEED:-}

# Cluster/architecture flags. See exp/vllm_compat.py for why each one is here.
engine_flags=(--language_model_only)
[[ "${ENFORCE_EAGER:-1}" == "1" ]] && engine_flags+=(--enforce_eager)
[[ "${DISABLE_CUSTOM_ALL_REDUCE:-1}" == "1" ]] && engine_flags+=(--disable_custom_all_reduce)
# GDN prefill kernel. 'triton' by default, which is NOT the vLLM default ('auto' ->
# FlashInfer on Hopper). Reason: the FlashInfer GatedDeltaNet prefill kernel is
# JIT-compiled, and the compile is paid once per ENGINE BUILD, not once per job --
# measured at 301-772 s across a dozen Qwen3.6 runs in the sibling repos. This pipeline
# builds three engines, so that is 15-40 minutes of pure JIT. Meanwhile this workload is
# decode-bound: instruction prompts are ~60-120 tokens and response prompts a few
# hundred, so prefill kernel choice barely affects throughput.
# Set GDN_PREFILL_BACKEND=auto to get FlashInfer back. Note the kernels are numerically
# different, so do not mix triton- and FlashInfer-generated data inside one comparison.
# Tested and exonerated: jobs 418032 (triton) and 418055 (FlashInfer) produced the same
# ~91% empty-instruction rate, so the kernel is NOT the cause of that failure -- the
# immediate-EOS behaviour is the model's. Don't re-litigate the kernel over it.
engine_flags+=(--gdn_prefill_backend "${GDN_PREFILL_BACKEND:-triton}")
# Prefix caching stays OFF: on this hybrid architecture the block size is forced up to
# ~784 tokens, so the pre-query templates (~60-120 tokens) never fill a block and cache
# nothing, while enabling it silently moves the engine onto an experimental mamba cache
# path. gen_ins.py/gen_res.py default it off; unitag.py needs it turned off explicitly.

timestamp=${TIMESTAMP:-$(date +%s)}

job_name="${model_path##*/}_topp${ins_topp}_temp${ins_temp}_${timestamp}"
job_path="${output_folder}/${job_name}"
mkdir -p "$job_path"

exec > >(tee -a "$job_path/${job_name}.log") 2>&1
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] Pretty name: $job_name"
echo "[magpie.sh] Total Prompts: $total_prompts"
echo "[magpie.sh] Stage(s): $stage"
echo "[magpie.sh] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp, max_tokens=$ins_max_tokens, min_tokens=$ins_min_tokens, control_tasks=$control_tasks"
echo "[magpie.sh] Response Generation Config: temp=$res_temp, top_p=$res_topp, top_k=$res_topk, presence_penalty=$res_presence_penalty, rep=$res_rep, max_tokens=$res_max_tokens (non-thinking)"
echo "[magpie.sh] Tagging Config: tagger=$tag_model_path, mission=classification"
echo "[magpie.sh] System Config: device=$device, n=$n/template, batch_size=$batch_size, tensor_parallel=$tensor_parallel"
echo "[magpie.sh] Engine flags: ${engine_flags[*]}"
echo "[magpie.sh] Output folder: $job_path"
echo "[magpie.sh] Timestamp: $timestamp"

ins_file="$job_path/Magpie_${model_path##*/}_${total_prompts}_${timestamp}_ins.json"
res_file="${ins_file%.json}_res.json"
tag_file="${res_file%.json}_category.json"

run_stage() { [[ "$stage" == "all" || "$stage" == "$1" ]]; }

if run_stage ins; then
    echo "[magpie.sh] Start Generating Instructions..."
    # Only pass the anti-repetition knobs when set, so the default stays full-entropy.
    ins_penalty_flags=()
    [[ "$ins_presence_penalty" != "0" ]] && ins_penalty_flags+=(--presence_penalty "$ins_presence_penalty")
    [[ "$ins_repetition_penalty" != "0" ]] && ins_penalty_flags+=(--repetition_penalty "$ins_repetition_penalty")
    CUDA_VISIBLE_DEVICES=$device python ../exp/gen_ins.py \
        --device "$device" \
        --model_path "$model_path" \
        --total_prompts "$total_prompts" \
        --top_p "$ins_topp" \
        --temperature "$ins_temp" \
        --max_tokens "$ins_max_tokens" \
        --min_tokens "$ins_min_tokens" \
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
        ${ins_penalty_flags[@]+"${ins_penalty_flags[@]}"} \
        "${engine_flags[@]}"
    echo "[magpie.sh] Finish Generating Instructions!"
fi

if run_stage res; then
    echo "[magpie.sh] Start Generating Responses..."
    # --enable_thinking is deliberately NOT passed: Qwen3.6's chat template opens an
    # unclosed <think> block by default, and a reasoning trace would be truncated by
    # --max_tokens before the answer appeared.
    CUDA_VISIBLE_DEVICES=$device python ../exp/gen_res.py \
        --device "$device" \
        --model_path "$model_path" \
        --batch_size "$batch_size" \
        --checkpoint_every "$res_checkpoint_every" \
        --top_p "$res_topp" \
        --temperature "$res_temp" \
        --top_k "$res_topk" \
        --presence_penalty "$res_presence_penalty" \
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
    # --quantization None: unitag.py defaults to fp8, which is not something to try for
    # the first time on a hybrid GatedDeltaNet checkpoint.
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

echo "[magpie.sh] Outputs:"
for f in "$ins_file" "$res_file" "$tag_file"; do
    [[ -f "$f" ]] && echo "  $f  ($(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f") records)"
done

if [[ -f "$tag_file" ]]; then
    echo "[magpie.sh] Diversity report:"
    # The report exits non-zero when the acceptance bar is not met. That is a verdict on
    # the DATA, not a job failure -- the files are written and worth keeping either way,
    # and letting it kill the script under `set -e` marks the whole job FAILED and hides
    # what actually happened. Report the verdict, keep going.
    if python tamia/diversity_report.py "$tag_file"; then
        echo "[magpie.sh] Diversity bar: PASSED"
    else
        echo "[magpie.sh] Diversity bar: NOT MET (see the report above; data still written)"
    fi
fi

echo "[magpie.sh] Done."
