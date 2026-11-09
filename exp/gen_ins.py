import torch
import os
import sys
import argparse
import json
import time
import random
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from str_utils import de_md_logits_processor_for_llama3_1, flaming_tokens
from vllm_compat import engine_kwargs, sampling_kwargs
import str_utils

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Instruction Generation Manager.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="We will support more models in the future.")
    
    # Generation Parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling. Omit to disable (vLLM default).")
    parser.add_argument("--min_p", type=float, default=None, help="Min-p sampling. Omit to disable.")
    parser.add_argument("--presence_penalty", type=float, default=None, help="Presence penalty. Omit for 0.0.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty. Omit for 1.0.")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of samples to generate per pre-query template per round. "
                             "With --mix_control_tasks the per-round yield is n * num_templates.")
    parser.add_argument("--repeat", type=int, default=None, help="Number of times to repeat the instruction generation. Only available when total prompts is not specified.")
    parser.add_argument("--total_prompts", type=int, default=1000, help="Total number of prompts to generate. If specified, repeat will be ignored.")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--min_tokens", type=int, default=0,
                        help="Suppress EOS and stop_token_ids until this many tokens have been "
                             "generated. 0 keeps upstream behaviour. Needed for models that end "
                             "the user turn immediately instead of writing a query: Qwen3.6 emits "
                             "EOS as its first token for ~91%% of pre-query prompts, which yields "
                             "empty instructions. This is the supported stand-in for the "
                             "first-token logits processors that vllm >= 0.20 removed.")
    parser.add_argument("--max_model_len", type=int, default=4096)

    # Generation Settings
    parser.add_argument("--early_stopping", type=bool, default=True, help="Stop generation when the \n is generated.")
    parser.add_argument("--disable_early_stopping", action="store_false", dest="early_stopping", help="Disable early stopping.")
    parser.add_argument("--system_prompt", action="store_true", help="Enable system prompt for extracting the input.")
    parser.add_argument("--sanitize", action="store_true",
                        help="Sanitize the generated instructions (Gemma, Llama-3, Qwen3.5/3.6).")
    parser.add_argument("--drop_unsanitizable", action="store_true",
                        help="With --sanitize, omit records whose sanitized instruction came "
                             "back empty instead of carrying them forward. Keeps the response "
                             "stage from spending GPU time on instructions that are unusable. "
                             "The count and the reason codes are reported either way.")
    parser.add_argument("--logits_processor", action="store_true", help="Enable logits processor for the generation.")
    parser.add_argument("--flaming_tokens", action="store_true", help="Enable flaming initial tokens (increase temperature) for more diverse generation.")
    parser.add_argument("--control_tasks", type=str, default=None,
                        help="Steer generation toward one task category, by selecting the model "
                             "config's pre_query_template_<name>. Validated against the keys "
                             "actually present in the model's config entry.")
    parser.add_argument("--mix_control_tasks", type=str, default=None,
                        help="Rotate over several categories within a single run: a comma-separated "
                             "list of names, or 'all' for every pre_query_template_* the model "
                             "config defines. Each record is tagged with its control_task. Pass "
                             "'all+free' (or include 'free' in the list) to also sample the bare, "
                             "unsteered pre_query_template as one of the shards.")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the outputs generated by vllm.")
    parser.add_argument("--skip_special_tokens", type=bool, default=True)

    # System Settings
    parser.add_argument('--engine', default="vllm", type=str, choices=["vllm", "hf"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--swap_space", type=float, default=None,
                        help="Deprecated and ignored by vLLM >= 0.20; only passed when set.")
    parser.add_argument("--enable_prefix_caching", action="store_true",
                        help="Enable vLLM prefix caching. OFF by default: on hybrid "
                             "(mamba/linear-attention) models such as Qwen3.5/3.6 the block size is "
                             "forced up to ~784 tokens, so pre-query templates shorter than one "
                             "block cache nothing, and enabling it silently switches the engine onto "
                             "the experimental 'align' mamba cache path.")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Skip torch.compile and CUDA graph capture.")
    parser.add_argument("--disable_custom_all_reduce", action="store_true",
                        help="Disable vLLM's custom all-reduce (multi-GPU stability on some nodes).")
    parser.add_argument("--language_model_only", action="store_true",
                        help="For multimodal checkpoints, skip the vision tower and its startup "
                             "profiling. Text-only generation.")
    parser.add_argument("--gdn_prefill_backend", type=str, default=None,
                        choices=["auto", "triton", "flashinfer", "cutedsl"],
                        help="GatedDeltaNet prefill kernel for Qwen3.5/3.6-style hybrids. The "
                             "default ('auto' -> flashinfer on Hopper) is JIT-compiled and costs "
                             "several minutes on a cold cache; 'triton' skips the JIT.")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every n repeats.")
    parser.add_argument("--output_folder", type=str, default="../data")
    parser.add_argument("--job_name", type=str, default=None, help="Job Name. Get from the script.")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job. Also used as the random seed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()

TEMPLATE_PREFIX = "pre_query_template_"
# Categories that are not task nudges and so are excluded from --mix_control_tasks all.
NON_CATEGORY_TEMPLATES = {"with_system_prompt"}
# Name given to the bare, unsteered pre_query_template when it is mixed in as a shard.
FREE_SHARD = "free"


def available_categories(model_config):
    return sorted(
        k[len(TEMPLATE_PREFIX):] for k in model_config
        if k.startswith(TEMPLATE_PREFIX) and k[len(TEMPLATE_PREFIX):] not in NON_CATEGORY_TEMPLATES
    )


def resolve_templates(args, model_config):
    """Return the list of (name, pre_query_template) pairs to sample from.

    One pair for the plain/system-prompt/single-category cases; several for
    --mix_control_tasks, which is how a single run produces category-balanced data.
    """
    known = available_categories(model_config)

    def category_template(name):
        key = f"{TEMPLATE_PREFIX}{name}"
        if key not in model_config:
            raise ValueError(
                f"'{args.model_path}' has no {key}. Available categories: "
                f"{known if known else '(none - this model config defines no category templates)'}"
            )
        return model_config[key]

    if args.mix_control_tasks:
        requested = [t.strip() for t in args.mix_control_tasks.split(",") if t.strip()]
        names, include_free = [], False
        for token in requested:
            if token == "all":
                names.extend(known)
            elif token in ("all+free", FREE_SHARD):
                if token == "all+free":
                    names.extend(known)
                include_free = True
            else:
                names.append(token)
        # Deduplicate while preserving order.
        names = list(dict.fromkeys(names))
        if not names and not include_free:
            raise ValueError(f"--mix_control_tasks {args.mix_control_tasks!r} selected no templates.")
        templates = [(name, category_template(name)) for name in names]
        if include_free:
            templates.append((FREE_SHARD, model_config["pre_query_template"]))
        return templates

    if args.control_tasks:
        return [(args.control_tasks, category_template(args.control_tasks))]

    if args.system_prompt:
        print("System prompt enabled. Warning: The system prompt may degrade the performance.")
        key = f"{TEMPLATE_PREFIX}with_system_prompt"
        if key not in model_config:
            raise ValueError(f"'{args.model_path}' has no {key}.")
        return [("with_system_prompt", model_config[key])]

    return [(FREE_SHARD, model_config["pre_query_template"])]


# Main function to control workflow
def main():
    args = get_args()
    print(f"Instruction Generation Manager. Arguments: {args}") # For logging

    # Raise error if sanitization is requested for unsupported models
    if args.sanitize:
        supported = ("gemma", "llama-3", "qwen3.5", "qwen3.6", "qwen3_5")
        if not any(family in args.model_path.lower() for family in supported):
            raise ValueError(
                "Sanitization is only implemented for Gemma, Llama-3 and Qwen3.5/3.6 "
                f"models (str_utils.instruction_post_process); got {args.model_path}."
            )
    
    if args.control_tasks and args.mix_control_tasks:
        raise ValueError("Use either --control_tasks (one category) or --mix_control_tasks (several).")

    ################
    # Resolve the pre-query template(s) from configs/model_configs.json
    # Done BEFORE the engine is built so a bad category name fails in a second rather
    # than after several minutes of model loading.
    ################
    with open("../configs/model_configs.json", "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    if args.model_path not in model_configs:
        raise ValueError(
            f"No entry for '{args.model_path}' in configs/model_configs.json. "
            f"Add one (see the README) or pick one of: {sorted(model_configs)}"
        )
    model_config = model_configs[args.model_path]

    templates = resolve_templates(args, model_config)

    stop_tokens = list(model_config["stop_tokens"]) + list(model_config["stop_tokens_assistant"])
    stop_token_ids = model_config["stop_token_ids"]
    # We found that sometimes LLM will generate responses immediately after the \n token.
    if args.early_stopping:
        stop_tokens.append("\n")

    print(f"Pre-query templates ({len(templates)}):")
    for name, template in templates:
        print(f"  [{name}] {template!r}")
    print(f"Stop tokens: {stop_tokens}")
    print(f"Stop token ids: {stop_token_ids}")

    # Each round samples args.n completions from EVERY template, so one round yields
    # n * len(templates) instructions.
    per_round = args.n * len(templates)
    if args.total_prompts is None:
        if args.repeat is None:
            raise ValueError("Either total prompts or repeat should be specified.")
        args.total_prompts = args.repeat * per_round
    else:
        # If total prompts is specified, repeat will be ignored
        args.repeat = int(np.ceil(args.total_prompts / per_round))
    print(f"Rounds: {args.repeat} x {per_round} instructions/round "
          f"(n={args.n} per template) -> {args.repeat * per_round}")

    # Set the random seed for NumPy
    if args.seed is not None:
        np.random.seed(args.seed)
        # Set the random seed for PyTorch
        torch.manual_seed(args.seed)
        # If you are using CUDA (i.e., a GPU), also set the seed for it
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output file / folder
    output_filename = f"Magpie_{args.model_path.split('/')[-1]}_{args.total_prompts}_{args.timestamp}_ins.json"
    if not args.job_name:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        output_dir = f"{args.output_folder}/{output_filename}"
    else:
        output_dir = f"{args.output_folder}/{args.job_name}/{output_filename}"
    
    # Set the device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # Set generation engine
    if args.engine == "vllm":
        # Create vllm instance
        llm = LLM(model=args.model_path,
                dtype=args.dtype,
                trust_remote_code=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                seed=args.seed if args.seed is not None else args.timestamp,
                enable_prefix_caching=args.enable_prefix_caching,
                **engine_kwargs(args))
    elif args.engine == "hf":
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={'':torch.cuda.current_device()},
            torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        )
    
    
    # Apply logits processors
    if args.logits_processor and args.flaming_tokens:
        raise ValueError("Cannot enable both logits processor and flaming tokens")

    if args.logits_processor and "llama-3.1" in args.model_path.lower():
        logits_processor = de_md_logits_processor_for_llama3_1
        print(f"Logits processor applied: {logits_processor}")
    elif args.flaming_tokens:
        logits_processor = flaming_tokens
        print(f"Logits processor applied: {logits_processor}")
    else:
        logits_processor = None

    # Define sampling parameters. Note that vllm >= 0.20 removed per-request callable
    # logits processors, so the keyword is only passed when one is actually in use -
    # and on those versions --logits_processor / --flaming_tokens will fail loudly.
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        skip_special_tokens=args.skip_special_tokens,
        stop=stop_tokens,
        stop_token_ids=stop_token_ids,
        **sampling_kwargs(args, logits_processor=logits_processor),
    )

    ################
    # Generate outputs
    ################
    results = []
    for rounds in tqdm(range(args.repeat)):
        # (control_task, pre_query_template, raw instruction text) for this round.
        round_items = []

        if args.engine == "vllm":
            # One request per template; vLLM returns them in the order they were passed,
            # so outputs can be attributed back to the template that produced them.
            outputs = llm.generate([template for _, template in templates], sampling_params)
            for (name, template), request_output in zip(templates, outputs):
                for completion in request_output.outputs:
                    round_items.append((name, template, completion.text,
                                        getattr(completion, "finish_reason", None),
                                        getattr(completion, "stop_reason", None)))

        elif args.engine == "hf":
            for name, template in templates:
                input = tokenizer.encode(template, add_special_tokens=False, return_tensors="pt").to(torch.cuda.current_device())
                # Gemma-2 bug, so we cannot set num_return_sequences > 1.
                # Instead, we repeat the input n times.
                inputs = input.repeat(args.n, 1).to(torch.cuda.current_device())
                output = model.generate(inputs,
                                        tokenizer=tokenizer,
                                        do_sample=True,
                                        temperature=args.temperature,
                                        top_p=args.top_p,
                                        max_length=args.max_tokens,
                                        num_return_sequences=1,
                                        )
                # Remove the input from the output
                decoded = tokenizer.batch_decode(output[i][len(inputs[0]):] for i in range(args.n))
                # Stop on the first stop token
                for completion in decoded:
                    for stop_token in stop_tokens:
                        if stop_token in completion:
                            completion = completion[:completion.index(stop_token)]
                    round_items.append((name, template, completion, None, None))

        if args.shuffle:
            random.shuffle(round_items)

        # Save outputs
        for i, (control_task, template, raw, finish_reason, stop_reason) in enumerate(round_items):
            instruction = raw.strip()
            result = {
                "id": rounds * per_round + i,
                "pre_query_template": f"{template}",
                "control_task": control_task,
                "instruction": instruction,
                # Recorded because the failure mode of this whole method is a model that
                # closes the user turn immediately: that shows up as an empty
                # instruction whose stop_reason is the turn-end token id.
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "response": None,
                "created": int(time.time()),
                "gen_input_configs": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "input_generator": f"{args.model_path}",
                    "seed": args.seed,
                },
                "gen_response_configs": None,
            }
            if args.sanitize:
                sanitized_instruction, class_num = str_utils.instruction_post_process(instruction, args.model_path)
                result["raw_instruction"] = instruction
                result["instruction"] = sanitized_instruction
                result["instruction_sanitize_class_num"] = class_num
                if args.drop_unsanitizable and not sanitized_instruction.strip():
                    continue
            results.append(result)

        # Save the checkpoints every args.checkpoint_every rounds
        if rounds % args.checkpoint_every == 0:
            with open(output_dir, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Checkpoint saved. Total prompts: {len(results)}")
    
    # Save the final results
    with open(output_dir, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Instruction generated from {args.model_path}. Total prompts: {len(results)}")

    ################
    # Health summary. Magpie only works if the model continues AS THE USER; when it
    # instead closes the turn immediately the instruction is empty, and without this
    # summary that is invisible until the tagger reports everything as "Others".
    ################
    from collections import Counter
    total = len(results)
    empty = [r for r in results if not r["instruction"].strip()]
    lengths = sorted(len(r["instruction"]) for r in results)
    finish = Counter(r.get("finish_reason") for r in results)
    stop = Counter(r.get("stop_reason") for r in results)
    print(f"[gen_ins.py] empty instructions: {len(empty)}/{total} "
          f"({len(empty) / max(total, 1):.1%})")
    print(f"[gen_ins.py] instruction chars: min={lengths[0]} "
          f"median={lengths[len(lengths) // 2]} max={lengths[-1]}")
    print(f"[gen_ins.py] finish_reason={dict(finish)}")
    print(f"[gen_ins.py] stop_reason={dict(stop)}")
    per_shard = Counter(r["control_task"] for r in results if not r["instruction"].strip())
    if per_shard:
        print(f"[gen_ins.py] empty-by-shard: {dict(per_shard)}")
    if args.sanitize:
        # Reason codes come from str_utils; 97/98/99 mean "nothing usable" (repetitive,
        # too short, no complete sentence), anything else is a successful trim.
        codes = Counter(r.get("instruction_sanitize_class_num") for r in results)
        print(f"[gen_ins.py] sanitize class_num={dict(sorted(codes.items(), key=lambda kv: str(kv[0])))}")
        ends = sum(1 for r in results
                   if r["instruction"].strip()
                   and r["instruction"].strip()[-1] in str_utils.TERMINAL_PUNCTUATION)
        usable = sum(1 for r in results if r["instruction"].strip())
        print(f"[gen_ins.py] usable after sanitize: {usable}/{total} ({usable / max(total, 1):.1%}); "
              f"of those ending on a sentence boundary: {ends}/{max(usable, 1)} "
              f"({ends / max(usable, 1):.1%})")
        if args.drop_unsanitizable:
            print("[gen_ins.py] note: --drop_unsanitizable was on, so unusable generations "
                  "were omitted from this file and the counts above are post-drop.")
    if len(empty) / max(total, 1) > 0.25:
        print("[gen_ins.py] WARNING: over a quarter of instructions are EMPTY. The model is "
              "closing the user turn instead of writing a query, so Magpie's premise is not "
              "holding as configured. Before changing the prompt, rule out a bad prefill "
              "kernel (--gdn_prefill_backend) or a sampling misconfiguration -- empties that "
              "correlate with prompt length point at the kernel, not at the model.")
    for r in results[:5]:
        print(f"[gen_ins.py] sample [{r['control_task']}] {r['instruction']!r}")

# Run the main function
if __name__ == "__main__":
    main()
