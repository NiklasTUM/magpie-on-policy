"""Shared vLLM engine/sampling kwargs for gen_ins.py, gen_res.py and unitag.py.

Magpie was written against vllm 0.6.5. Two things changed in the versions that can
actually load recent architectures (>= 0.20):

* ``SamplingParams`` no longer accepts a ``logits_processors`` keyword at all, so
  passing it - even as ``None`` - raises ``TypeError``. Sampling kwargs therefore have
  to be built up conditionally rather than passed positionally.
* ``LLM(swap_space=...)`` is deprecated (warned about and dropped).

The engine-side flags collected here exist because of hybrid multimodal checkpoints
such as Qwen3.5/Qwen3.6 (``Qwen3_5ForConditionalGeneration``): a vision tower we never
use, and a GatedDeltaNet prefill kernel that is JIT-compiled on first use.
"""


def engine_kwargs(args):
    """Optional ``LLM(...)`` kwargs, included only when the caller asked for them.

    Keeping them optional means this stays a no-op for the model families the upstream
    scripts already covered.
    """
    kwargs = {}

    if getattr(args, "enforce_eager", False):
        # Skips torch.compile and CUDA graph capture. On multi-GPU nodes this both
        # avoids all-reduce/graph-capture crashes during KV-cache profiling and cuts
        # several minutes off a cold start, at some steady-state throughput cost.
        kwargs["enforce_eager"] = True

    if getattr(args, "disable_custom_all_reduce", False):
        kwargs["disable_custom_all_reduce"] = True

    if getattr(args, "language_model_only", False):
        # Text-only: skip the vision tower and the multimodal startup profiling, which
        # frees that memory for KV cache.
        kwargs["language_model_only"] = True
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    if getattr(args, "gdn_prefill_backend", None):
        # 'auto' resolves to the FlashInfer GDN kernel on Hopper, which is JIT-compiled;
        # 'triton' skips the JIT (worth it for short smoke runs).
        kwargs["additional_config"] = {"gdn_prefill_backend": args.gdn_prefill_backend}

    swap_space = getattr(args, "swap_space", None)
    if swap_space is not None:
        kwargs["swap_space"] = swap_space

    return kwargs


def sampling_kwargs(args, *, logits_processor=None):
    """Optional ``SamplingParams(...)`` kwargs.

    ``logits_processors`` is only included when a processor is actually supplied,
    because on vllm >= 0.20 the keyword itself no longer exists.
    """
    kwargs = {}
    for name in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    if logits_processor is not None:
        kwargs["logits_processors"] = [logits_processor]
    return kwargs
