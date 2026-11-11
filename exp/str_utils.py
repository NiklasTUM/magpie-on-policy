import re

def input_difficulty_rating(input):
    user_message = f'''
# Instruction 

You first need to identify the given user intent and then label the difficulty level of the user query based on the content of the user query.

## User Query
```
{input}
```

## Output Format
Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query.
Then, rate the difficulty level of the user query as `very easy`, `easy`, `medium`, `hard`, or `very hard`.

Now, please output the user intent and difficulty level below in a json format by filling in the placeholders in []:
```
{{   
    "intent": "The user wants to [....]",
    "knowledge": "To solve this problem, the models need to know [....]",
    "difficulty": "[very easy/easy/medium/hard/very hard]"
}}
```
'''
    return user_message

def input_classification(input):
    user_message = f'''
# Instruction

Please label the task tags for the user query.

## User Query
```
{input}
```

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and select the most relevant task tag from the list below.

all_task_tags = [
    "Information seeking",  # Users ask for specific information or facts about various topics.
    "Reasoning",  # Queries require logical thinking, problem-solving, or processing of complex ideas.
    "Planning",  # Users need assistance in creating plans or strategies for activities and projects.
    "Editing",  # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
    "Coding & Debugging",  # Users seek help with writing, reviewing, or fixing code in programming.
    "Math",  # Queries related to mathematical concepts, problems, and calculations.
    "Role playing",  # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
    "Data analysis",  # Requests involve interpreting data, statistics, or performing analytical tasks.
    "Creative writing",  # Users seek assistance with crafting stories, poems, or other creative texts. 
    "Advice seeking",  # Users ask for recommendations or guidance on various personal or professional issues.
    "Brainstorming",  # Involves generating ideas, creative thinking, or exploring possibilities. 
    "Others"  # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]

## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
```
{{ 
    "primary_tag": "<primary tag>",
    "other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```
'''
    return user_message

def input_quality_rating(input):
    user_message = f'''
# Instruction

You need to rate the quality of the user query based on its clarity, specificity, and coherence.

The rating scale is as follows:

- very poor: The query is unclear, vague, or incoherent. It lacks essential information and context.
- poor: The query is somewhat unclear or lacks important details. It requires significant clarification.
- average: The query is moderately clear and specific. It may require some additional information for a complete understanding.
- good: The query is clear, specific, and mostly well-formed. It provides sufficient context for understanding the user's intent.
- excellent: The query is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive response.

## User Query
```
{input}
```

## Output Format
Given the user query, you first need to give an assesement, highlighting the strengths and/or weaknesses of the user query.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
```
{{   
    "explanation": "[...]",
    "input_quality": "[very poor/poor/average/good/excellent]"
}}
```
'''
    return user_message


# For statistcal analysis
MATH_KEYWORDS = [
    "radius", "diameter", "circumference", "perimeter", "triangle", "rectangle",
    "algebra", "geometry", "calculus", "trigonometry", "set theory", "linear regression",
    "number theory", "graph theory", "topology", "differential equations", "integral", "derivative",
    "logarithm", "exponent", "polynomial", "quadratic", "vector", "matrix", "determinant", "eigenvalue",
    "eigenvector", "complex number", "real number", "rational number", "irrational number", "prime number",
    "factorial", "permutation", "combination", "binomial", "continuity", "eigen vector",
    "domain", "range", "inverse", "composition", "angle", "radian", "sine",
    "cosine", "tangent", "cotangent", "secant", "cosecant", "pythagorean theorem", "euclidean geometry",
    "non-euclidean geometry", "hyperbolic geometry", "elliptic geometry", "fractal", "chaos theory",
    "golden ratio", "fibonacci sequence", "pascal's triangle", "binomial theorem", "fourier series",
    "laplace transform", "z-transform", "normal distribution", "poisson distribution", "chi-square distribution",
    "Student's t-distribution", "confidence interval", "hypothesis testing", "p-value", "bayes' theorem",
    "law of large numbers", "central limit theorem", "regression analysis", "correlation coefficient",
    "covariance", "ANOVA", "chi-square test", "t-test", "F-test", "Mann-Whitney U test", "Kolmogorov-Smirnov test",
    "Wilcoxon signed-rank test", "Kruskal-Wallis test", "Fisher's exact test", "McNemar's test", "Spearman's rank correlation",
    "Kendall's tau", "Pearson's correlation", "linear programming", "integer programming", "dynamic programming",
]

CODE_KEYWORDS = [
    "Python", "Java", "in C", "C++", "C#", "JavaScript", "Ruby", "PHP", "Swift", "Objective-C",
    "Kotlin", "in Go", "Rust", "Lua", "Perl", "MATLAB", "Fortran", "Julia", "TypeScript", 
    "F#", "Bash", "PowerShell", "SQL", "HTML", "CSS", "XML", "YAML", "JSON", 
    "Markdown", "LaTeX", "Assembly", "VBA", "Pascal",
    "in D", "in R", "Solidity", "Verilog", "VHDL", "Visual Basic", "BASIC",
    "\n\n```", "\n```", "```", "import ", "def", "elif", "#include"
]


def find_mcq_end(string):
    patterns = [
        r'\([A-D]\)',
        r'[A-D][\.\)]',
        r'[a-d][\.\)]',
    ]
    
    combined_pattern = '|'.join(patterns)
    matches = list(re.finditer(combined_pattern, string, re.IGNORECASE))
    
    if len(matches) >= 4:
        last_match = matches[-1]
        return True, last_match.end()
    
    return False, -1

def find_next_newline(string):
    single_newline = string.find('\n')
    double_newline = string.find('\n\n')
    
    if single_newline == -1 and double_newline == -1:
        return -1
    elif single_newline == -1:
        return double_newline
    elif double_newline == -1:
        return single_newline
    else:
        return min(single_newline, double_newline)

def generate_variants(prefixes):
    variants = []
    for word in prefixes:
        clean_word = word.lstrip('#* ').rstrip(':')
        variants.append(f"{clean_word}:")
        variants.append(f"**{clean_word}**")
        variants.append(f"## {clean_word}:")
    
    return variants

def remove_prefix(string):
    # Remove numbers and alphabets with periods or brackets at the beginning
    string = re.sub(r'^(\d+\.|\w+\))\s*', '', string).strip()
    
    # Remove prefixes like "Task:", "Prompt:", "Question:"
    prefixes = [
    "Task", "Prompt", "Original Prompt", "Question", "The Problem", "Problem",
    "Scenario", "The senario", "Situation", "The situation", "Context",
    "Challenge", "Query", "Request", "Instructions", "Instruction",
    "Descriptions", "Description"
    ]
    prefixes = generate_variants(prefixes)

    prefix_patterns = [re.escape(prefix) for prefix in prefixes]
    
    prefix_patterns.append(r"Question \d+:") # Question 1:, Question 2:, etc.
    prefix_patterns.append(r"Question \d+\.") # Question 1., Question 2., etc.
    prefix_patterns.append(r"Part \d+:") # Part 1:, Part 2:, etc.
    prefix_patterns.append(r"Q\d+\.") # Q1., Q2., etc.
    prefix_patterns.append(r"Q\d+:") # Q1:, Q2:, etc.
    
    combined_pattern = "|".join(prefix_patterns)
    
    match = re.match(f"^({combined_pattern})\s*", string, re.IGNORECASE)
    if match:
        return string[match.end():].strip()
    
    return string


# --- Qwen3.5 / Qwen3.6 sanitizer -------------------------------------------------------
# These models will not role-play the user unprompted: fed a bare pre-query template they
# emit EOS as the FIRST token ~91% of the time (measured, job 418055), and ~100% of the
# time when a system prompt is present. gen_ins.py must therefore force generation with
# --min_tokens, and what comes back needs trimming, because past the forced tokens the
# model reliably does one of three things:
#   1. flips role and answers itself, usually via a <think> block  (54% of generations)
#   2. collapses into repetition -- largely cured at the source by --presence_penalty 1.5,
#      which took heavy repetition from 7.1% to 0.8%
#   3. stacks several separate queries separated by blank lines
# Measured yield of the rules below on 240 real generations (job 418106): 85.8% kept, and
# 100% of what is kept ends on a sentence boundary (vs 22.9% before trimming).
QWEN3_ROLE_MARKERS = re.compile(
    r'<think>|</think>'
    # Literal chat-template markers. These arrive as TEXT, not as the special token ids,
    # so neither stop_token_ids nor skip_special_tokens removes them -- one slipped
    # through into the 12k pilot as '...correctly.<|im_end|>  <|user|> This is ...'.
    r'|<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|user\|>|<\|assistant\|>'
    r'|\n\s*Assistant\s*:|\n\s*assistant\b', re.IGNORECASE)
QWEN3_SENTENCE_SPLIT = re.compile(r'(?<=[.?!。？！])\s')
TERMINAL_PUNCTUATION = '.?!。？！'


def repetition_ratio(text, shingle=6):
    """Fraction of duplicated word n-grams: a cheap detector for repetition collapse."""
    words = text.split()
    if len(words) < shingle * 2:
        return 0.0
    shingles = [' '.join(words[i:i + shingle]) for i in range(len(words) - shingle + 1)]
    return 1 - len(set(shingles)) / len(shingles)


def qwen3_instruction_post_process(instruction, max_repetition=0.25, min_chars=25):
    """Extract the leading user query from a forced Qwen3.x generation.

    Returns (instruction, class_num); an empty instruction means nothing usable was
    found, and the caller should drop the record. class_num records which rule ended the
    instruction, mirroring the gemma/llama branches below.
    """
    class_num = 0

    # 1. Cut where the model stopped being the user.
    parts = QWEN3_ROLE_MARKERS.split(instruction)
    if len(parts) > 1:
        class_num = 1
    instruction = parts[0]
    # A generation may open with a stray closing tag before any content.
    instruction = re.sub(r'^\s*</?think>\s*', '', instruction).strip()

    # 2. Keep only the first query when several are stacked.
    blocks = re.split(r'\n\s*\n', instruction)
    if len(blocks) > 1:
        class_num = class_num or 2
    instruction = blocks[0].strip()

    # 3. Trim to the last COMPLETE sentence. Without this ~77% of instructions end
    #    mid-sentence, because the model was still writing when the token budget ran out.
    if instruction and instruction[-1] in TERMINAL_PUNCTUATION:
        class_num = class_num or 3
    else:
        sentences = QWEN3_SENTENCE_SPLIT.split(instruction)
        if len(sentences) > 1:
            instruction = ' '.join(s.strip() for s in sentences[:-1]).strip()
            class_num = class_num or 4
        else:
            # No sentence boundary anywhere: there is no complete query to salvage.
            return '', 99

    instruction = remove_prefix(instruction)

    if len(instruction) < min_chars:
        return '', 98
    if repetition_ratio(instruction) > max_repetition:
        return '', 97

    return instruction, class_num


_NO_THINK_SUFFIX = {}


def close_open_think(prompt, model_path, enable_thinking):
    """Close the think block a chat template leaves open when thinking should be off.

    Some templates ignore enable_thinking=False and always end the generation prompt with
    an open '<think>' (GLM-5.3-Flash does). The model's own no-thinking form is set per
    model in configs/model_configs.json as 'assistant_no_think_suffix'; for GLM that is
    '</think>', matching how its template renders a past assistant turn without reasoning.
    Models without the key are returned unchanged.
    """
    if enable_thinking:
        return prompt
    if model_path not in _NO_THINK_SUFFIX:
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs",
                            "model_configs.json")
        with open(path, encoding="utf-8") as f:
            entry = json.load(f).get(model_path, {})
        _NO_THINK_SUFFIX[model_path] = entry.get("assistant_no_think_suffix")
    suffix = _NO_THINK_SUFFIX[model_path]
    if suffix and prompt.rstrip().endswith("<think>"):
        return prompt + suffix
    return prompt


# GLM-5.x chat markers. As with Qwen, these can arrive as TEXT inside a completion, where
# neither stop_token_ids nor skip_special_tokens removes them.
GLM5_ROLE_MARKERS = re.compile(
    r'<think>|</think>|<\|assistant\|>|<\|user\|>|<\|system\|>|<\|observation\|>'
    r'|<\|endoftext\|>|\[gMASK\]|<sop>|<eop>'
    r'|\n\s*Assistant\s*:', re.IGNORECASE)


def glm5_instruction_post_process(instruction, finish_reason=None, max_repetition=0.25,
                                  min_chars=25):
    """Sanitize a GLM-5.x pre-query completion. Same return contract as the Qwen3 version.

    Deliberately LIGHTER than qwen3_instruction_post_process. The Qwen rules exist because
    Qwen3.6 had to be forced past an immediate EOS, so every generation ran to the token
    cap and had to be cut back. A model that ends the user turn on its own produces a
    complete instruction, and applying "keep the first paragraph" to it would cut off
    legitimate multi-part prompts (a request followed by the code or text it is about).
    So the paragraph/sentence trimming only runs when the generation was truncated
    (finish_reason == 'length'), or when the model ran on past the user turn without
    emitting the turn-end token.
    """
    class_num = 0
    parts = GLM5_ROLE_MARKERS.split(instruction)
    ran_on = len(parts) > 1
    if ran_on:
        class_num = 1
    instruction = parts[0].strip()

    if ran_on or finish_reason == "length":
        blocks = re.split(r'\n\s*\n', instruction)
        if len(blocks) > 1 and finish_reason == "length":
            # Truncated mid-way: keep the complete leading block(s) only.
            instruction = blocks[0].strip()
            class_num = class_num or 2
        if instruction and instruction[-1] not in TERMINAL_PUNCTUATION:
            sentences = QWEN3_SENTENCE_SPLIT.split(instruction)
            if len(sentences) > 1:
                instruction = ' '.join(s.strip() for s in sentences[:-1]).strip()
                class_num = class_num or 4
            else:
                return '', 99

    instruction = remove_prefix(instruction)
    if len(instruction) < min_chars:
        return '', 98
    if repetition_ratio(instruction) > max_repetition:
        return '', 97
    return instruction, class_num


def instruction_post_process(instruction, model_path, finish_reason=None):
    lowered = model_path.lower()
    if "qwen3.5" in lowered or "qwen3.6" in lowered or "qwen3_5" in lowered:
        return qwen3_instruction_post_process(instruction)
    if "glm-5" in lowered or "glm5" in lowered:
        return glm5_instruction_post_process(instruction, finish_reason)

    if "gemma-2" in model_path.lower():
        # remove the prefix
        instruction = remove_prefix(instruction)
        # find mcq problems
        is_mcq, end_pos = find_mcq_end(instruction)
        assistant_markers = ["Answer:", "Answers:", "The answer is", "Correct answer", "The correct answer", "Answer is", "Explanation:", "Here are some", "Solution Approach:", "Solution:"]
        assistant_pattern = r'(?:' + '|'.join(assistant_markers) + ')s?:'
        assistant_match = re.search(assistant_pattern, instruction) # Exact match, no re.IGNORECASE

        # TODO
        # if is_mcq:
        #     print(f"MCQ detected: {instruction}")
        #     rest_of_string = instruction[end_pos:]
        #     newline_pos = find_next_newline(rest_of_string)
        #     if newline_pos != -1:
        #         instruction = instruction[:end_pos + newline_pos].strip()
        #     else:
        #         instruction = instruction.strip()
        #     print(f"Sanitized MCQ instruction: {instruction}")
        #     class_num = 0

        if instruction.startswith("*"):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("*", "").strip() + '?'
                instruction = remove_prefix(instruction)
                class_num = 1
                return instruction, class_num

        instruction = remove_prefix(instruction)

        if instruction.startswith('\"'):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("\"", "").strip() + '?'
                class_num = 2.1
            else:
                instruction = instruction.split('\n')[0].replace("\"", "").strip()
                instruction = instruction.replace("*", "").strip()  
                class_num = 2.2
        elif instruction.startswith('<b>'):
            instruction.split('\n')[0].replace('</b>', "").replace('<b>', "").strip()
            instruction = instruction.replace("*", "").strip()
            class_num = 3
        elif assistant_match:
            instruction = instruction[:assistant_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 4
        elif instruction.split('\n')[0].strip().endswith(':'):
            colon_pos = instruction.split('\n')[0].strip().rfind(':')
            if '#' in instruction:
                instruction = instruction.split('#')[0].strip()
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 5.1
            elif '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 5.2
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 5.3
        else:
            if '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.1
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 6.2

        # Remove prefixes again
        instruction = remove_prefix(instruction)

        return instruction, class_num

    elif "llama-3" in model_path.lower():
        # remove the prefix
        instruction = remove_prefix(instruction)
        # find mcq problems
        is_mcq, end_pos = find_mcq_end(instruction)
        assistant_markers = ["Answer:", "Answers:", "The answer is", "Correct answer", "The correct answer", "Answer is", "Explanation:", "Here are some", "Solution Approach:", "Solution:"]
        assistant_pattern = r'(?:' + '|'.join(assistant_markers) + ')s?:'
        assistant_match = re.search(assistant_pattern, instruction) # Exact match, no re.IGNORECASE
        # print(f"Assistant match: {assistant_match}")

        step_makers = ["# Step 1", "## Step 1", "### Step 1"]
        step_pattern = r'(?:' + '|'.join(step_makers) + r'):?'
        step_match = re.search(step_pattern, instruction) # Exact match, no re.IGNORECASE
        # print(f"Step match: {step_match}")

        # TODO
        # if is_mcq:
        #     print(f"MCQ detected: {instruction}")
        #     rest_of_string = instruction[end_pos:]
        #     newline_pos = find_next_newline(rest_of_string)
        #     if newline_pos != -1:
        #         instruction = instruction[:end_pos + newline_pos].strip()
        #     else:
        #         instruction = instruction.strip()
        #     print(f"Sanitized MCQ instruction: {instruction}")
        #     class_num = 0

        if instruction.startswith("*"):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("*", "").strip() + '?'
                instruction = remove_prefix(instruction)
                class_num = 1
                return instruction, class_num

        instruction = remove_prefix(instruction)
        if instruction.startswith('\"'):
            if '?' in instruction:
                instruction = instruction.split('?')[0].replace("\"", "").strip() + '?'
                class_num = 2.1
            else:
                instruction = instruction.split('\n')[0].replace("\"", "").strip()
                instruction = instruction.replace("*", "").strip()  
                class_num = 2.2
        elif instruction.startswith('<b>'):
            instruction.split('\n')[0].replace('</b>', "").replace('<b>', "").strip()
            instruction = instruction.replace("*", "").strip()
            class_num = 3
        elif assistant_match:
            instruction = instruction[:assistant_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 4
        elif step_match:
            instruction = instruction[:step_match.start()].strip()
            instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
            class_num = 5
        elif instruction.split('\n')[0].strip().endswith(':'):
            colon_pos = instruction.split('\n')[0].strip().rfind(':')
            if '#' in instruction:
                instruction = instruction.split('#')[0].strip()
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.1
            elif '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 6.2
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 6.3
        else:
            if '?' in instruction:
                instruction = instruction.split('?')[0].strip() + '?'
                instruction = instruction.replace("**", "").strip() if instruction.find("**") == 1 else instruction.strip()
                class_num = 99.1
            else:
                instruction = instruction.split('\n')[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 99.2

        # Remove prefixes again
        instruction = remove_prefix(instruction)

        return instruction, class_num
    
    else:
        return instruction, 0


# Define logits processor for llama-3.1 for de-markdown
def de_md_logits_processor_for_llama3_1(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999 # "#": 2,
        logits[567] = -9999.999 # "##": 567,
        logits[14711] = -9999.999 # "###": 14711,
        logits[827] = -9999.999 # "####": 827,

    return logits


# Define logits processor for flaming initial tokens
def flaming_tokens(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        # Slightly increase the temperature for the first token
        logits = logits / 1.2
    
    return logits