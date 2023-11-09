# copied, refactored and inspired from the community pipeline from diffusers by Andrew Zhu
# https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion_xl.py

import re
from transformers import CLIPTokenizer


def parse_prompt_attention(text):
    # Define the regular expression pattern
    re_attention = re.compile(
        r"""
            \\\(|\\\)|\\\\|\\|\(|:([+-]?[.\d]+)\)|
            \)|[^\\()\[\]:]+|:
        """,
        re.X,
    )

    res = []
    parenthesis = []

    # Function to multiply range
    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            if res[p][1] != 0.0:
                res[p][1] *= multiplier

    # Iterate over the text
    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            parenthesis.append(len(res))
        elif weight is not None and len(parenthesis) > 0:
            multiply_range(parenthesis.pop(), float(weight))
        elif text == ")" and len(parenthesis) > 0:
            multiply_range(parenthesis.pop(), 1.0)
        else:
            res.append([text, 1.0])

    # Process remaining parenthesis
    for pos in parenthesis:
        multiply_range(pos, 1.0)

    # If no result, add default
    if len(res) == 0:
        res = [["", 1.0]]

    # Merge consecutive segments with same weight
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    # Remove segments with zero weight
    res = [segment for segment in res if segment[1] != 0.0]

    return res


def get_tokens_and_weights(clip_tokenizer: CLIPTokenizer, prompt):
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens, text_weights = [], []

    for word, weight in texts_and_weights:
        token = clip_tokenizer(word).input_ids[1:-1]
        if len(text_tokens) + len(token) > 75:
            break
        text_tokens.extend(token)
        chunk_weights = [weight] * len(token)
        text_weights.extend(chunk_weights)

    return text_tokens, text_weights


def get_prompts_tokens_with_weights(
    clip_tokenizer: CLIPTokenizer, prompt: str, negative_prompt: str
):
    prompt_tokens, prompt_weights = get_tokens_and_weights(clip_tokenizer, prompt)
    neg_prompt_tokens, neg_prompt_weights = get_tokens_and_weights(
        clip_tokenizer, negative_prompt
    )

    return prompt_tokens, prompt_weights, neg_prompt_tokens, neg_prompt_weights


def pad_and_group_tokens_and_weights(*args):
    bos, eos = 49406, 49407

    # Find the maximum length among all token lists
    max_length = max(len(tokens) for tokens, weights in args)

    result = []
    for tokens, weights in args:
        # If tokens is shorter than max_length, pad it with eos
        if len(tokens) < max_length:
            tokens.extend([eos] * (max_length - len(tokens)))
            weights.extend([1.0] * (max_length - len(weights)))
        # adding bos and eos tokens
        tokens = [bos] + tokens + [eos]
        weights = [1.0] + weights + [1.0]
        result.append((tokens, weights))

    return result
