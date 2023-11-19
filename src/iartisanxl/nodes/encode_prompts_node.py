# pylint: disable=no-member
import re

import torch
from transformers import CLIPTokenizer

from iartisanxl.nodes.node import Node


class EncodePromptsNode(Node):
    REQUIRED_ARGS = [
        "prompt1",
        "prompt2",
        "negative_prompt1",
        "negative_prompt2",
        "tokenizer1",
        "tokenizer2",
        "text_encoder1",
        "text_encoder2",
        "clip_skip",
        "device",
    ]

    def __call__(self):
        # Get tokens with first tokenizer
        prompt_tokens, prompt_weights = self.get_tokens_and_weights(
            self.prompt1, self.tokenizer1
        )
        neg_prompt_tokens, neg_prompt_weights = self.get_tokens_and_weights(
            self.negative_prompt1, self.tokenizer1
        )

        if not self.prompt2:
            prompt_2 = self.prompt1

        if not self.negative_prompt2:
            negative_prompt_2 = self.negative_prompt1

        # Get tokens with second tokenizer
        prompt_tokens_2, prompt_weights_2 = self.get_tokens_and_weights(
            prompt_2, self.tokenizer1
        )
        neg_prompt_tokens_2, neg_prompt_weights_2 = self.get_tokens_and_weights(
            negative_prompt_2, self.tokenizer1
        )

        # pylint: disable=unbalanced-tuple-unpacking
        (
            (prompt_tokens, prompt_weights),
            (neg_prompt_tokens, neg_prompt_weights),
            (prompt_tokens_2, prompt_weights_2),
            (neg_prompt_tokens_2, neg_prompt_weights_2),
        ) = self.pad_and_group_tokens_and_weights(
            (prompt_tokens, prompt_weights),
            (neg_prompt_tokens, neg_prompt_weights),
            (prompt_tokens_2, prompt_weights_2),
            (neg_prompt_tokens_2, neg_prompt_weights_2),
        )

        token_tensor = torch.tensor(
            [prompt_tokens], dtype=torch.long, device=self.device
        )
        weight_tensor = torch.tensor(
            prompt_weights, dtype=torch.float16, device=self.device
        )

        token_tensor_2 = torch.tensor(
            [prompt_tokens_2], dtype=torch.long, device=self.device
        )

        embeds = []
        neg_embeds = []

        # positive prompts - use first text encoder
        prompt_embeds_1 = self.text_encoder1(token_tensor, output_hidden_states=True)

        if self.clip_skip is None:
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]
        else:
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[
                -(self.clip_skip + 2)
            ]

        # positive prompts - use second text encoder
        prompt_embeds_2 = self.text_encoder2(token_tensor_2, output_hidden_states=True)
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds = prompt_embeds_2[0]

        prompt_embeds_list = [
            prompt_embeds_1_hidden_states,
            prompt_embeds_2_hidden_states,
        ]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)

        for j, weight in enumerate(weight_tensor):
            if weight != 1.0:
                token_embedding[j] = (
                    token_embedding[-1]
                    + (token_embedding[j] - token_embedding[-1]) * weight
                )

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)

        neg_token_tensor = torch.tensor(
            [neg_prompt_tokens], dtype=torch.long, device=self.device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_tokens_2], dtype=torch.long, device=self.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weights, dtype=torch.float16, device=self.device
        )

        # negative prompts - use first text encoder
        neg_prompt_embeds_1 = self.text_encoder1(
            neg_token_tensor.to(self.device), output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]

        # negative prompts - use second text encoder
        neg_prompt_embeds_2 = self.text_encoder2(
            neg_token_tensor_2.to(self.device), output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [
            neg_prompt_embeds_1_hidden_states,
            neg_prompt_embeds_2_hidden_states,
        ]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)

        for z, weight in enumerate(neg_weight_tensor):
            if weight != 1.0:
                neg_token_embedding[z] = (
                    neg_token_embedding[-1]
                    + (neg_token_embedding[z] - neg_token_embedding[-1]) * weight
                )

        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding)

        prompt_embeds = torch.cat(embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def get_tokens_and_weights(self, prompt: str, tokenizer: CLIPTokenizer):
        texts_and_weights = self.parse_prompt_attention(prompt)
        text_tokens, text_weights = [], []

        for word, weight in texts_and_weights:
            token = tokenizer(word).input_ids[1:-1]
            if len(text_tokens) + len(token) > 75:
                break
            text_tokens.extend(token)
            chunk_weights = [weight] * len(token)
            text_weights.extend(chunk_weights)

        return text_tokens, text_weights

    def parse_prompt_attention(self, text):
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

    def pad_and_group_tokens_and_weights(self, *args):
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
