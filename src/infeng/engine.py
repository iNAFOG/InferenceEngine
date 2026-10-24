from __future__ import annotations

import time
from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EngineConfig
from .sampler import SamplingParams, sample_next_token, validate_sampling_params

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class InferenceEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.device = self._resolve_device(self.config.device)
        self.dtype = self._resolve_dtype(self.config.dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        '''When it uses padding = True During engine_batch
        The GPT-2 based architechture doesn't have a padding as it was trained without one'''
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left padding so that the LAST position of every row is a real token,
        # that's the position we read the next-token logits from
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_ids = self._resolve_eos_ids()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}")
        if device == "cpu" and dtype == "float16":
            raise ValueError("float16 is only supported on cuda, use float32 or bfloat16 on cpu")
        return DTYPES[dtype]

    def _resolve_eos_ids(self) -> torch.Tensor:
        # some models (Qwen, Llama) have more than one "stop" token
        ids = {self.tokenizer.eos_token_id}
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(gen_eos, int):
            ids.add(gen_eos)
        elif isinstance(gen_eos, list):
            ids.update(gen_eos)
        ids.discard(None)
        return torch.tensor(sorted(ids), device=self.device)

    def _make_generator(self, seed: int | None) -> torch.Generator | None:
        # a per-request generator instead of torch.manual_seed(): seeding the
        # global RNG would affect every other request running in the process
        if seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator

    # ------------------------------------------------------------------
    # The core decode loop. Everything else (single, batch, stream) uses this.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        params: SamplingParams,
    ) -> Iterator[torch.Tensor]:
        """Yields a (batch,) tensor of new token ids, one step at a time.

        With the KV cache on, step 1 (prefill) runs the whole prompt and every
        later step (decode) only feeds the single newest token, re-using the
        keys/values of all earlier tokens from `past_key_values`.
        Without the cache every step re-runs the full sequence from scratch.
        """
        use_cache = self.config.use_kv_cache
        generator = self._make_generator(params.seed)
        past_key_values = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)

        for _ in range(params.max_new_tokens):
            # positions count only real tokens, so a left-padded row still starts at position 0
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

            if use_cache and past_key_values is not None:
                model_input = input_ids[:, -1:]
                position_ids = position_ids[:, -1:]
            else:
                model_input = input_ids

            out = self.model(
                input_ids=model_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                logits_to_keep=1,  # we only need logits for the last position
            )
            if use_cache:
                past_key_values = out.past_key_values

            next_tokens = sample_next_token(out.logits[:, -1, :], params, generator)
            # rows that already hit EOS keep emitting padding
            next_tokens = torch.where(finished, self.pad_token_id, next_tokens)
            yield next_tokens

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens[:, None])], dim=-1)
            if params.ignore_eos:
                continue
            finished |= torch.isin(next_tokens, self.eos_token_ids)
            if finished.all():
                break

    def _sync(self) -> None:
        # cuda runs async, so wait for the gpu before reading the clock
        if self.device == "cuda":
            torch.cuda.synchronize()

    def generate(self, prompt: str, params: SamplingParams) -> dict:
        """Generate text given a prompt.

        Args:
            prompt: Input text to extend.
            params: Sampling parameters (temperature, top_k, seed, etc.).

            Returns:
                dict with keys:
                - text: Full decoded output (prompt + generated).
                - completion: Only the generated part.
                - prompt_tokens: Number of tokens in input.
                - generated_tokens: Number of new generated tokens.
                - total_tokens: Total tokens in output.
                - model: Model name used.
                - latency_ms: Wall-clock generation time.
                - ttft_ms: Time to first token (prefill time).
                - tokens_per_sec: generated_tokens / latency.
            """
        return self.generate_batch([prompt], params)[0]

    def generate_batch(self, prompts: list[str], params: SamplingParams) -> list[dict]:
        validate_sampling_params(params)
        if not prompts:
            return []

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        batch_size = len(prompts)

        self._sync()
        start = time.perf_counter()
        ttft_ms = None
        steps: list[torch.Tensor] = []
        for next_tokens in self._decode(inputs["input_ids"], inputs["attention_mask"], params):
            if ttft_ms is None:
                self._sync()
                ttft_ms = (time.perf_counter() - start) * 1000
            steps.append(next_tokens)
        self._sync()
        latency_ms = (time.perf_counter() - start) * 1000

        generated = torch.stack(steps, dim=1).tolist()  # (batch, steps)
        eos_ids = set() if params.ignore_eos else set(self.eos_token_ids.tolist())

        results = []
        for i in range(batch_size):
            # keep tokens up to and including the first EOS, drop the padding after it
            row = []
            for token in generated[i]:
                row.append(token)
                if token in eos_ids:
                    break

            prompt_ids = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
            prompt_tokens = len(prompt_ids)
            results.append(
                {
                    "text": self.tokenizer.decode(prompt_ids + row, skip_special_tokens=True),
                    "completion": self.tokenizer.decode(row, skip_special_tokens=True),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(row),
                    "total_tokens": prompt_tokens + len(row),
                    "model": self.config.model_name,
                    "latency_ms": round(latency_ms, 2),
                    "ttft_ms": round(ttft_ms, 2),
                    "tokens_per_sec": round(len(row) / (latency_ms / 1000), 2),
                }
            )
        return results

    def stream_generate(self, prompt: str, params: SamplingParams) -> Iterator[dict]:
        validate_sampling_params(params)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        start = time.perf_counter()
        ttft_ms = None
        generated: list[int] = []
        sent_text = ""

        for next_tokens in self._decode(inputs["input_ids"], inputs["attention_mask"], params):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000
            generated.append(int(next_tokens[0]))

            # decode everything generated so far and only send the new part.
            # decoding token-by-token breaks words/emoji that span multiple tokens.
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if text.endswith("�"):
                continue  # half of a multi-byte character, wait for the next token
            delta = text[len(sent_text):]
            sent_text = text
            if delta:
                yield {"type": "token", "text": delta}

        latency_ms = (time.perf_counter() - start) * 1000
        yield {
            "type": "done",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated),
            "total_tokens": prompt_tokens + len(generated),
            "model": self.config.model_name,
            "latency_ms": round(latency_ms, 2),
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "tokens_per_sec": round(len(generated) / (latency_ms / 1000), 2),
        }
