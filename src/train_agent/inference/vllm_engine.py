"""vLLM inference engine with LoRA adapter loading and merging support."""

import os
from typing import Optional, List, Dict

import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train_agent.model_schemas import VLLMConfig, SamplingConfig


class VLLMEngine:
    """vLLM inference engine with LoRA adapter support.

    This class provides:
    - vLLM-based inference for fast parallel generation
    - LoRA adapter loading and merging into base model
    - Batched generation for rollouts
    - GPU memory management
    """

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.llm: Optional[LLM] = None
        self.tokenizer = None
        self._load_engine()

    def _load_engine(self):
        print(f"Loading vLLM engine with model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )

        self.llm = LLM(
            model=self.config.model_name,
            max_model_len=self.config.max_seq_length,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
            seed=self.config.seed,
        )

        print("vLLM engine loaded successfully")

    @staticmethod
    def merge_lora_into_base(
        base_model_name: str,
        lora_adapter_path: str,
        output_path: str,
        dtype: str = "auto"
    ) -> str:
        """Merge LoRA adapter weights into base model and save.

        Args:
            base_model_name: HuggingFace model name or path to base model
            lora_adapter_path: Path to saved LoRA adapter checkpoint
            output_path: Path to save merged model
            dtype: Data type for model loading (auto, float16, bfloat16)

        Returns:
            Path to the merged model
        """
        print(f"Loading base model: {base_model_name}")

        torch_dtype = torch.float16
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "auto":
            torch_dtype = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"Loading LoRA adapter from: {lora_adapter_path}")

        model_with_adapter = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            torch_dtype=torch_dtype,
        )

        print("Merging LoRA weights into base model...")

        merged_model = model_with_adapter.merge_and_unload()

        print(f"Saving merged model to: {output_path}")

        os.makedirs(output_path, exist_ok=True)

        merged_model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)

        print("LoRA merge completed successfully!")

        del base_model
        del model_with_adapter
        del merged_model
        torch.cuda.empty_cache()

        return output_path

    def load_merged_model(self, merged_model_path: str):
        print(f"Reloading vLLM engine with merged model: {merged_model_path}")

        self.config.model_name = merged_model_path

        self._load_engine()

    def generate(
        self,
        prompts: List[str],
        sampling_config: Optional[SamplingConfig] = None
    ) -> List[str]:
        if self.llm is None:
            raise RuntimeError("vLLM engine not initialized")

        if sampling_config is None:
            sampling_config = SamplingConfig()

        sampling_params = SamplingParams(
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            top_k=sampling_config.top_k,
            max_tokens=sampling_config.max_tokens,
            stop=sampling_config.stop,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = [output.outputs[0].text for output in outputs]

        return results

    def generate_with_messages(
        self,
        messages_list: List[List[Dict[str, str]]],
        sampling_config: Optional[SamplingConfig] = None
    ) -> List[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        prompts = []
        for messages in messages_list:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        return self.generate(prompts, sampling_config)

    def get_tokenizer(self):
        return self.tokenizer

    def __del__(self):
        if self.llm is not None:
            del self.llm
            torch.cuda.empty_cache()
