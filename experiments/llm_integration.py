"""
LLM Integration for CAF Experiments
====================================
Real LLM implementations using Hugging Face transformers for local GPU inference.

Supports:
- Llama 2/3 models via transformers
- Local GPU inference (A40, etc.)
- Constraint injection into prompts
- Configurable generation parameters
"""

import torch
from typing import Optional, List
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

from experiments.caf_algorithm import InferenceLayer


@dataclass
class LLMConfig:
    """Configuration for Hugging Face LLM."""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"  # or Llama-3-8b-Instruct
    device: str = "cuda"  # Use GPU
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    load_in_8bit: bool = False  # Set True for memory efficiency
    load_in_4bit: bool = False  # Set True for even more efficiency
    trust_remote_code: bool = False


class HuggingFaceLlamaLayer(InferenceLayer):
    """
    Real Inference Layer using Hugging Face Llama models.

    Loads Llama model locally on GPU and generates responses with
    optional constraint injection.

    Example models:
    - meta-llama/Llama-2-7b-chat-hf (requires HF token, gated)
    - meta-llama/Llama-3-8B-Instruct (newer, better)
    - meta-llama/Llama-2-13b-chat-hf (larger, more capable)
    - NousResearch/Llama-2-7b-chat-hf (ungated alternative)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the Llama model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")
        print(f"Device: {self.config.device}")

        # Check GPU availability
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.config.device = "cpu"

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Configure quantization for memory efficiency
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.float16 if self.config.device == "cuda" else torch.float32,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.config.device

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if quantization_config else self.config.device,
        )

        print(f"Model loaded successfully on {self.config.device}")

    def _format_prompt(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """
        Format prompt for Llama chat model with optional constraints.

        Uses Llama's chat template format:
        [INST] <<SYS>>
        {system_message}
        <</SYS>>

        {user_message} [/INST]
        """
        system_message = """You are a helpful AI assistant that provides accurate, logically consistent responses grounded in factual knowledge.
Your responses should be clear, well-reasoned, and avoid contradictions."""

        # Add constraints if provided
        if constraints and len(constraints) > 0:
            system_message += "\n\nIMPORTANT CONSTRAINTS to follow:"
            for i, constraint in enumerate(constraints, 1):
                system_message += f"\n{i}. {constraint}"

        # Format using Llama chat template
        # For Llama 2
        if "Llama-2" in self.config.model_name:
            formatted_prompt = f"""[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]"""
        # For Llama 3 (uses different format)
        elif "Llama-3" in self.config.model_name or "llama-3" in self.config.model_name.lower():
            # Llama 3 uses a different chat template
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for other models
            formatted_prompt = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"

        return formatted_prompt

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        """
        Generate response using Llama model.

        Args:
            prompt: Input prompt
            constraints: Optional list of constraints to inject

        Returns:
            Generated response text
        """
        formatted_prompt = self._format_prompt(prompt, constraints)

        # Generate (suppress max_length to avoid conflicts with max_new_tokens)
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=self.config.max_new_tokens,
            max_length=None,  # Disable max_length to avoid warning
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            num_return_sequences=1,
            return_full_text=False,  # Only return generated text
        )

        # Extract generated text
        response = outputs[0]["generated_text"]

        # Clean up response (remove any residual prompt artifacts)
        response = response.strip()

        return response

    def __del__(self):
        """Clean up GPU memory when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OpenSourceLlamaLayer(InferenceLayer):
    """
    Alternative implementation using ungated open-source Llama variants.

    Uses models that don't require Meta's gated access:
    - NousResearch/Llama-2-7b-chat-hf
    - NousResearch/Llama-2-13b-chat-hf
    - teknium/OpenHermes-2.5-Mistral-7B (Mistral-based alternative)
    """

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-chat-hf",
        device: str = "cuda",
        load_in_4bit: bool = False
    ):
        config = LLMConfig(
            model_name=model_name,
            device=device,
            load_in_4bit=load_in_4bit
        )
        self.hf_layer = HuggingFaceLlamaLayer(config)

    def generate(
        self,
        prompt: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        return self.hf_layer.generate(prompt, constraints)


# Convenience factory function
def create_llama_layer(
    model_size: str = "7b",
    use_4bit: bool = False,
    use_8bit: bool = False,
    open_source: bool = True
) -> InferenceLayer:
    """
    Factory function to create a Llama inference layer.

    Args:
        model_size: "7b", "8b", or "13b"
        use_4bit: Use 4-bit quantization (saves memory)
        use_8bit: Use 8-bit quantization
        open_source: Use ungated open-source variants

    Returns:
        Configured HuggingFaceLlamaLayer instance
    """
    if open_source:
        # Use NousResearch ungated models
        model_map = {
            "7b": "NousResearch/Llama-2-7b-chat-hf",
            "13b": "NousResearch/Llama-2-13b-chat-hf",
        }
        model_name = model_map.get(model_size, model_map["7b"])
    else:
        # Use official Meta models (requires HF token and access)
        model_map = {
            "7b": "meta-llama/Llama-2-7b-chat-hf",
            "8b": "meta-llama/Llama-3-8B-Instruct",
            "13b": "meta-llama/Llama-2-13b-chat-hf",
        }
        model_name = model_map.get(model_size, model_map["7b"])

    config = LLMConfig(
        model_name=model_name,
        device="cuda",
        load_in_4bit=use_4bit,
        load_in_8bit=use_8bit,
    )

    return HuggingFaceLlamaLayer(config)


if __name__ == "__main__":
    """Demo of LLM integration."""
    print("=" * 60)
    print("Testing Llama Integration")
    print("=" * 60)

    # Create inference layer (using 4-bit for memory efficiency)
    print("\nInitializing Llama model...")
    llm = create_llama_layer(model_size="7b", use_4bit=True, open_source=True)

    # Test basic generation
    print("\n" + "=" * 60)
    print("Test 1: Basic Generation")
    print("=" * 60)
    prompt1 = "Explain why rain makes roads slippery in 2-3 sentences."
    response1 = llm.generate(prompt1)
    print(f"Prompt: {prompt1}")
    print(f"Response: {response1}")

    # Test generation with constraints
    print("\n" + "=" * 60)
    print("Test 2: Generation with Constraints")
    print("=" * 60)
    prompt2 = "Explain how photosynthesis works."
    constraints = [
        "Focus on the chemical equation",
        "Mention chlorophyll",
        "Keep response under 3 sentences"
    ]
    response2 = llm.generate(prompt2, constraints)
    print(f"Prompt: {prompt2}")
    print(f"Constraints: {constraints}")
    print(f"Response: {response2}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
