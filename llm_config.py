import os
from dotenv import load_dotenv
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union

import google.generativeai as genai
from huggingface_hub import InferenceClient
from openai import AsyncOpenAI, OpenAI
from PIL import Image
import anthropic
import cohere
import ai21
import replicate
import mistralai
from llama_cpp import Llama
from litellm import completion as litellm_completion
from litellm import acompletion as litellm_acompletion
from groq import Groq
from together import Together

path = Path(__file__).parent / ".env"   
load_dotenv(dotenv_path=path)

class LLMConfig:
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url or self._get_base_url()
        self.params = kwargs
        self.api_key = self._get_api_key(api_key)

    def _get_api_key(self, provided_key: Optional[str]) -> str:
        if provided_key:
            return provided_key
        if self.provider in ["ollama", "llamacpp"]:
            return "not_required"
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "huggingface": "HF_TOKEN",
            "huggingface-openai": "HF_TOKEN",
            "huggingface-text": "HF_TOKEN",
            "gemini": "GENAI_API_KEY",
            "sdxl": "HF_TOKEN",
            "anthropic": "ANTHROPIC_API_KEY",
            "cohere": "COHERE_API_KEY",
            "replicate": "REPLICATE_API_TOKEN",
            "mistral": "MISTRAL_API_KEY",
            "litellm": "LITELLM_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
        }
        env_var = env_var_map.get(self.provider)
        api_key = os.environ.get(env_var) if env_var else None
        if not api_key:
            raise ValueError(f"API key for {self.provider} is not set. Please set the appropriate environment variable or provide it directly.")
        return api_key

    def _get_base_url(self) -> Optional[str]:
        if self.provider == "ollama":
            return "http://localhost:11434/v1"
        return None

class BaseLLM(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._create_client()

    @abstractmethod
    def _create_client(self):
        pass

    @abstractmethod
    def get_response(self, prompt: str) -> Any:
        pass

    @abstractmethod
    async def get_aresponse(self, prompt: str) -> Any:
        pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "parameters": self.config.params
        }
    
class GroqLLM(BaseLLM):
    def _create_client(self):
        return Groq(api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens'),
            temperature=self.config.params.get('temperature', 1),
            top_p=self.config.params.get('top_p', 1),
            stream=False
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens'),
            temperature=self.config.params.get('temperature', 1),
            top_p=self.config.params.get('top_p', 1),
            stream=True
        )
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_streaming_response(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens'),
            temperature=self.config.params.get('temperature', 1),
            top_p=self.config.params.get('top_p', 1),
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class OpenAILLM(BaseLLM):
    def _create_client(self):
        self.sync_client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        self.async_client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        response = self.sync_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **self.config.params
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class GeminiLLM(BaseLLM):
    def _create_client(self):
        genai.configure(api_key=self.config.api_key)
        return genai.GenerativeModel(model_name=self.config.model)

    def get_response(self, prompt: str) -> str:
        generation_config = genai.GenerationConfig(**{k: v for k, v in self.config.params.items() if k in ['temperature', 'max_output_tokens', 'top_p', 'top_k']})
        response = self.client.generate_content(prompt, generation_config=generation_config)
        response.resolve()
        return response.text

    async def get_aresponse(self, prompt: str):
        # Gemini doesn't support async streaming, so we'll simulate it
        response = self.get_response(prompt)
        for chunk in response.split():
            yield chunk
            await asyncio.sleep(0.01)

class SDXLLLM(BaseLLM):
    def _create_client(self):
        return InferenceClient(model=self.config.model, token=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        try:
            image = self.client.text_to_image(prompt, **self.config.params)
            if isinstance(image, Image.Image):
                image_path = f"{prompt[:20].replace(' ', '_')}.jpg"
                image.save(image_path)
                return f"Image saved as {image_path}"
            else:
                return "Failed to generate image"
        except Exception as e:
            return f"Error generating image: {str(e)}"

    async def get_aresponse(self, prompt: str):
        # SDXL doesn't support async streaming, so we'll return the full response
        yield self.get_response(prompt)

class HFOpenAIAPILLM(BaseLLM):
    def _create_client(self):
        base_url = f"https://api-inference.huggingface.co/models/{self.config.model}/v1/"
        self.sync_client = OpenAI(base_url=base_url, api_key=self.config.api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        response = self.sync_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **self.config.params
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class Ollama(BaseLLM):
    def _create_client(self):
        self.sync_client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        self.async_client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        response = self.sync_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        stream = await self.async_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **self.config.params
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class HFTextLLM(BaseLLM):
    def _create_client(self):
        return InferenceClient(model=self.config.model, token=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        parameters = {k: v for k, v in self.config.params.items() if k in ['temperature', 'max_new_tokens', 'top_p', 'top_k', 'tools', 'tool_choice', 'tool_prompt']}
        response = self.client.text_generation(prompt, **parameters)
        if 'tools' in self.config.params:
            return response.choices[0].message.tool_calls[0].function
        else:
            return response

    async def get_aresponse(self, prompt: str):
        parameters = {k: v for k, v in self.config.params.items() if k in ['temperature', 'max_new_tokens', 'top_p', 'top_k', 'tools', 'tool_choice', 'tool_prompt']}
        parameters['stream'] = True
        async for response in self.client.text_generation(prompt, **parameters, stream=True):
            yield response

class AnthropicLLM(BaseLLM):
    def _create_client(self):
        return anthropic.Anthropic(api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        response = self.client.completions.create(
            model=self.config.model,
            prompt=f"Human: {prompt}\n\nAssistant:",
            max_tokens_to_sample=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
        )
        return response.completion

    async def get_aresponse(self, prompt: str):
        async for chunk in self.client.completions.create(
            model=self.config.model,
            prompt=f"Human: {prompt}\n\nAssistant:",
            max_tokens_to_sample=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
            stream=True,
        ):
            yield chunk.completion

class CohereLLM(BaseLLM):
    def _create_client(self):
        return cohere.Client(api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.config.model,
            prompt=prompt,
            max_tokens=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
        )
        return response.generations[0].text

    async def get_aresponse(self, prompt: str):
        response = self.client.generate(
            model=self.config.model,
            prompt=prompt,
            max_tokens=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
            stream=True,
        )
        for chunk in response:
            yield chunk.text

class MistralLLM(BaseLLM):
    def _create_client(self):
        return mistralai.MistralClient(api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        async for chunk in self.client.chat_stream(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
        ):
            yield chunk.choices[0].delta.content

class LlamaCppLLM(BaseLLM):
    def _create_client(self):
        return Llama(model_path=self.config.model, **self.config.params)

    def get_response(self, prompt: str) -> str:
        response = self.client(prompt, max_tokens=self.config.params.get('max_tokens', 300))
        return response['choices'][0]['text']

    async def get_aresponse(self, prompt: str):
        # llama.cpp doesn't support async streaming, so we'll simulate it
        response = self.get_response(prompt)
        for chunk in response.split():
            yield chunk
            await asyncio.sleep(0.01)

class LiteLLM(BaseLLM):
    def _create_client(self):
        # LiteLLM doesn't require a client to be created
        return None

    def get_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = litellm_completion(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        response = await litellm_acompletion(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
        )
        return response.choices[0].message.content

    def get_streaming_response(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        response = litellm_completion(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens', 300),
            temperature=self.config.params.get('temperature', 0.7),
            stream=True
        )
        for part in response:
            yield part.choices[0].delta.content or ""

class ReplicateLLM(BaseLLM):
    def _create_client(self):
        return replicate.Client(api_token=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        output = self.client.run(
            self.config.model,
            input={"prompt": prompt}
        )
        return ''.join(output)

    async def get_aresponse(self, prompt: str):
        for chunk in self.client.run(
            self.config.model,
            input={"prompt": prompt}
        ):
            yield chunk

class TogetherAILLM(BaseLLM):
    def _create_client(self):
        return Together(api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens'),
            temperature=self.config.params.get('temperature', 1),
            top_p=self.config.params.get('top_p', 1),
            stream=False
        )
        return response.choices[0].message.content

    async def get_aresponse(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens'),
            temperature=self.config.params.get('temperature', 1),
            top_p=self.config.params.get('top_p', 1),
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_streaming_response(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.params.get('max_tokens'),
            temperature=self.config.params.get('temperature', 1),
            top_p=self.config.params.get('top_p', 1),
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class LLMFactory:
    @staticmethod
    def create_llm(config: LLMConfig) -> BaseLLM:
        llm_classes = {
            "openai": OpenAILLM,
            "gemini": GeminiLLM,
            "sdxl": SDXLLLM,
            "huggingface-openai": HFOpenAIAPILLM,
            "huggingface-text": HFTextLLM,
            "ollama": Ollama,
            "anthropic": AnthropicLLM,
            "cohere": CohereLLM,
            "replicate": ReplicateLLM,
            "mistral": MistralLLM,
            "llamacpp": LlamaCppLLM,
            "litellm": LiteLLM,
            "groq": GroqLLM,
            "together": TogetherAILLM,
        }
        if config.provider not in llm_classes:
            raise ValueError(f"Unsupported provider: {config.provider}")
        return llm_classes[config.provider](config)

def get_llm(provider: str, model: str, **kwargs) -> BaseLLM:
    config = LLMConfig(provider, model, **kwargs)
    return LLMFactory.create_llm(config)
# Utility functions

def batch_process(llm: BaseLLM, prompts: List[str]) -> List[str]:
    """Process a batch of prompts and return their responses."""
    return [llm.get_response(prompt) for prompt in prompts]

async def batch_process_async(llm: BaseLLM, prompts: List[str]) -> List[str]:
    """Process a batch of prompts asynchronously and return their responses."""
    async def process_prompt(prompt):
        result = ""
        async for chunk in llm.get_aresponse(prompt):
            result += chunk
        return result
    
    return await asyncio.gather(*[process_prompt(prompt) for prompt in prompts])

def compare_responses(llms: List[BaseLLM], prompt: str) -> Dict[str, str]:
    """Compare responses from multiple LLMs for the same prompt."""
    return {llm.get_model_info()['model']: llm.get_response(prompt) for llm in llms}

async def stream_to_file(llm: BaseLLM, prompt: str, filename: str):
    """Stream the LLM response to a file."""
    with open(filename, 'w') as f:
        async for chunk in llm.get_aresponse(prompt):
            f.write(chunk)
            f.flush()

