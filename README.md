# llm_factory
Got tired of always having to change url / chat formats for various LLM providers (OpenAI, HuggingFace, Ollama), so created classes to more easily use them in my apps.
APIKeyManager uses load_dotenv() so you need to have python-dotenv installed and an .env file created with your api_keys

#_Note:_
_The HuggingFace models are setup to use the HuggingFace Pro inference type, since that is what I use.  You can change this as needed if you want to use InferenceClient. See SDXL for a reference for InferenceClientm but use text_generation instead of text_to_image._

## Sample Usage:

```python
from llm_config import get_llm, APIKeyManager
import os
from dotenv import load_dotenv

load_dotenv()

# Set API keys
APIKeyManager.set_api_key("huggingface", os.environ.get("HF_TOKEN"))
APIKeyManager.set_api_key("gemini", os.environ.get("GENAI_API_KEY"))
APIKeyManager.set_api_key("sdxl", os.environ.get("HF_TOKEN"))

# Hugging Face (using OpenAI interface)
hf_llm = get_llm("huggingface", "meta-llama/Meta-Llama-3-70B-Instruct", temperature=0.7, max_tokens=500)
print("HuggingFace:", hf_llm.get_response("What is the capital of France?"))

# Gemini
gemini_llm = get_llm("gemini", "gemini-1.5-pro", max_output_tokens=100, temperature=0.7)
print("Gemini:", gemini_llm.get_response("What is the capital of Spain?"))

# SDXL
sdxl_llm = get_llm("sdxl", "stabilityai/stable-diffusion-xl-base-1.0")
print("SDXL:", sdxl_llm.get_response("A futuristic cityscape of Tokyo"))
```

## Adding Additional Providers and Response Formats:
Add provider in LLMFactory llm_classes:
```python
        llm_classes = {
            "openai": OpenAILLM,
            "gemini": GeminiLLM,
            "sdxl": SDXLLLM,
            "huggingface": HFLLM
        }
```

Create a class for the provider/response:

```python
class MyProvider(BaseLLM):

  def _create_client(self):
    base_url = "https://whatevertheproviderrequires"
    return <this is where you return whatever is required to create that client.  for example Gemini requires genai.GenerateModel(model_name = self.config.model), while openai may require OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
 def get_response(self, prompt: str) -> str:
    <whatever format response/params your LLM or python library of choice requires here>
    return response
```



