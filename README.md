# llm_factory
Got tired of always having to change url / chat formats for various LLM providers (OpenAI, HuggingFace, Ollama), so created classes to more easily use them in my apps.
APIKeyManager uses load_dotenv() so you need to have python-dotenv installed and an .env file created with your api_keys in this naming convention.:
```python

OPENAI_API_KEY = "  <openai key>  ",
HF_TOKEN = " <huggingface token>   ",
GENAI_API_KEY = "<google gemini key>",

```
You can add others or change the naming convention in LLMConfig -> _get_api_key -> env_var_map

**_Current Models/APIs:_**

- HuggingFace Inference Client (Text): 'huggingface-text'
- OpenAI: 'openai'
- Gemini: 'gemini'
- SDXL: 'sdxl'
- HuggingFace Inference using OpenAI API: 'huggingface-openai'
- Ollama: 'ollama'

## Sample Usage:

```python
from llm_config import get_llm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face (using OpenAI interface)
hf_llm = get_llm("huggingface-openai", "meta-llama/Meta-Llama-3-70B-Instruct", temperature=0.7, max_tokens=500)
print("HuggingFace (OpenAI API):", hf_llm.get_response("What is the capital of France?"))

# Gemini
gemini_llm = get_llm("gemini", "gemini-1.5-flash", max_output_tokens=100, temperature=0.7)
print("Gemini:", gemini_llm.get_response("What is the capital of Spain?"))

# SDXL
sdxl_llm = get_llm("sdxl", "stabilityai/stable-diffusion-xl-base-1.0")
print("SDXL:", sdxl_llm.get_response("A futuristic cityscape of Tokyo"))

# HuggingFace Text
hf_text_llm = get_llm("huggingface-text", "google/gemma-2b", temperature=0.1, max_tokens=10)
print("HuggingFace Text:", hf_text_llm.get_response("What is the capital of France?"))

# Ollama
ollama_llm = get_llm("ollama", "l3custom", temperature=0.7, max_tokens=500)
print("Ollama:", ollama_llm.get_response("What is the capital of Germany?"))
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



