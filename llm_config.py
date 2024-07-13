from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from openai import OpenAI
import google.generativeai as genai
from huggingface_hub import InferenceClient
from PIL import Image

class APIKeyManager:
    """
    A class for managing API keys.

    This class provides methods to set and get API keys for different providers.
    """

    _instance = None
    _api_keys: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIKeyManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def set_api_key(cls, provider: str, api_key: str):
        """
        Set the API key for a provider.

        Args:
            provider (str): The provider name.
            api_key (str): The API key.

        Returns:
            None
        """
        cls._api_keys[provider.lower()] = api_key

    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """
        Get the API key for a provider.

        Args:
            provider (str): The provider name.

        Returns:
            Optional[str]: The API key if found, None otherwise.
        """
        return cls._api_keys.get(provider.lower())

class LLMConfig:
    """
    Represents the configuration for the LLM (Language Model) service.

    Args:
        provider (str): The provider of the language model.
        model (str): The specific language model to use.
        api_key (Optional[str]): The API key to authenticate the requests. If not provided, it will be retrieved from the APIKeyManager.
        base_url (Optional[str]): The base URL for the API endpoint.
        **kwargs: Additional parameters that can be passed to the language model.

    Attributes:
        provider (str): The provider of the language model.
        model (str): The specific language model to use.
        api_key (str): The API key to authenticate the requests.
        base_url (str): The base URL for the API endpoint.
        params (dict): Additional parameters that can be passed to the language model.

    Raises:
        ValueError: If the API key for the provider is not set.

    """

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or APIKeyManager().get_api_key(self.provider)
        if not self.api_key:
            raise ValueError(f"API key for {self.provider} is not set. Use APIKeyManager.set_api_key() to set it.")
        self.base_url = base_url
        self.params = kwargs

class BaseLLM(ABC):
    """
    Base class for LLM (Language Model) implementations.

    Args:
        config (LLMConfig): The configuration object for the LLM.

    Attributes:
        config (LLMConfig): The configuration object for the LLM.
        client: The client object for the LLM.

    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._create_client()

    @abstractmethod
    def _create_client(self):
        pass

    @abstractmethod
    def get_response(self, prompt: str) -> Any:
        pass

class OpenAILLM(BaseLLM):
    """
    A class representing an OpenAI Language Model.

    This class extends the BaseLLM class and provides methods for interacting with the OpenAI API.

    Attributes:
        config (LLMConfig): The configuration object for the language model.

    Methods:
        _create_client: Creates an OpenAI client using the configuration settings.
        get_response: Generates a response from the language model given a prompt.

    """

    def _create_client(self):
        """
        Creates an OpenAI client using the configuration settings.

        Returns:
            OpenAI: An instance of the OpenAI client.

        """
        return OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def get_response(self, prompt: str) -> str:
        """
        Generates a response from the language model given a prompt.

        Args:
            prompt (str): The prompt for generating the response.

        Returns:
            str: The generated response from the language model.

        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "system", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content

class GeminiLLM(BaseLLM):
    """
    A class representing a Gemini Language Model.

    This class extends the BaseLLM class and provides methods for creating a client,
    getting a response based on a prompt, and handling Gemini-specific parameters.

    Attributes:
        config (LLMConfig): The configuration object for the GeminiLLM.
        client (GenerativeModel): The client for the Gemini Language Model.

    Methods:
        _create_client(): Creates a Gemini Language Model client.
        get_response(prompt: str) -> str: Generates a response based on the given prompt.

    """

    def _create_client(self):
        """
        Creates a Gemini Language Model client.

        Returns:
            GenerativeModel: The Gemini Language Model client.

        """
        genai.configure(api_key=self.config.api_key)
        return genai.GenerativeModel(model_name=self.config.model)

    def get_response(self, prompt: str) -> str:
        """
        Generates a response based on the given prompt.

        Args:
            prompt (str): The prompt for generating the response.

        Returns:
            str: The generated response.

        """
        # Define Gemini-specific parameters
        generation_params = {}
        if 'temperature' in self.config.params:
            generation_params['temperature'] = self.config.params['temperature']
        if 'max_output_tokens' in self.config.params:
            generation_params['max_output_tokens'] = self.config.params['max_output_tokens']
        if 'top_p' in self.config.params:
            generation_params['top_p'] = self.config.params['top_p']
        if 'top_k' in self.config.params:
            generation_params['top_k'] = self.config.params['top_k']
        
        # Create GenerationConfig with non-None parameters
        generation_config = genai.GenerationConfig(**generation_params)
        
        response = self.client.generate_content(prompt, generation_config=generation_config)
        response.resolve()
        return response.text

class SDXLLLM(BaseLLM):
    """
    SDXLLLM class represents a specific implementation of the BaseLLM class.
    It provides methods for creating a client and generating a response based on a prompt.
    """

    def _create_client(self):
        """
        Creates and returns an InferenceClient object based on the model and API key specified in the configuration.
        """
        return InferenceClient(model=self.config.model, token=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        """
        Generates a response based on the given prompt.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response as a string.

        Raises:
            Exception: If there is an error generating the image.
        """
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

class HFLLM(BaseLLM):
    """
    Hugging Face Language Model (HFLLM) class.

    This class represents a Hugging Face Language Model and provides methods for creating a client and getting a response.

    Attributes:
        config (LLMConfig): The configuration object for the HFLLM.
        client (OpenAI): The client object for making API requests.

    Methods:
        _create_client: Creates a client object for making API requests.
        get_response: Gets a response from the language model given a prompt.

    """

    def _create_client(self):
        """
        Creates a client object for making API requests.

        Returns:
            OpenAI: The client object.

        """
        base_url = f"https://api-inference.huggingface.co/models/{self.config.model}/v1/"
        return OpenAI(base_url=base_url, api_key=self.config.api_key)

    def get_response(self, prompt: str) -> str:
        """
        Gets a response from the language model given a prompt.

        Args:
            prompt (str): The prompt for generating the response.

        Returns:
            str: The generated response.

        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "system", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content
    
class Ollama(BaseLLM):
    """
    A class representing an Ollama served Language Model.

    This class extends the BaseLLM class and provides methods for interacting with the Ollama API.

    Attributes:
        config (LLMConfig): The configuration object for the language model.

    Methods:
        _create_client: Creates an Ollama client using the configuration settings.
        get_response: Generates a response from the language model given a prompt.

    """

    def _create_client(self):
        """
        Creates an Ollama client using the configuration settings.

        Returns:
            Ollama: An instance of the Ollama client.

        """
        return Ollama(api_key=self.config.api_key, base_url=self.config.base_url)

    def get_response(self, prompt: str) -> str:
        """
        Generates a response from the language model given a prompt.

        Args:
            prompt (str): The prompt for generating the response.

        Returns:
            str: The generated response from the language model.

        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "system", "content": prompt}],
            **self.config.params
        )
        return response.choices[0].message.content

class LLMFactory:
    """
    Factory class for creating Language Model Managers (LLMs).
    """

    @staticmethod
    def create_llm(config: LLMConfig) -> BaseLLM:
        """
        Creates an instance of a Language Model Manager (LLM) based on the given configuration.

        Args:
            config (LLMConfig): The configuration object specifying the LLM provider.

        Returns:
            BaseLLM: An instance of the LLM based on the provider specified in the configuration.

        Raises:
            ValueError: If the provider specified in the configuration is not supported.
        """
        llm_classes = {
            "openai": OpenAILLM,
            "gemini": GeminiLLM,
            "sdxl": SDXLLLM,
            "huggingface": HFLLM,
            "ollama": Ollama
        }
        if config.provider not in llm_classes:
            raise ValueError(f"Unsupported provider: {config.provider}")
        return llm_classes[config.provider](config)

def get_llm(provider: str, model: str, **kwargs) -> BaseLLM:
    config = LLMConfig(provider, model, **kwargs)
    return LLMFactory.create_llm(config)