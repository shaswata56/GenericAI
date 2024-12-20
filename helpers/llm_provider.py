import os
from langchain_core.language_models import BaseLLM


class LLMProvider:
    temperature: float

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature
    
    def get_llm(self) -> BaseLLM:
        provider = os.getenv("LLM_PROVIDER").lower()
        model = os.getenv("LLM_MODEL")
        
        if provider == "ollama":
            from langchain_ollama.llms import OllamaLLM
            return OllamaLLM(model=model, temperature=self.temperature)
        elif provider == "together":
            from langchain_together import ChatTogether
            return ChatTogether(model=model, temperature=self.temperature)
        elif provider == "openrouter":
            from openrouter import ChatOpenRouter
            return ChatOpenRouter(model_name=model, temperature=self.temperature)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=self.temperature)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=self.temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")