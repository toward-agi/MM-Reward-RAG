from typing import Dict, Any, Optional
from pathlib import Path
import yaml

class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._prompts_cache = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from yaml file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_prompts(self, prompt_file: str) -> Dict[str, str]:
        """Load prompts from a yaml file."""
        if prompt_file in self._prompts_cache:
            return self._prompts_cache[prompt_file]
            
        prompt_path = Path(self.config_path).parent.parent / prompt_file
        with open(prompt_path, 'r') as f:
            prompts = yaml.safe_load(f)
            
        self._prompts_cache[prompt_file] = prompts
        return prompts
    
    def get_generator_config(self, generator_name: str) -> Dict[str, Any]:
        """Get configuration for a specific generator."""
        if generator_name not in self.generators:
            raise ValueError(f"Generator {generator_name} not found in config")
            
        config = self.generators[generator_name].copy()
        
        # Load prompts if prompt_file is specified
        if 'prompt_file' in config:
            config['prompts'] = self._load_prompts(config['prompt_file'])
            del config['prompt_file']
            
        return config
    
    @property
    def generators(self) -> Dict[str, Any]:
        return self._config.get('generators', {})
    
    @property
    def indexers(self) -> Dict[str, Any]:
        return self._config.get('indexers', {})
    
    @property
    def embeddings(self) -> Dict[str, Any]:
        return self._config.get('embedders', {})
    
    @property
    def retrieval(self) -> Dict[str, Any]:
        return self._config.get('retrieval', {})

    @property
    def critic(self) -> Dict[str, Any]:
        return self._config.get('critic', {})
    
    def get_embedder_config(self, embedder_name: str) -> Dict[str, Any]:
        """Get configuration for a specific embedder."""
        if embedder_name not in self.embeddings:
            raise ValueError(f"Embedder {embedder_name} not found in config")
            
        config = self.embeddings[embedder_name].copy()
        return config
    
    def get_critic_config(self, critic_name: str) -> Dict[str, Any]:
        """Get configuration for the critic."""
        if critic_name not in self.critic:
            raise ValueError(f"Critic {critic_name} not found in config")
            
        config = self.critic[critic_name].copy()
        return config

settings = Config()
