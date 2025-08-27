"""
llm_client.py

This module provides classes to interact with SAP AI Core services for foundation models and orchestration.
"""

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy import GenAIHubProxyClient
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService

import logging

logger = logging.getLogger(__name__)


class CL_Foundation_Service:
    """
    Foundation Model Service for direct LLM interactions.
    Supports multiple models with flexible parameter configuration.
    """
    
    def __init__(self, config):
        """
        Initialize the Foundation Service.
        
        Args:
            config (dict): Configuration containing AI Core credentials
                - aic_client_id: AI Core client ID
                - aic_client_secret: AI Core client secret
                - aic_resource_group: AI Core resource group
                - aic_base_url: AI Core base URL
                - default_model: Default model name (optional)
                - default_temperature: Default temperature (optional)
        """
        self.config = config
        self.aic_client_id = config.get("aic_client_id")
        self.aic_client_secret = config.get("aic_client_secret")
        self.aic_resource_group = config.get("aic_resource_group")
        self.aic_base_url = config.get("aic_base_url")
        self.default_model = config.get("default_model", "gpt-4")
        self.default_temperature = config.get("default_temperature", 0.7)
        self._proxy_client = None

    def get_proxy_client(self):
        """Get or create proxy client (cached for efficiency)."""
        if self._proxy_client is None:
            self._proxy_client = get_proxy_client(
                base_url=self.aic_base_url,
                client_id=self.aic_client_id,
                client_secret=self.aic_client_secret,
                resource_group=self.aic_resource_group
            )
        return self._proxy_client

    def get_llm_client(self, model_name=None, temperature=None, **kwargs):
        """
        Get a configured LLM client.
        
        Args:
            model_name (str, optional): Model name to use
            temperature (float, optional): Temperature for generation
            **kwargs: Additional parameters for ChatOpenAI
            
        Returns:
            ChatOpenAI: Configured LLM client
        """
        model_name = model_name or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        
        llm = ChatOpenAI(
            temperature=temperature,
            proxy_model_name=model_name,
            proxy_client=self.get_proxy_client(),
            **kwargs
        )
        return llm

    def invoke_llm(self, prompt, model_name=None, temperature=None, **kwargs):
        """
        Invoke LLM with given prompt and parameters.
        
        Args:
            prompt (str): Input prompt
            model_name (str, optional): Model name to use
            temperature (float, optional): Temperature for generation
            **kwargs: Additional parameters
            
        Returns:
            str: LLM response content
        """
        try:
            llm = self.get_llm_client(model_name=model_name, temperature=temperature, **kwargs)
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}", exc_info=True)
            raise Exception(f"Error during LLM invocation: {str(e)}")


class CL_Orchestration_Service:
    """
    Orchestration Service for advanced LLM interactions.
    Supports multiple models with orchestration capabilities.
    
    Available orchestration models include:
    - Amazon: nova-premier, nova-pro, nova-lite, nova-micro, titan-text-lite, titan-text-express
    - OpenAI: gpt-5, gpt-5-mini, gpt-5-nano, gpt-4, gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o1, o1-mini, o3, o3-mini, o4-mini
    - Google: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-pro, gemini-2.5-flash
    - Anthropic: claude-3-haiku, claude-3-sonnet, claude-3-opus, claude-3.5-sonnet, claude-3.7-sonnet, claude-4-sonnet, claude-4-opus
    - Other: alephalpha-pharia-1-7b-control, meta--llama3-70b-instruct, meta--llama3.1-70b-instruct, mistralai--mistral-large-instruct, mistralai--mistral-small-instruct, mistralai--pixtral-large-instruct, ibm--granite-13b-chat
    """
    
    def __init__(self, aic_config, default_orch_params=None):
        """
        Initialize the Orchestration Service.
        
        Args:
            aic_config (dict): AI Core configuration
            default_orch_params (dict, optional): Default orchestration parameters
                - orch_url: Orchestration service URL
                - orch_model: Default model name
                - parameters: Default model parameters
        """
        self.aic_auth_url = aic_config["aic_auth_url"]
        self.aic_client_id = aic_config["aic_client_id"]
        self.aic_client_secret = aic_config["aic_client_secret"]
        self.aic_resource_group = aic_config["aic_resource_group"]
        
        # Set default orchestration parameters
        if default_orch_params:
            self.default_orch_url = default_orch_params.get("orch_url")
            self.default_model = default_orch_params.get("orch_model")
            self.default_parameters = default_orch_params.get("parameters", {
                "temperature": 0.5,
                "max_tokens": 20000,
                "top_p": 0.9
            })
        else:
            print("Please provide default_orch_params in the constructor in the .env file.")
        
        self._orch_client = None

    def get_orch_llm_client(self, orch_url=None):
        """
        Get or create orchestration client.
        
        Args:
            orch_url (str, optional): Orchestration service URL
            
        Returns:
            OrchestrationService: Configured orchestration client
        """
        orch_url = orch_url or self.default_orch_url
        if not orch_url:
            raise ValueError("Orchestration URL must be provided either in constructor or method call")
            
        if self._orch_client is None:
            proxy_client = GenAIHubProxyClient(
                base_url=orch_url,
                auth_url=self.aic_auth_url,
                client_id=self.aic_client_id,
                client_secret=self.aic_client_secret,
                resource_group=self.aic_resource_group
            )
            self._orch_client = OrchestrationService(api_url=orch_url, proxy_client=proxy_client)
        return self._orch_client

    def invoke_llm(self, prompt, model_name=None, temperature=None, max_tokens=None, 
                   top_p=None, orch_url=None, **kwargs):
        """
        Invoke LLM with common parameters.
        
        Args:
            prompt (str): Input prompt
            model_name (str, optional): Model name
            temperature (float, optional): Temperature
            max_tokens (int, optional): Maximum tokens
            top_p (float, optional): Top-p sampling
            orch_url (str, optional): Orchestration URL
            **kwargs: Additional parameters
            
        Returns:
            str: LLM response content
        """
        # Build parameters dictionary
        parameters = self.default_parameters.copy()
        if temperature is not None:
            parameters["temperature"] = temperature
        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens
        if top_p is not None:
            parameters["top_p"] = top_p
        parameters.update(kwargs)

        try:
            # Create model configuration
            model_name = model_name or self.default_model
            model_config = LLM(
                name=model_name,
                parameters=parameters
            )
            
            # Create template and configuration
            template = Template(messages=[UserMessage("{{ ?extraction_prompt }}")])
            config = OrchestrationConfig(template=template, llm=model_config)
            
            # Get orchestration client and run
            orch_client = self.get_orch_llm_client(orch_url=orch_url)
            response = orch_client.run(
                config=config,
                template_values=[TemplateValue("extraction_prompt", str(prompt))]
            )
            
            # Extract and return result
            return response.orchestration_result.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in orchestration: {str(e)}", exc_info=True)
            raise Exception(f"Error in orchestration: {str(e)}")
