# utils.py
import yaml
import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import importlib
import logging
import json
from typing import TypedDict, Optional, List, Dict, Any
from graph_state import AgentState

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY not found in .env file. Application may not function.")
    # raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

# Initialize OpenAI client
# Use a try-except block for robustness, especially if key might be missing/invalid
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except OpenAIError as e:
    logger.critical(f"Failed to initialize OpenAI client: {e}")
    client = None # Ensure client is None if initialization fails


AGENT_REGISTRY_PATH = "agent_registry.yaml"
_agent_registry_cache = None

def load_agent_registry(force_reload=False):
    """Loads the agent registry YAML file with caching."""
    global _agent_registry_cache
    if _agent_registry_cache is None or force_reload:
        try:
            with open(AGENT_REGISTRY_PATH, 'r') as f:
                _agent_registry_cache = yaml.safe_load(f)
            logger.info("Agent registry loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Agent registry file not found at {AGENT_REGISTRY_PATH}")
            _agent_registry_cache = {} # Return empty dict on error
        except yaml.YAMLError as e:
            logger.error(f"Error parsing agent registry YAML: {e}")
            _agent_registry_cache = {}
    return _agent_registry_cache

def get_node_config(node_name: str, version: Optional[str] = None) -> dict:
    registry = load_agent_registry()
    if not registry:
        logger.warning("Agent registry not loaded")
        return {}

    node_entry = registry.get("nodes", {}).get(node_name)
    if not node_entry:
        logger.warning(f"Node '{node_name}' not found in registry")
        return {}

    # If version is not provided, try to use the active version
    if version is None:
        version = registry.get("active_node_versions", {}).get(node_name)
        if not version:
            logger.warning(f"No active version for node '{node_name}'")
            return {}

    if node_entry.get("version") != version:
        logger.warning(f"Requested version '{version}' for node '{node_name}' does not match the configured version '{node_entry.get('version')}'")
        return {}

    # Optionally merge with global defaults if you had a defaults section
    # For now, assume no global defaults
    return node_entry


def load_prompt_from_path(prompt_path: str) -> str:
    """Loads a prompt from a .txt file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_path}")
        return "" # Return an empty string or raise an error

def get_llm_response(prompt: str, system_prompt: Optional[str] = None, model: str = "gpt-4o", temperature: float = 0.1, max_tokens: int = 5000, json_mode: bool = False) -> str:
    """Gets a response from the specified LLM, supporting JSON mode."""
    if not client:
        logger.error("OpenAI client not initialized. Cannot make API call.")
        return "Error: OpenAI client not initialized."
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    #     # Include conversation history if present in state
    # if "history" in state and state["history"]:
    #     messages.extend(state["history"])

    messages.append({"role": "system", "content": prompt})
    logger.info(f"messages: {messages}")

    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        request_params["response_format"] = {"type": "json_object"}

    try:
        logger.debug(f"Sending request to LLM ({model}) with prompt: {prompt[:1000]}...")
        response = client.chat.completions.create(**request_params)
        content = response.choices[0].message.content
        # Log the response content, but truncate for readability
        logger.info(f"LLM ({model}) response: {content[:1000]}...")

        if json_mode:
            # For json_mode, the content is already a JSON string
            try:
                if isinstance(content, dict):  # Already parsed (shouldn't happen with current API)
                    return content
                elif isinstance(content, str):
                    return json.loads(content)
                else:
                    raise ValueError(f"Unexpected response type: {type(content)}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"JSON parse failed: {str(e)}")
                return {"error": "Invalid JSON response", "details": str(e)}
        
        logger.info(f"content: {content}")
        return content

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error: OpenAI API call failed. Details: {e}"
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI API: {e}")
        return f"Error: Could not get response from LLM. Details: {e}"


def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Generates embeddings for a list of texts."""
    if not client:
        logger.error("OpenAI client not initialized. Cannot get embeddings.")
        return [[] for _ in texts] # Return list of empty lists for compatibility
    if not texts:
        return []
    try:
        # Ensure texts are non-empty strings, API might error otherwise
        processed_texts = [text if text.strip() else " " for text in texts]

        response = client.embeddings.create(input=processed_texts, model=model)
        return [item.embedding for item in response.data]
    except OpenAIError as e:
        logger.error(f"OpenAI API error getting embeddings: {e}")
        return [[] for _ in texts]
    except Exception as e:
        logger.error(f"Unexpected error getting embeddings: {e}")
        return [[] for _ in texts]

DB_SCHEMA_FOR_PROMPT = """
Database Schema:
Tables:
1. Customers(id INTEGER PRIMARY KEY, name TEXT, email TEXT)
2. Products(id INTEGER PRIMARY KEY, name TEXT, price REAL, inventory_count INTEGER)
3. Orders(id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT)
4. Returns(id INTEGER PRIMARY KEY, order_id INTEGER, reason TEXT, approved_by TEXT, status TEXT)

Relationships:
- Orders.customer_id references Customers.id
- Returns.order_id references Orders.id
"""