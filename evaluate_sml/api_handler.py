import time
import yaml
import os
from openai import OpenAI

def get_model_response(
    model_name: str,
    api_key: str,
    input_text: str,
    max_tokens: int,
    temperature: float,
    retries: int = None,
    timeout: int = None,
    base_url: str = None
):
    """
    Sends a request to the specified model API and retrieves the response using OpenAI client.

    Args:
        model_name (str): The name of the model to use.
        api_key (str): The API key for authorization.
        input_text (str): The input prompt for the model.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature.
        retries (int): The number of times to retry the request in case of failure.
        timeout (int): The timeout for the request in seconds.
        base_url (str): The base URL for the API endpoint.

    Returns:
        str: The content of the model's response.

    Raises:
        RuntimeError: If the request fails after all retries.
    """
    # Load config if parameters not provided
    if retries is None or timeout is None or base_url is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        retries = retries or config["api"]["retries"]
        timeout = timeout or config["api"]["timeout"]
        base_url = base_url or config["api"]["base_url"]
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )

    last_exception = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": input_text}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            return content, usage
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            last_exception = e
            if attempt < retries - 1:
                # Linear backoff for timeout issues - less aggressive than exponential
                sleep_time = config["api"].get("sleep_time", 0.5)
                sleep_time = sleep_time + (2 * attempt)
                time.sleep(sleep_time)
            # Remove the redundant 'else: continue' as the loop will continue naturally

    raise RuntimeError(f"Failed to get a response after {retries} attempts") from last_exception


if __name__ == "__main__":
    # Load config for test
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    MY_API_KEY = config["api"]["api_key"]
    MY_MODEL = "DeepSeek-V3.1"
    MY_PROMPT = "Hi! How are you?"
    MY_PROMPT = "Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\mathcal{B}$. The value of $r^2$ can be written as $\rac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$."

    model_output, usage = get_model_response(
        model_name=MY_MODEL,
        api_key=MY_API_KEY,
        input_text=MY_PROMPT,
        max_tokens=512,
        temperature=0.0
    )

    print("--- Model Output ---")
    print(model_output)

    print("--- Token Usage ---")
    print(usage)

