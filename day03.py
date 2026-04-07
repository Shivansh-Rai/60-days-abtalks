import os
from typing import List
from dotenv import load_dotenv # type: ignore
from openai import OpenAI


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

def load_api_key() -> str:
    """
    Loads API key from environment variables.
    Raises error if not found.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

    return api_key


def create_client(api_key: str) -> OpenAI:
    """
    Initializes and returns OpenAI client.
    """
    return OpenAI(api_key=api_key)


# -----------------------------------------
# CORE LOGIC
# -----------------------------------------

def generate_response(client: OpenAI, prompt: str,
                      model: str = "gpt-4o-mini",
                      max_tokens: int = 80,
                      temperature: float = 0.7) -> str:
    """
    Generates response from OpenAI model for a given prompt.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return handle_error(e)


def handle_error(error: Exception) -> str:
    """
    Handles API-related errors gracefully.
    """
    error_msg = str(error).lower()

    if "quota" in error_msg or "429" in error_msg:
        return "Error: API quota exceeded. Check your usage and billing."

    if "authentication" in error_msg or "api key" in error_msg:
        return "Error: Invalid or missing API key."

    return f"Unexpected Error: {str(error)[:150]}"


# -----------------------------------------
# UTILITIES
# -----------------------------------------

def print_response(index: int, prompt: str, response: str) -> None:
    """
    Prints formatted output.
    """
    print(f"\nPrompt {index}: {prompt}")
    print("Response:")
    print(response)
    print("-" * 50)


def run_prompts(client: OpenAI, prompts: List[str]) -> None:
    """
    Runs multiple prompts and prints responses.
    """
    print("\n===== OPENAI API RESPONSES =====\n")

    for i, prompt in enumerate(prompts, start=1):
        response = generate_response(client, prompt)
        print_response(i, prompt, response)


# -----------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------

def main():
    """
    Main execution function.
    """
    api_key = load_api_key()
    client = create_client(api_key)

    prompts = [
        "What is logistic regression? (1 sentence)",
        "One line motivational quote for coders.",
        "Why evaluate ML models? (1 sentence)"
    ]

    run_prompts(client, prompts)


if __name__ == "__main__":
    main()