from openai import OpenAI

from src.config import settings


def get_openai_client():
    """Return the initialized OpenAI client."""
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return openai_client
