import re

def separate_thinking_and_response(response: str) -> tuple[str | None, str]:
    """
    Separates the thinking part from the actual response.

    Returns a tuple of (thinking, actual) where:
    - thinking: content inside <think>...</think> tags, or None if not present
    - actual: the response with thinking removed and code fences stripped
    """
    thinking = None

    if '</think>' in response:
        think_start = response.find('<think>')
        think_end = response.find('</think>') + len('</think>')

        if think_start != -1:
            thinking = response[think_start + len('<think>'):think_end - len('</think>')].strip()

        response = response[think_end:].strip()

    actual = re.sub(r"```(?:json|xml)?\s*([\s\S]*?)```", r"\1", response, flags=re.IGNORECASE).strip()

    return thinking, actual