from typing import Optional

SQL_EXPLAINER_SYSTEM_PROMPT_DS_PREP = """You are an expert SQL analyst. When the user provides a SQL query, respond with a single concise paragraph explaining how the query works, covering what data it retrieves, the tables or sources involved, any filtering conditions, joins, groupings, or ordering applied, and the expected output. Do not use bullet points, headings, or code blocks. Keep the explanation clear and accessible, as if explaining to a developer who understands SQL but wants a quick summary. Do not write the given query again in your response."""
SQL_EXPLAINER_SYSTEM_PROMPT_FINAL = """You are an expert SQL analyst. Provide a single concise paragraph explaining the provided query's logic, data sources, filters, and expected output. Do not use formatting, code blocks, or repeat the original query."""


class PromptManager:
    """Utility class to construct prompts using a chat-style template."""

    def __init__(self, system: Optional[str] = None):
        """Initializes the PromptManager."""
        self.SYSTEM_PROMPT = system

    def get_prompt(
        self,
        query: str,
        output: Optional[str] = None
    ) -> str:
        """Construct a formatted prompt for the language model."""
        prompt = ""

        if self.SYSTEM_PROMPT:
            prompt += f"""<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"""

        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""

        if output:
            prompt += f"""{output}<|im_end|>"""

        return prompt
    
    def get_prompt_messages(
        self, 
        query: str,
        output: Optional[str] = None
    ) -> list[dict]:
        """Construct a list of chat messages in OpenAI-style format."""
        messages = []

        if self.SYSTEM_PROMPT:
            messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
        messages.append({"role": "user", "content": query})
        if output:
            messages.append({"role": "assistant", "content": output})
        
        return messages


# EXAMPLE USAGE
# if __name__ == "__main__":
#     SYSTEM = """You are a smart AI assistant who specializes in answering questions belonging to the healthcare domain."""
#     USER = "Which medicine is used when you have fever?"
#     ASSISTANT = "Paracetamol"

#     prompt_manager = PromptManager(SYSTEM)
#     prompt = prompt_manager.get_prompt(USER, ASSISTANT)
#     messages = prompt_manager.get_prompt_messages(USER, ASSISTANT)
#     print(prompt)
#     print(messages)