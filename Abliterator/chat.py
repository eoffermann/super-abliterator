from transformer_lens import HookedTransformer

class ChatTemplate:
    """
    A class to manage and apply a chat template for a given transformer model.

    This class provides functionality to format instructions based on a predefined template
    and temporarily override the model's chat template within a context manager.

    Attributes:
        model (HookedTransformer): The transformer model to which the template is applied.
        template (str): The format string defining the chat template.
    """

    def __init__(self, model: HookedTransformer, template: str):
        """
        Initializes a ChatTemplate instance.

        Args:
            model (HookedTransformer): The transformer model that will use the chat template.
            template (str): A format string that defines how instructions are formatted.
        """
        self.model = model
        self.template = template
        self.prev = None

    def format(self, instruction: str) -> str:
        """
        Formats an instruction string according to the defined chat template.

        Args:
            instruction (str): The instruction text to be formatted.

        Returns:
            str: The formatted instruction string based on the chat template.
        """
        return self.template.format(instruction=instruction)

    def __enter__(self):
        """
        Enters the context, temporarily overriding the model's chat template.

        Returns:
            ChatTemplate: The current instance with the applied chat template.
        """
        self.prev = self.model.chat_template  # Store the current chat template
        self.model.chat_template = self  # Apply this template to the model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the context, restoring the previous chat template.

        Args:
            exc_type (Type[BaseException] | None): The exception type if an error occurred.
            exc_value (BaseException | None): The exception instance if an error occurred.
            traceback (TracebackType | None): The traceback object if an error occurred.
        """
        self.model.chat_template = self.prev  # Restore the original chat template
        del self.prev  # Clean up stored reference

# Predefined chat templates for different models

LLAMA3_CHAT_TEMPLATE = (
    "<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
"""
LLAMA3 Chat Template:
Defines the structure for user-assistant interactions in the LLAMA3 model.

Format:
    - User input: Enclosed between "<|start_header_id|>user<|end_header_id|>" and "<|eot_id|>"
    - Assistant response: Begins with "<|start_header_id|>assistant<|end_header_id|>"
"""

PHI3_CHAT_TEMPLATE = "<|user|>\n{instruction}<|end|>\n<|assistant|>"
"""
PHI3 Chat Template:
Defines the structure for user-assistant interactions in the PHI3 model.

Format:
    - User input: Prefixed with "<|user|>", followed by "{instruction}", and terminated with "<|end|>"
    - Assistant response: Begins with "<|assistant|>"
"""
