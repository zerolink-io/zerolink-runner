import textwrap
from typing import Optional
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from llama_cpp.llama_types import ChatCompletionRequestMessage


class ModelRunner(object):
    """
    Model evaluation using the llama.cpp
    """

    def __init__(self, model_name: str, max_tokens=1024):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.system = "Translate English to Postgres SQL."

    def setup(self) -> None:
        self.model = Llama(
            model_path=self.model_name,
            n_gpu_layers=-1,
            chat_format="chatml",
            n_ctx=2048,
            draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2),
            logits_all=True,
            verbose=False,
        )

    def predict(self, inputs: str, context: Optional[str] = "") -> str:
        prompt = textwrap.dedent(
            f"""
        Using the schema:
        {context}
        Generate SQL for the following question: {inputs}
        """
        ).strip()

        message = [
            ChatCompletionRequestMessage(role="system", content=self.system),
            ChatCompletionRequestMessage(role="user", content=prompt),
        ]

        output = self.model.create_chat_completion(
            message, max_tokens=self.max_tokens, temperature=0.1
        )

        return output.choices[0].text
