import os
import textwrap
import subprocess
from typing import Optional
from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding


class ModelRunner(object):
    """
    Model evaluation using the llama.cpp
    """

    def __init__(self, model_name: str, max_tokens=1024, quantization="q5_0"):
        self.model_name = model_name
        self.gguf_name = f"{model_name}_{quantization}.gguf"
        self.max_tokens = max_tokens
        self.system = "Translate English to Postgres SQL."
        self.model_folder = os.path.join(os.getcwd(), ".models")

    def setup(self) -> None:
        url = f"https://huggingface.co/zerolink/{self.model_name}/resolve/main/{self.gguf_name}"

        # if the model folder doesn't exist, create it
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        subprocess.run(
            [
                "curl",
                "-L",
                url,
                "-o",
                os.path.join(self.model_folder, self.gguf_name),
            ]
        )

        # .models/zsql-en-postgres/zsql-en-postgres_q5_0.gguf
        model_file = os.path.join(self.model_folder, self.gguf_name)

        self.model = Llama(
            model_path=model_file,
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
