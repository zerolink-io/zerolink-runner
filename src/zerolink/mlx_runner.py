import textwrap
from mlx_lm import load, generate


class ModelRunner(object):
    """
    Model evaluation using MLX.
    """

    def __init__(self, model_name, quantized=False, max_tokens=1024, num_beams=4):
        self.model_name = model_name
        self.quantized = quantized
        self.max_tokens = max_tokens
        self.system = "Translate English to Postgres SQL."

    def setup(self):
        self.model, self.tokenizer = load(self.model_name)

    def predict(self, context, inputs):
        prompt = textwrap.dedent(
            f"""
        Using the schema:
        {context}
        Generate SQL for the following question: {inputs}
        """
        ).strip()

        message = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )

        outputs = self.model.generate(self.model, self.tokenizer, prompt, verbose=False)

        return outputs
