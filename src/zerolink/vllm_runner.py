import torch
import textwrap
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"


class ModelRunner(object):
    """
    Model evaluation using VLLM.
    """

    def __init__(self, model_name, quantized=False, max_tokens=1024, num_beams=4):
        self.model_name = model_name
        self.quantized = quantized
        self.max_tokens = max_tokens
        self.num_beams = num_beams
        self.system = "Translate English to Postgres SQL."

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=TOKEN_CACHE
        )
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        self.params = SamplingParams(
            n=1,
            best_of=self.num_beams,
            use_beam_search=True,
            stop_token_ids=[self.tokenizer.eos_token_id],
            max_tokens=self.max_tokens,
            temperature=0,
        )

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

        outputs = self.model.generate(prompt, self.params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        return generated_text
