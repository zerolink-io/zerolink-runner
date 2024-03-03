import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def run_vllm(model_name, inputs, quantized=False, max_tokens=1024, num_beams=4):
    """
    Run the given model using vllm and return the generated text.
    """
    model_name = "zerolink/zsql-en-postgres"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantized:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            quantization="AWQ",
        )
    else:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
        )

    sampling_params = SamplingParams(
        n=1,
        best_of=num_beams,
        use_beam_search=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=max_tokens,
        temperature=0,
    )

    outputs = llm.generate(inputs, sampling_params)
    generated_text = outputs.outputs[0].text

    return generated_text
