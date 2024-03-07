import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def run_vllm(model_name, inputs, quantized=False, max_tokens=1024, num_beams=4):
    """
    Run the given model using vllm and return the generated text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    system = "Translate English to Postgres SQL."
    prompt = f"Generate SQL for the following question: {inputs}"

    message = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        message, add_generation_prompt=True, tokenize=False
    )

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
        # n=1,
        # best_of=num_beams,
        # use_beam_search=True,
        # stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=max_tokens,
        temperature=0.1,
    )

    outputs = llm.generate(prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text
    print(outputs[0].outputs[0])
    return generated_text


print(run_vllm("zerolink/zsql-en-postgres", "What are the best selling products?"))
