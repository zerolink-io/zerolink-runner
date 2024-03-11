import logging
from typing import Optional
from importlib.util import find_spec


def to_sql(
    question: str,
    context: Optional[str] = None,
    model_name: str = "zerolink/zsql-en-postgres",
    framework: Optional[str] = None,
) -> str:
    """
    Generate SQL from a question and context.
    """

    if not framework:
        if find_spec("mlx") is not None:
            logging.info("Using MLX")
            from .mlx_runner import ModelRunner
        elif find_spec("vllm") is not None:
            logging.info("Using VLLM")
            from .vllm_runner import ModelRunner
        elif find_spec("llamacpp") is not None:
            logging.info("Using Llama.cpp")
            from .llamacpp_runner import ModelRunner
    else:
        if framework == "mlx":
            from .mlx_runner import ModelRunner
        elif framework == "vllm":
            from .vllm_runner import ModelRunner
        elif framework == "llamacpp":
            from .llamacpp_runner import ModelRunner
        else:
            raise ValueError(f"Framework {framework} not supported")

    model = ModelRunner(model_name)
    model.setup()
    return model.predict(question, context=context)
