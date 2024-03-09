from typing import Optional


def to_sql(
    question: str,
    context: Optional[str] = None,
    model_name: Optional[str] = "zerolink/zsql-en-postgres",
    framework: Optional[str] = "mlx",
) -> str:
    """
    Generate SQL from a question and context.
    """

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
    return model.predict(context, question)
