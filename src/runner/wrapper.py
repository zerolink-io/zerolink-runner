import typing
from .vllm_runner import ModelRunner


def to_sql(
    question: str,
    context: Optional[str] = None,
    model_name: Optional[str] = "zerolink/zsql-en-postgres",
) -> str:
    """
    Generate SQL from a question and context.
    """
    model = ModelRunner(model_name)
    model.setup()
    return model.predict(context, question)
