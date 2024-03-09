# ZeroLink

Example usage of how to run ZeroLink text to SQL models on various inference
frameworks.

* MLX ( Apple Silicon )
* VLLM ( Nvidia GPUs )
* llama.cpp ( CPU )

To install the `zerolink` package, you can use the following command:

```shell
pip install 'zerolink[mlx] @ git+https://github.com/zerolink-io/zerolink-runner.git'
```

## Using an Mac

To use MLX with your Apple Silicon Mac, you can install the `mlx` dependency
group with the following command:

```shell
poetry install --with mlx
```

Then you can use the `to_sql` function to convert a natural language question,
optionally pass the `context` parameter to provide schema information.

```python
from zerolink import to_sql

sql = to_sql("What are the top selling products?", framework="mlx")
print(sql)
```

## Using an Nvidia GPU

```shell
poetry install --with vllm
```

```python
from zerolink import to_sql

to_sql("What are the top selling products?", framework="vllm")
```

## Using a CPU

```shell
poetry install --with llamacpp
```

```python
from zerolink import to_sql

to_sql("What are the top selling products?", framework="llamacpp")
```

## License

This project is licensed under the Apache License - see
[LICENSE.md](LICENSE.md) file for details
