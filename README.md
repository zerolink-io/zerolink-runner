# ZeroLink

To install the `zerolink` package, you can use the following command:

```shell
pip install 'zerolink @ git+https://github.com/zerolink-io/zerolink-runner.git'
```

In addition you will need one of the following dependencies to run the `zerolink`
model inference:

* MLX ( Apple Silicon )
* VLLM ( Nvidia GPUs )
* llama.cpp ( CPU )

## Using a Mac

To use MLX with your Apple Silicon Mac, you can install the `mlx` dependency
group with one of the following command:

```shell
pip install 'zerolink[mlx] @ git+https://github.com/zerolink-io/zerolink-runner.git'
poetry install --with mlx
```

You will need a system with at least 20GB of memory and 15GB of free disk space
to install the model locally.

You can use the `to_sql` function to convert a natural language question into
SQL. Optionally pass the `context` parameter to provide schema information.

```python
from zerolink import to_sql

sql = to_sql("What are the top selling products?", framework="mlx")
print(sql)
```

## Using an Nvidia GPU

To use VLLM on a server with attached Nvidia GPUs, you can install the `vllm`
dependency group with one of the following command:

```shell
pip install 'zerolink[vllm] @ git+https://github.com/zerolink-io/zerolink-runner.git'
poetry install --with vllm
```

Supported GPUs:

- Nvidia T4
- Nvidia A40
- Nvidia A100 (40GB / 80GB)
- Nvidia H100

To run the model inference on the GPU, you can use the following code:

```python
from zerolink import to_sql

to_sql("What are the top selling products?", framework="vllm")
```

## Using a CPU

To use llama.cpp on stock Intel or AMD CPUs, you can install the `llamacpp`
dependency group with one of the following command:

```shell
pip install 'zerolink[llamacpp] @ git+https://github.com/zerolink-io/zerolink-runner.git'
poetry install --with llamacpp
```

To run the model inference on a CPU, you can use the following code:

```python
from zerolink import to_sql

to_sql("What are the top selling products?", framework="llamacpp")
```

## License

This project is licensed under the Apache License - see [LICENSE.md](LICENSE.md) file for details
