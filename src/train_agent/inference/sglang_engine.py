"""SGLang OpenAI-compatible client with hot weight update support.

SGLang vs vLLM Comparison
==========================

This engine provides a drop-in replacement for VLLMEngine with the added
capability of hot weight updates during training.

Key Differences:
----------------

1. **Weight Updates (CRITICAL)**:
   - SGLang: Hot-swap weights via `update_weights_from_disk()` without restart
   - vLLM: Requires full server stop/restart to load new weights
   - Performance: SGLang saves ~30-60 seconds per training step

2. **Default Ports**:
   - SGLang: 30000 (configurable)
   - vLLM: 8011 (configurable)

3. **Server Launch Command**:
   - SGLang: `python -m sglang.launch_server`
   - vLLM: `vllm serve`

4. **Parameter Mapping**:
   - `max_seq_length` → `--context-length` (SGLang) vs `--max-model-len` (vLLM)
   - `gpu_memory_utilization` → `--mem-fraction-static` (SGLang) vs `--gpu-memory-utilization` (vLLM)
   - `tensor_parallel_size` → `--tp` (SGLang) vs `--tensor-parallel-size` (vLLM)

5. **Tool Calling**:
   - SGLang: `--tool-call-parser <parser_name>`
   - vLLM: `--enable-auto-tool-choice --tool-call-parser <parser_name>`
   - Both support: qwen25, hermes, llama3, etc.

6. **Health Check Endpoints**:
   - SGLang: `/health`
   - vLLM: `/v1/models`

Compatibility Guarantees:
-------------------------

Both engines implement identical interfaces:
- `__init__(config: VLLMConfig)` - Same config object
- `start_server(port, host)` - Start inference server
- `stop_server()` - Stop inference server
- `generate_with_messages(...)` - OpenAI-compatible generation API

The only difference is the additional `update_weights_from_disk()` method
available in SGLang for hot weight updates.

When to Use SGLang:
-------------------

- **Online RL Training**: When you need to update model weights frequently
  without restarting the server (recommended for all training workflows)
- **Zero-downtime Updates**: When you can't afford server restart latency
- **Memory Efficiency**: When you need fine-grained control over GPU memory

When to Use vLLM:
-----------------

- **Static Inference**: When model weights don't change during execution
- **Legacy Compatibility**: When you need specific vLLM features
- **Production Serving**: When you don't need weight updates

Usage Example:
--------------

```python
from train_agent.inference import SGLangEngine, VLLMEngine
from train_agent.model_schemas import VLLMConfig

# Both use the same config
config = VLLMConfig.from_config()

# SGLang with hot weight updates (recommended for training)
engine = SGLangEngine(config)
engine.start_server(port=30000)

# ... collect rollouts ...

# Update weights without restart (SGLang only!)
engine.update_weights_from_disk("/path/to/checkpoint")

# ... collect more rollouts with new weights ...

engine.stop_server()

# vLLM (traditional approach)
engine = VLLMEngine(config)
engine.start_server(port=8011)

# ... collect rollouts ...

# To update weights, must restart entire server
engine.stop_server()
config.model_name = "/path/to/checkpoint"
engine = VLLMEngine(config)
engine.start_server(port=8011)
```
"""

import os
import subprocess
import time
from typing import Optional, List, Dict, Any

import requests
from openai import AsyncOpenAI

from train_agent.model_schemas import VLLMConfig, SamplingConfig


class SGLangEngine:
    """SGLang OpenAI-compatible client wrapper with weight update support.

    Key advantages over vLLM:
    - Hot weight updates without server restart (update_weights_from_disk)
    - Designed specifically for online RL workflows
    - Zero-downtime model weight swapping
    """

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.client = None
        self.server_process = None
        self.base_url = os.environ.get("SGLANG_BASE_URL", "http://localhost:30000")
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client to hit SGLang server."""
        api_key = os.environ.get("SGLANG_API_KEY", "EMPTY")

        print(f"Initializing SGLang client with base_url: {self.base_url}/v1")

        self.client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
        )

        print("SGLang client initialized")

    def _wait_for_server_ready(self, timeout: int = 300, poll_interval: int = 10):
        """Wait for server to be ready by polling the health endpoint.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: Time to wait between polling attempts in seconds

        Raises:
            TimeoutError: If server doesn't become ready within timeout
        """
        start_time = time.time()
        health_url = f"{self.base_url}/health"

        print(f"Waiting for SGLang server to be ready at {health_url}...")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print("SGLang server is ready!")
                    return
            except (requests.RequestException, Exception) as e:
                print(f"Server not ready yet: {e}")

            time.sleep(poll_interval)

        raise TimeoutError(f"SGLang server failed to start within {timeout} seconds")

    def start_server(self, port: int = 30000, host: str = "0.0.0.0"):
        """Start SGLang server as a subprocess.

        Args:
            port: Server port (default 30000 for SGLang)
            host: Server host
        """
        if self.server_process is not None:
            print("Server already running")
            return

        print(f"Starting SGLang server on {host}:{port}")

        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.config.model_name,
            "--host", host,
            "--port", str(port),
            "--context-length", str(self.config.max_seq_length),
            "--mem-fraction-static", str(self.config.gpu_memory_utilization),
            "--tp", str(self.config.tensor_parallel_size),
            "--dtype", self.config.dtype if self.config.dtype != "auto" else "auto",
        ]

        # Add tool calling support if configured
        if self.config.tool_call_parser:
            cmd.extend(["--tool-call-parser", self.config.tool_call_parser])

        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")

        # Set CUDA_VISIBLE_DEVICES to use GPU 0 for SGLang
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        print(f"Starting SGLang server with command: {' '.join(cmd)}")
        print(f"SGLang CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")

        # Open log files for SGLang output
        stdout_log = open("sglang_stdout.log", "a")
        stderr_log = open("sglang_stderr.log", "a")

        # Write timestamp header to logs
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stdout_log.write(f"\n{'='*80}\n")
        stdout_log.write(f"SGLang server started at {timestamp}\n")
        stdout_log.write(f"{'='*80}\n")
        stdout_log.flush()

        self.server_process = subprocess.Popen(
            cmd,
            stdout=stdout_log,
            stderr=stderr_log,
            env=env
        )
        self._stdout_log = stdout_log
        self._stderr_log = stderr_log

        # Update base_url to match the started server
        self.base_url = f"http://{host}:{port}"
        self._init_client()  # Reinitialize client with new URL

        # Wait for server to be ready
        print("Waiting for server to start...")
        self._wait_for_server_ready()
        print("Server started and ready")

    def stop_server(self):
        """Stop SGLang server subprocess."""
        if self.server_process is None:
            print("No server running")
            return

        print("Stopping SGLang server...")
        self.server_process.terminate()
        try:
            self.server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.server_process.kill()
            self.server_process.wait()

        # Close log files
        if hasattr(self, '_stdout_log'):
            self._stdout_log.close()
            delattr(self, '_stdout_log')
        if hasattr(self, '_stderr_log'):
            self._stderr_log.close()
            delattr(self, '_stderr_log')

        self.server_process = None
        print("Server stopped")

    def update_weights_from_disk(self, model_path: str) -> bool:
        """Update model weights from a checkpoint on disk WITHOUT restarting server.

        This is the key feature that makes SGLang superior to vLLM for online RL:
        - No server restart required
        - Zero-downtime weight swapping
        - Much faster than stop/start cycle

        Args:
            model_path: Path to model checkpoint directory

        Returns:
            True if update successful, False otherwise
        """
        if self.server_process is None:
            print("Warning: No server running, cannot update weights")
            return False

        print(f"Updating model weights from: {model_path}")

        try:
            response = requests.post(
                f"{self.base_url}/update_weights_from_disk",
                json={"model_path": model_path},
                timeout=120  # Weight loading can take time
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✓ Weights updated successfully: {result}")
                return True
            else:
                print(f"✗ Weight update failed: {response.status_code} - {response.text}")
                return False

        except requests.RequestException as e:
            print(f"✗ Weight update request failed: {e}")
            return False

    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        sampling_config: Optional[SamplingConfig] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """Generate completion using OpenAI client with streaming support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            sampling_config: Sampling parameters (temperature, top_p, etc.)
            tools: List of tool definitions
            tool_choice: Tool choice strategy ('auto', 'required', or specific tool)
            stream: Whether to stream the response
            extra_body: Additional parameters to pass (e.g., tool parser config)
        """
        if self.client is None:
            raise RuntimeError("SGLang client not initialized")

        if sampling_config is None:
            sampling_config = SamplingConfig()

        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": sampling_config.temperature,
            "top_p": sampling_config.top_p,
            "max_tokens": sampling_config.max_tokens,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = tools
            # Default to "auto" if tool_choice not specified
            if tool_choice is None:
                kwargs["tool_choice"] = "auto"

        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        if sampling_config.stop:
            kwargs["stop"] = sampling_config.stop

        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await self.client.chat.completions.create(**kwargs)

        return response
