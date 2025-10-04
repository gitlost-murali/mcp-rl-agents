"""vLLM OpenAI-compatible client for hitting a vLLM server."""

import os
import subprocess
import time
from typing import Optional, List, Dict, Any

import requests
from openai import AsyncOpenAI

from train_agent.model_schemas import VLLMConfig, SamplingConfig


class VLLMEngine:
    """vLLM OpenAI-compatible client wrapper."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.client = None
        self.server_process = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client to hit vLLM server."""
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

        print(f"Initializing vLLM client with base_url: {base_url}")

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        print("vLLM client initialized")

    def _wait_for_server_ready(self, host: str, port: int, timeout: int = 180, poll_interval: int = 10):
        """Wait for server to be ready by polling the models endpoint.

        Args:
            host: Server host
            port: Server port
            timeout: Maximum time to wait in seconds
            poll_interval: Time to wait between polling attempts in seconds

        Raises:
            TimeoutError: If server doesn't become ready within timeout
        """
        start_time = time.time()
        models_url = f"http://{host}:{port}/v1/models"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(models_url, timeout=5)
                if response.status_code == 200:
                    return
            except (requests.RequestException, Exception) as e:
                print(f"Server not ready yet: {e}")
                pass

            time.sleep(poll_interval)

        raise TimeoutError(f"Server failed to start within {timeout} seconds")

    def start_server(self, port: int = 8000, host: str = "0.0.0.0"):
        """Start vLLM server as a subprocess."""
        if self.server_process is not None:
            print("Server already running")
            return

        print(f"Starting vLLM server on {host}:{port}")

        cmd = [
            "vllm", "serve",
            self.config.model_name,
            "--host", host,
            "--port", str(port),
            "--max-model-len", str(self.config.max_seq_length),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--dtype", self.config.dtype,
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]

        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")

        print(f"Starting vLLM server with command: {cmd}")
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to be ready
        print("Waiting for server to start...")
        self._wait_for_server_ready(host, port)
        print("Server started and ready")

    def stop_server(self):
        """Stop vLLM server subprocess."""
        if self.server_process is None:
            print("No server running")
            return

        print("Stopping vLLM server...")
        self.server_process.terminate()
        try:
            self.server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.server_process.kill()
            self.server_process.wait()

        self.server_process = None
        print("Server stopped")

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
            raise RuntimeError("vLLM client not initialized")

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
