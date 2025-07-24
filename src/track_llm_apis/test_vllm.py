import copy
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from track_llm_apis.util import available_gpu_memory_fraction

DEVICE = "cuda"


class WorkerExtension:
    def debug(self):
        return (
            repr(self.model_runner.model),
            repr(dir(self.model_runner.model)),
        )

    def debug_code(self, code: str):
        """
        Execute code dynamically within the worker context for debugging.

        Args:
            code: Python source code as string

        Returns:
            dict with 'output', 'error', 'result', and 'stderr' keys
        """
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        stdout_capture = StringIO()
        stderr_capture = StringIO()
        result = None
        error = None

        # Make worker context available to the debug code
        local_vars = {
            "self": self,
            "model": self.model_runner.model,
            "model_runner": self.model_runner,
        }

        global_vars = {
            "torch": __import__("torch"),
            "time": __import__("time"),
            "copy": __import__("copy"),
        }

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to evaluate as expression first
                try:
                    result = eval(code, global_vars, local_vars)
                except SyntaxError:
                    # If it's not an expression, execute as statements
                    exec(code, global_vars, local_vars)
                    result = None

        except Exception as e:
            error = str(e)
            import traceback

            error += "\n" + traceback.format_exc()

        return {
            "output": stdout_capture.getvalue(),
            "error": error,
            "result": result,
            "stderr": stderr_capture.getvalue(),
        }

    def update_weights_from_tensors(self, weight_dict):
        """
        Update model weights from a dictionary of GPU tensors.
        """
        model = self.model_runner.model
        params_dict = dict(model.named_parameters())
        update_results = {}

        for name, new_weight in weight_dict.items():
            if name in params_dict:
                param = params_dict[name]

                # Ensure the new weight is on the same device and has the same shape
                if new_weight.device != param.device:
                    new_weight = new_weight.to(param.device)

                if new_weight.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected {param.shape}, got {new_weight.shape}"
                    )

                # Update the parameter in place
                with torch.no_grad():
                    param.data.copy_(new_weight)

                update_results[name] = True
            else:
                print(f"Warning: Parameter '{name}' not found in model")
                update_results[name] = False

        return update_results

    def get_parameter_names(self):
        """Get all parameter names in the model."""
        model = self.model_runner.model
        return list(dict(model.named_parameters()).keys())

    def get_parameter_info(self):
        """Get information about all model parameters."""
        model = self.model_runner.model
        param_info = {}

        for name, param in model.named_parameters():
            param_info[name] = {
                "shape": list(param.shape),
                "device": str(param.device),
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad,
            }

        return param_info


def random_noise(scale: float, model: torch.nn.Module):
    new_model = copy.deepcopy(model)
    for _, param in new_model.named_parameters():
        if param.requires_grad:
            param.data += torch.normal(
                mean=0.0,
                std=scale,
                size=param.data.shape,
                device=param.data.device,
                dtype=param.data.dtype,
            )
    return new_model


def cleanup_vllm(llm):
    """Clean up vLLM instance and free GPU memory."""
    destroy_model_parallel()
    # https://github.com/vllm-project/vllm/issues/1908#issuecomment-2975218097
    llm.llm_engine.engine_core.shutdown()
    del llm
    torch.cuda.empty_cache()


def load_model_to_vllm(llm: LLM | None, model, tokenizer):
    """Load a model into vLLM using temporary directory."""
    if llm is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model and tokenizer to temporary directory
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)

            # Load into vLLM
            available_memory_fraction = available_gpu_memory_fraction()
            llm = LLM(
                model=temp_dir,
                enforce_eager=True,
                gpu_memory_utilization=0.95 * available_memory_fraction,
                worker_extension_cls="__main__.WorkerExtension",
            )
            return llm
    else:
        print(llm.collective_rpc("debug"))
        pass


def run_inference(llm, prompt="Hi"):
    """Run inference on the given prompt and return the result."""
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    second_model = random_noise(scale=0.5, model=model)

    prompt = "Hi"
    num_switches = 5  # Number of times to switch between models

    print(f"Starting model switching benchmark with prompt: '{prompt}'")
    print(f"Will switch between models {num_switches} times\n")

    print("Loading original model into vLLM for the first time...")
    start_time = time.time()
    llm = load_model_to_vllm(None, model, tokenizer)
    load_time = time.time() - start_time
    print(f"Original model loaded in {load_time:.2f}s")
    result1 = run_inference(llm, prompt)
    print(f"Original model output: '{result1}'")

    for i in range(num_switches):
        print(f"=== Switch {i + 1} ===")

        print("Loading noisy model into vLLM...")
        start_time = time.time()
        llm = load_model_to_vllm(llm, second_model, tokenizer)
        load_time = time.time() - start_time
        print(f"Noisy model loaded in {load_time:.2f}s")

        result2 = run_inference(llm, prompt)
        print(f"Noisy model output: '{result2}'")

        print("Loading original model into vLLM...")
        start_time = time.time()
        llm = load_model_to_vllm(llm, model, tokenizer)
        print(f"Original model loaded in {load_time:.2f}s")
        result1 = run_inference(llm, prompt)
        print(f"Original model output: '{result1}'")

    print("Benchmark complete!")
