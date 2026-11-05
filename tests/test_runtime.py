"""GPU guard checks. No model loading, inference, tensors or fitting."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from qwen_one_two.policy import TOTAL_CONTEXT_TOKENS, MAX_THINKING_TOKENS, check_model_context

from qwen_one_two.model import Runtime


class RuntimeGuards(unittest.TestCase):
    def test_no_cuda_fails_before_loading_any_weights(self):
        with patch("qwen_one_two.model.torch.cuda.is_available", return_value=False), \
             patch("qwen_one_two.model.AutoModelForCausalLM.from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU required"):
                Runtime("unused", "unused", "cuda:0")
            load.assert_not_called()

    def test_cpu_selection_is_rejected_even_when_cuda_exists(self):
        with patch("qwen_one_two.model.torch.cuda.is_available", return_value=True), \
             patch("qwen_one_two.model.AutoModelForCausalLM.from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "CPU model execution is disabled"):
                Runtime("unused", "unused", "cpu")
            load.assert_not_called()

    def test_wrong_gpu_is_rejected_before_loading_weights(self):
        with patch("qwen_one_two.model.torch.cuda.is_available", return_value=True), \
             patch("qwen_one_two.model.torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3060"), \
             patch("qwen_one_two.model.AutoModelForCausalLM.from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "RTX 3090 explicitly"):
                Runtime("unused", "unused", "cuda:0")
            load.assert_not_called()


class ContextPolicy(unittest.TestCase):
    def test_maximum_is_half_the_verified_checkpoint_context(self):
        self.assertEqual(TOTAL_CONTEXT_TOKENS, 262144)
        self.assertEqual(MAX_THINKING_TOKENS, 131072)
        cfg = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144))
        self.assertEqual(check_model_context(cfg) // 2, MAX_THINKING_TOKENS)
        with self.assertRaises(ValueError):
            check_model_context(SimpleNamespace(max_position_embeddings=2048))
