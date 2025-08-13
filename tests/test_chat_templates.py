import pytest
import torch
from transformers import AutoTokenizer

from track_llm_apis.config import config
from track_llm_apis.util import patch_chat_template

# Get all models from the chat templates
chat_templates = config.chat_templates
all_models = []
for template_hash, cfg in chat_templates.items():
    all_models.extend(cfg["models"])


@pytest.fixture
def dummy_conversation():
    """Create a dummy conversation for testing."""
    return [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]


@pytest.mark.parametrize("model_name", all_models)
def test_chat_template_patching(model_name, dummy_conversation):
    """Test that patched chat templates work correctly for each model:
    - the generated string is exactly the same;
    - the decoded tokens corresponding to the assistant mask are exactly the assistant's response.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_tokenized = tokenizer.apply_chat_template(
        dummy_conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )

    original_string = tokenizer.apply_chat_template(
        dummy_conversation, tokenize=False, add_generation_prompt=False
    )

    patch_chat_template(tokenizer, chat_templates)

    patched_tokenized = tokenizer.apply_chat_template(
        dummy_conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )

    patched_string = tokenizer.apply_chat_template(
        dummy_conversation, tokenize=False, add_generation_prompt=False
    )

    assert torch.equal(original_tokenized["input_ids"], patched_tokenized["input_ids"])
    assert torch.equal(original_tokenized["attention_mask"], patched_tokenized["attention_mask"])

    assert original_string == patched_string

    assistant_mask = patched_tokenized["assistant_masks"]
    assistant_token_ids = patched_tokenized["input_ids"][assistant_mask == 1]
    assistant_text = tokenizer.decode(assistant_token_ids, skip_special_tokens=True)
    expected_assistant_text = dummy_conversation[1]["content"]
    assert assistant_text == expected_assistant_text
