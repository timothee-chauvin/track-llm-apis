import random

random.seed(0)


def generate_random_prompts(prompt_lengths: list[int], n_prompts_per_length: int) -> list[str]:
    """
    Generate random prompts of different lengths.

    Random bytes of the given size are generated and kept if they are valid UTF-8.
    """
    prompts = []
    for prompt_length in prompt_lengths:
        for _ in range(n_prompts_per_length):
            while True:
                try:
                    random_bytes = random.randbytes(prompt_length)
                    random_string = random_bytes.decode("utf-8")
                    prompts.append(random_string)
                    break
                except UnicodeDecodeError:
                    continue
    return prompts


if __name__ == "__main__":
    prompt_lengths = [2, 4, 8]
    n_prompts_per_length = 3
    prompts = generate_random_prompts(prompt_lengths, n_prompts_per_length)
    if prompt_lengths == [2, 4, 8] and n_prompts_per_length == 3:
        assert prompts == [
            "]\n",
            "HB",
            "e|",
            "xég",
            "\x04B\x02z",
            "\x1e·T",
            "\x06P\x1dz\x13ZTq",
            "ZZ\x17˚p|[",
            "\x14\x1ap88V?_",
        ]
    print(prompts)
