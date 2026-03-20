import string

PAD_TOKEN = "<PAD>"
SEP_TOKEN = "<SEP>"
EOS_TOKEN = "<EOS>"

PAD_ID = 0
SEP_ID = 1
EOS_ID = 2


def build_token_mapping() -> dict[str, int]:
    token_mapping = {
        PAD_TOKEN: PAD_ID,
        SEP_TOKEN: SEP_ID,
        EOS_TOKEN: EOS_ID
    }

    for i, char in enumerate(string.ascii_lowercase, start=3):
        token_mapping[char] = i

    return token_mapping


def build_reverse_mapping(token_mapping: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in token_mapping.items()}


TOKEN_MAPPING = build_token_mapping()
REVERSE_TOKEN_MAPPING = build_reverse_mapping(TOKEN_MAPPING)
VOCAB_SIZE = len(TOKEN_MAPPING)


def encode(input_str: str, token_mapping: dict[str, int] = TOKEN_MAPPING) -> list[int]:
    return_ids = []
    for char in input_str:
        if char not in token_mapping:
            raise ValueError(f"Character '{char}' not in token mapping.")
        return_ids.append(token_mapping[char])
    return return_ids


def decode(input_ids: list[int], reverse_mapping: dict[int, str] = REVERSE_TOKEN_MAPPING) -> str:
    return_str = ""
    for i in input_ids:
        if i < 3:
            continue
        return_str += reverse_mapping[i]
    return return_str
