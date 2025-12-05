_string_to_num: dict[str, int] = {
    "diatonic": 0,
    "non-diatonic":   1,
}
_num_to_string = {v: k for k, v in _string_to_num.items()}


def get_label_string_to_num(string: str) -> int:
    if string in _string_to_num:
        return _string_to_num[string]
    else:
        raise ValueError(f"No label found, invalid genre: {string}")


def get_label_num_to_string(label: int) -> str:
    if label in _num_to_string:
        return _num_to_string[label]
    else:
        raise ValueError(f"No label found, invalid label: {label}")