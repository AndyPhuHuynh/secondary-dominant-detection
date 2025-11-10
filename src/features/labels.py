genre_labels: dict[str, int] = {
    "happy": 0,
    "sad":   1,
}
label_map = {v: k for k, v in genre_labels.items()}


def get_label_from_effect(genre: str) -> int:
    if genre in genre_labels:
        return genre_labels[genre]
    else:
        raise ValueError(f"No label found, invalid genre: {genre}")


def get_effect_from_label(label: int) -> str:
    if label in label_map:
        return label_map[label]
    else:
        raise ValueError(f"No label found, invalid label: {label}")