def invert_dictionary(in_dict: dict) -> dict:
    return {
        value: key
        for key, value in in_dict.items()
    }
