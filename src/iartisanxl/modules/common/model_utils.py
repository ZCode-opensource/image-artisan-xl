import json


def get_metadata_from_safetensors(filepath):
    metada = {}
    with open(filepath, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (
            b'{"',
            b"{'",
        ), f"{filepath} is not a safetensors file"
        json_data = json_start + file.read(metadata_len - 2)
        json_obj = json.loads(json_data)

        for k, v in json_obj.get("__metadata__", {}).items():
            metada[k] = v

    return metada
