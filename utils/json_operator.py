import json
import os.path
from typing import Any


def read_json(source: str) -> list[Any]:
    json_list: list[Any] = []
    if not os.path.exists(source):
        return json_list
    with open(source, encoding="utf-8") as f:
        json_list.extend(json.loads(line) for line in f)
    f.close()
    return json_list


def dump_json(source: str, datas: list[Any]) -> None:
    with open(source, "w", encoding="utf-8") as f:
        for item in datas:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    f.close()
