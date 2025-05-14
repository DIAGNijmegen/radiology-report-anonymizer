from typing import Union
from pathlib import Path

import jsonlines
import json


def read_json_lines(json_path):
    """
    Reads specific annotation files in JSON Lines format.

    Input format should be:
    {"text": "Voorbeeld van Hendrix, Ward.", "labels": [[14, 27, "<PERSOON>"],
    "meta":{"filename":"example.txt", "rra_version": "v1.0.0"}]}

    """
    annotations = []
    with jsonlines.open(json_path) as reader:
        for obj in reader:
            text = obj["text"]
            labels = obj["labels"]
            metadata = obj["meta"]
            annotations.append([text, labels, metadata])

    return annotations


def load_json(file_path: Union[str, Path]) -> dict:
    """
    Reads a json file and returns a dictionary.
    """
    with open(file_path, "r") as fp:
        json_dict = json.load(fp)

    return json_dict
