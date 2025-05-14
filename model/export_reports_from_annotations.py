import argparse
import os

from tqdm import tqdm
from pathlib import Path

from utils.read_files import read_json_lines
from model.anonymizer_functions import Anonymizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Export anonymized reports from annotations."
    )
    parser.add_argument(
        "--input-annotations",
        type=Path,
        required=True,
        help="Path to JSON Lines file (.jsonl) with annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/output",
        help="Output folder for saving anonymized reports. Default: /output",
    )
    args = parser.parse_args()

    # Initialise anonymizer (domain does not matter)
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config")
    )
    anonymizer = Anonymizer(config_dir)

    print("Load annotation file...")
    reports_with_annotations = read_json_lines(args.input_annotations)

    anonymized_reports = []
    filenames = []
    sorted_labels = [
        sorted(i[1], key=lambda x: x[0], reverse=True) for i in reports_with_annotations
    ]
    for index, item in enumerate(
        tqdm(reports_with_annotations, total=len(reports_with_annotations))
    ):
        text = item[0]
        labels = sorted_labels[index]
        metadata = item[2]
        for label in labels:
            start_idx = label[0]
            end_idx = label[1]
            tag = label[2]
            text = text[:start_idx] + tag + text[end_idx:]
        anonymized_reports.append(text)
        filenames.append(metadata["filename"])

    print("Write data...")
    for index, report in enumerate(
        tqdm(anonymized_reports, total=len(anonymized_reports))
    ):
        file_path = args.output_dir / filenames[index]
        if file_path.suffix != ".txt":
            file_path = file_path.parent / f"{file_path.name}.txt"
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(report)
