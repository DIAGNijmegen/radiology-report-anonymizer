"""
Anonymization tool for radiology reports

Authors: Ward Hendrix (ward.hendrix@radboudumc.nl) and Nils Hendrix (nils.hendrix@radboudumc.nl)

"""

import argparse
from datetime import datetime
from pathlib import Path

import jsonlines
from unidecode import unidecode
from tqdm import tqdm

from model.anonymizer_functions import Anonymizer


if __name__ == "__main__":

    # Load settings
    parser = argparse.ArgumentParser(
        description="Settings for anonymizing radiology reports."
    )
    parser.add_argument(
        "--input",
        default="/input",
        help="Path to (1) folder with raw radiology reports (.txt) or (2) JSON Lines file (.jsonl). ",
    )
    parser.add_argument(
        "--output",
        default="/output",
        help="Destination path to (1) folder for saving anonymized reports (.txt) or "
        "(2) JSON Lines file (.jsonl). ",
    )
    parser.add_argument(
        "--flag-list",
        nargs="+",
        default=["registratiennr", "adres"],
        help="Flag reports that contain one or more of the provided keywords.",
    )
    parser.add_argument(
        "--dump-annotations",
        default=False,
        action="store_true",
        help="Save text span annotations as a json file to output directory and not the reports.",
    )
    default_entities = [
        "person",
        "date",
        "time",
        "internal_phone_number",
        "patient_id",
        "z_number",
        "report_id",
        "location",
    ]
    parser.add_argument(
        "--entities-to-anonymize",
        nargs="+",
        default=default_entities,
        help="List of entities to anonymize.",
    )
    args = parser.parse_args()

    # Input validation
    flag_list = list(args.flag_list)
    entities_to_anonymize = list(args.entities_to_anonymize)
    if not all(type(value) == str for value in flag_list):
        raise ValueError(
            "The provided list for the flag-entities parameter can only consist of strings!"
        )
    if not all(type(value) == str for value in entities_to_anonymize):
        raise ValueError(
            "The provided list for the anonymized-entities parameter can only consist of strings!"
        )
    for value in entities_to_anonymize:
        if value not in default_entities:
            raise NotImplementedError(
                f'The specified entity "{value}" for anonymization is not supported! Choose from: {default_entities}.'
            )

    # Initialise anonymizer
    anonymizer = Anonymizer(entities_to_anonymize=entities_to_anonymize)

    print("Load reports from disk")

    input_path = Path(args.input)
    if input_path.suffix == ".jsonl" and input_path.is_file():
        original_reports = []
        count = 0
        with jsonlines.open(input_path) as reader:
            for obj in reader:
                text = unidecode(obj["text"])
                metadata = obj["meta"]
                if "StudyInstanceUID" in metadata.keys():
                    report_id = metadata["StudyInstanceUID"]
                elif "filename" in metadata.keys():
                    report_id = metadata["filename"]
                else:
                    report_id = count
                    count += 1
                original_reports.append([text, report_id])
    elif input_path.is_dir():
        original_reports = []
        report_files = [
            file for file in input_path.glob("**/*") if file.suffix == ".txt"
        ]
        if len(report_files) == 0:
            print(
                "No reports found: please specify a path to (1) a directory with .txt files or "
                "(2) to a .jsonl file. Abort program."
            )
            quit()
        else:
            for report_file in tqdm(report_files, total=len(report_files)):
                with open(report_file, "r", encoding="utf-8") as report:
                    text = unidecode(report.read())
                    original_reports.append([text, report_file.stem])
    else:
        print(
            "The specified input path is invalid: please specify a path to (1) a directory with .txt files or "
            "(2) to a .jsonl file. Abort program."
        )
        quit()

    print("Anonymize reports")

    anonymized_reports = []
    flag_list = [i.strip() for i in flag_list]
    for index, report in enumerate(tqdm(original_reports, total=len(original_reports))):
        anonymized_text = anonymizer.anonymize_report(report[0])
        flagged = False
        if any([i for i in flag_list if i in anonymized_text]):
            flagged = True
        metadict = {"filename": report[1], "flagged": flagged}
        anonymizer.add_metadata(index, metadict)
        anonymized_reports.append({"text": anonymized_text, "meta": metadict})

    output_path = Path(args.output)
    if args.dump_annotations:
        print("Save annotations to disk")
        annotations_name = f"annotations_{datetime.now().date()}.jsonl"
        if output_path.suffix == ".jsonl":
            json_path = output_path
        elif output_path.is_dir():
            json_path = output_path / annotations_name
        else:
            raise Exception(
                "The specified output path is invalid: please specify a path to (1) a directory or "
                "(2) to a .jsonl file. Abort program."
            )
        json_path.parent.mkdir(exist_ok=True, parents=True)
        with jsonlines.open(json_path, "w") as writer:
            writer.write_all(anonymizer.annotations)
    else:
        print("Save anonymized reports to disk")
        anonymizer.save_reports(anonymized_reports, output_path)
