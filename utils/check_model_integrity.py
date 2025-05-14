from typing import Union
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

from model.anonymizer_functions import Anonymizer


DEFAULT_CONFIG_DIR = Path(__file__).parents[1] / "config"
DEFAULT_TESTSET = Path(__file__).parents[1] / "tests" / "test_set_v1_0_0.csv"


def test_csv(
    csv_filepath: Union[str, Path] = DEFAULT_TESTSET,
    config_dir_path: Union[str, Path] = DEFAULT_CONFIG_DIR,
):
    # Load csv file as a pandas dataframe
    df_reports = pd.read_csv(csv_filepath, sep=";", encoding="utf-8")

    # Load test sentences
    original_reports = [unidecode(i.strip()) for i in df_reports["text"].tolist()]

    # Load target sentences
    targets = [unidecode(i.strip()) for i in df_reports["target"].tolist()]

    # Load pattern type (like "1 initial (end with dot) + family name")
    patterns = [unidecode(i.strip()) for i in df_reports["pattern"].tolist()]

    # Load tags that are included in the test sentences (like "PERSOON", "DATUM", etc.)
    tags = []
    for tag in df_reports["tag"].tolist():
        if type(tag) == str:
            tags.append(unidecode(tag.strip()))
        elif pd.isna(tag):
            tags.append(None)
        else:
            try:
                tags.append(unidecode(tag.strip()))
            except ValueError:
                raise ValueError(
                    f"The tag {tag} has been found in the ground-truth annotation file and could not be converted to a string! Abort program."
                )

    # Perform anonymization
    anonymizer = Anonymizer(config_dir_path)

    anonymized_reports = []
    for report in tqdm(original_reports, total=len(original_reports)):
        anonymized_report = anonymizer.anonymize_report(report)
        anonymized_reports.append(anonymized_report)

    msg, success = anonymizer.generate_testreport(
        csv_filepath, anonymized_reports, original_reports, targets, tags, patterns
    )

    assert success, msg
