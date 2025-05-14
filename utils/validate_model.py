import argparse
import os
from datetime import datetime

import pandas as pd
from tabulate import tabulate

from utils.read_files import read_json_lines


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-annotations",
        required=True,
        help="Path to JSON Lines file (.jsonl) with annotations.",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to JSON Lines file (.jsonl) with ground truth annotations.",
    )
    parser.add_argument(
        "--output-dir", default="/output", help="Output folder for saving the results."
    )
    args = parser.parse_args()

    # Load all labels and save failure cases to list
    annotations = read_json_lines(args.input_annotations)
    ground_truth = read_json_lines(args.ground_truth)
    rra_version = annotations[0][2]["rra_version"]
    ground_truth_name = os.path.split(args.ground_truth)[1]

    corpus_dict = {}
    truth_label_dict = {}
    model_label_dict = {}

    all_labels_model = []
    for index, item in enumerate(annotations):
        text = item[0]
        labels = item[1]
        metadata = item[2]
        filename = str(metadata["filename"])
        for label in labels:
            all_labels_model.append(label + [filename])
        corpus_dict[filename] = text
        model_label_dict[filename] = labels

    all_labels_ground_truth = []
    for index, item in enumerate(ground_truth):
        labels = ground_truth[index][1]
        metadata = ground_truth[index][2]
        if "StudyInstanceUID" in metadata.keys():
            filename = str(metadata["StudyInstanceUID"])
        elif "filename" in metadata.keys():
            filename = str(metadata["filename"])
        else:
            raise KeyError(
                'The "metadata" field in the ground-truth annotations file should have a "StudyInstanceUID" or "filename" key to use as report ID! Abort program.'
            )
        for label in labels:
            all_labels_ground_truth.append(label + [filename])
        truth_label_dict[filename] = labels

    ground_truth_dict = {}
    model_dict = {}
    correct_dict = {}
    for item in [
        ("names", "<PERSOON>"),
        ("dates", "<DATUM>"),
        ("times", "<TIJD>"),
        ("phonenumbers", "<TELEFOONNUMMER>"),
        ("patientnumbers", "<PATIENTNUMMER>"),
        ("znumbers", "<ZNUMMER>"),
        ("locations", "<PLAATS>"),
        ("signatures", "<GEAUTORISEERD>"),
    ]:
        ground_truth_dict[item[0]] = [
            i for i in all_labels_ground_truth if item[1] in i
        ]
        model_dict[item[0]] = [i for i in all_labels_model if item[1] in i]

    # Calculate scores and generate table
    table = []
    errors = []
    for item in [
        ("Name", "names"),
        ("Date", "dates"),
        ("Time", "times"),
        ("Phone number", "phonenumbers"),
        ("Patient number", "patientnumbers"),
        ("Z-number", "znumbers"),
        ("Location", "locations"),
    ]:

        # If a text span annotation from a name falls within a signature annotation, then it counts as a true positive
        if item[1] == "names":
            names_in_signatures = [
                n
                for n in ground_truth_dict[item[1]]
                if any(
                    [
                        s
                        for s in model_dict["signatures"]
                        if ((n[0] >= s[0]) and (n[1] <= s[1]) and (n[3] == s[3]))
                    ]
                )
            ]
            model_dict[item[1]] += names_in_signatures

        tp_items = [i for i in model_dict[item[1]] if i in ground_truth_dict[item[1]]]
        fp_items = [
            i for i in model_dict[item[1]] if i not in ground_truth_dict[item[1]]
        ]
        fn_items = [
            i for i in ground_truth_dict[item[1]] if i not in model_dict[item[1]]
        ]
        tp, fp, fn = len(tp_items), len(fp_items), len(fn_items)
        errors += fp_items
        errors += fn_items

        precision = round(tp / ((tp + fp) + 1e-16), 2)
        recall = round(tp / ((tp + fn) + 1e-16), 2)
        fraction = round(
            (len(ground_truth_dict[item[1]]) / len(all_labels_ground_truth)) * 100, 2
        )

        table.append(
            [
                item[0],
                len(ground_truth_dict[item[1]]),
                fraction,
                "{} ({}/{})".format(precision, tp, tp + fp),
                "{} ({}/{})".format(recall, tp, tp + fn),
            ]
        )
    table.append(["Total", len(all_labels_ground_truth), "100", "-", "-"])

    # Convert table to dataframe and save this dataframe
    df_table = pd.DataFrame(
        table, columns=["PHI Tag", "Count", "Fraction (%)", "Precision", "Recall"]
    )
    pd.DataFrame.to_csv(
        df_table,
        os.path.join(
            args.output_dir, "Table_RRA_{}.csv".format(rra_version.replace(".", "_"))
        ),
        index=False,
    )
    # Convert list of labels to dataframe
    df_errors = pd.DataFrame(errors, columns=["s_start", "s_end", "tag", "filename"])

    # Save scores and examples
    report_path = os.path.abspath(
        os.path.join(
            args.output_dir,
            "Test_report_RRA_{}.txt".format(rra_version.replace(".", "_")),
        )
    )

    with open(report_path, "w", encoding="utf-8") as file:

        # Write header
        file.write("1. INFO:\n\n")
        file.write(
            "Test report created on {}\n".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        file.write(
            "Annotations from Radiology Report Anonymizer (RRA) {}\n".format(
                rra_version
            )
        )
        file.write("Test database: {}\n\n".format(ground_truth_name))

        # Write table
        file.write("2. RESULTS:\n\n")
        file.write("Number of test reports: {}\n".format(len(annotations)))
        file.write(
            str(tabulate(df_table, headers="keys", tablefmt="psql", showindex=False))
            + "\n\n"
        )

        # Write examples
        failure_cases = set(df_errors["filename"])
        file.write("3. FAILURE CASES ({} reports):\n\n".format(len(failure_cases)))
        window = 20

        for name in list(failure_cases):

            text = corpus_dict[name]
            labels_model = model_label_dict[name]
            labels_truth = truth_label_dict[name]

            file.write("\nFilename: {}\n".format(name))
            file.write("Model tags: {}\n".format(labels_model))
            file.write("Truth tags: {}\n".format(labels_truth))

            fp_labels = [i for i in labels_model if i not in labels_truth]
            fn_labels = [i for i in labels_truth if i not in labels_model]

            for error_type in ["fp", "fn"]:

                if error_type == "fp":
                    labels = fp_labels
                else:
                    labels = fn_labels

                for label in labels:
                    start_idx = label[0]
                    end_idx = label[1]
                    tag = label[2]
                    tag_length = end_idx - start_idx
                    window_start_idx = max(0, start_idx - window)
                    window_end_idx = min(len(text), end_idx + window)

                    model_text = text[:start_idx] + tag + text[end_idx:]

                    original_snippet = text[window_start_idx:window_end_idx]
                    original_snippet = original_snippet.replace("\n", " ")
                    anonymized_snippet = model_text[
                        window_start_idx : window_end_idx + len(tag) - tag_length
                    ]
                    anonymized_snippet = anonymized_snippet.replace("\n", " ")

                    if error_type == "fp":
                        file.write("fp (model): ...{}...\n".format(anonymized_snippet))
                        file.write("fp (original): ...{}...\n".format(original_snippet))
                    else:
                        file.write("fn (truth): ...{}...\n".format(anonymized_snippet))
                        file.write("fn (original): ...{}...\n".format(original_snippet))
