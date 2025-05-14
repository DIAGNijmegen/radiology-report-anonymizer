import re
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from unidecode import unidecode
from nltk import ngrams
import jsonlines

from utils.read_files import load_json


DEFAULT_CONFIG_DIR = Path(__file__).parents[1] / "config"


class Anonymizer:

    def __init__(
        self,
        config_dir=DEFAULT_CONFIG_DIR,
        entities_to_anonymize=[
            "person",
            "date",
            "time",
            "internal_phone_number",
            "patient_id",
            "z_number",
            "report_id",
            "location",
        ],
    ):
        """
        INITIALISATION

        When initialising the anonymizer, load all resources (family names, Dutch cities, etc.).
        Create a dictionary that keeps track of all replacement operations.

        """

        self.version = "v1.0.0"
        self.config_dir = Path(config_dir)

        print("Load resources")
        domain_dict = load_json(self.config_dir / "domains.json")
        whitelist_dict = load_json(self.config_dir / "whitelists.json")
        self.whitelist_full = set()
        self.whitelist_abbreviated = set()
        for domain, enabled in domain_dict.items():
            if enabled:
                self.whitelist_full = self.whitelist_full.union(
                    set(whitelist_dict[domain]["full"])
                )
                self.whitelist_abbreviated = self.whitelist_abbreviated.union(
                    set(whitelist_dict[domain]["abbreviated"])
                )
        self.whitelist = self.whitelist_full.copy()
        self.entities_to_anonymize = entities_to_anonymize
        self.dates = load_json(self.config_dir / "dates_lookup_lists.json")
        self.days = self.dates["days"]["full"]
        self.days_abbreviated = self.dates["days"]["abbreviated"]
        self.months = self.dates["months"]["full"]
        self.months_abbreviated = self.dates["months"]["abbreviated"]
        self.locations = load_json(self.config_dir / "locations_lookup_lists.json")
        self.names = load_json(self.config_dir / "names_lookup_lists.json")
        self.titles = load_json(self.config_dir / "titles_lookup_lists.json")
        self.cues = load_json(self.config_dir / "cue_lists.json")
        self.names["person_names"]["family_names"] = self.process_family_names(
            self.names["person_names"]["family_names"]
        )
        self.names["person_names"]["first_names"] = [
            name.strip().lower() for name in self.names["person_names"]["first_names"]
        ]
        self.names["person_names"]["first_names"] = set(
            [
                unidecode(x)
                for x in self.names["person_names"]["first_names"]
                if len(unidecode(x)) > 1
            ]
        )
        self.locations["dutch_cities"] = set(
            [x.strip().lower() for x in self.locations["dutch_cities"]]
        )
        self.recist_table_delimiters = load_json(
            self.config_dir / "recist_table_delimiters.json"
        )

        # Add days and months to the whitelist
        # So that "Mr. Wednesday" will be recognized as a name, but not "Wednesday"
        self.whitelist = self.whitelist.union(self.dates["days"], self.dates["months"])

        # keep a dictionary of replacement operations
        self.replace_dict = {}

        # Define counter for generating unique labels (always 5 digits)
        self.counter = 10000

        # Initialise original report variable
        self.original_report = ""

        # keep annotations for all reports as a list of dictionaries
        self.annotations = []

    def anonymize_report(self, report):
        """
        REPORT ANONYMIZER

        Main function for anonymizing Dutch radiology reports.
        The function accepts one report as a string and replaces all PHI.
        Please refer to README.md for more information about this function.

        """

        # Clean report (double spacings, etc.)
        report = self.clean_report(report)

        # Tokenize report and generate n-grams
        (
            report_1_grams,
            report_2_grams,
            report_3_grams,
            report_4_grams,
            report_5_grams,
        ) = self.tokenize_reports(report)

        # Keep a copy of the report
        self.original_report = report

        # Radboudumc reports may have sections with hard-to-detect names (e.g. initials attached to family name, etc.)
        # These sections should be processed in a very specific way, before running the rest of the code.
        if "person" in self.entities_to_anonymize:
            report = self.remove_report_status(report)
            report = self.remove_name_addendum(report)

        for five_gram in report_5_grams:
            if "person" in self.entities_to_anonymize:
                report = self.replace_family_names(report, five_gram)
                report = self.replace_initials(report, five_gram)
            if "location" in self.entities_to_anonymize:
                report = self.replace_cities(report, five_gram)

        for four_gram in report_4_grams:
            if "person" in self.entities_to_anonymize:
                report = self.replace_family_names(report, four_gram)
            if "date" in self.entities_to_anonymize:
                report = self.replace_dates(report, four_gram)
            if "person" in self.entities_to_anonymize:
                report = self.replace_initials(report, four_gram)
            if "location" in self.entities_to_anonymize:
                report = self.replace_cities(report, four_gram)

        for three_gram in report_3_grams:
            if "report_id" in self.entities_to_anonymize:
                report = self.replace_reportid(report, three_gram)
            if "person" in self.entities_to_anonymize:
                report = self.replace_titles(report, three_gram)
                report = self.replace_family_names(report, three_gram)
            if "internal_phone_number" in self.entities_to_anonymize:
                report = self.replace_phonenumbers(report, three_gram)
            if "time" in self.entities_to_anonymize:
                report = self.replace_time(report, three_gram)
            if "date" in self.entities_to_anonymize:
                report = self.replace_dates(report, three_gram)
            if "person" in self.entities_to_anonymize:
                report = self.replace_initials(report, three_gram)
            if "location" in self.entities_to_anonymize:
                report = self.replace_cities(report, three_gram)

        for two_gram in report_2_grams:
            if "report_id" in self.entities_to_anonymize:
                report = self.replace_reportid(report, two_gram)
            if "person" in self.entities_to_anonymize:
                report = self.replace_titles(report, two_gram)
                report = self.replace_cues(report, two_gram)
                report = self.replace_family_names(report, two_gram)
            if "time" in self.entities_to_anonymize:
                report = self.replace_time(report, two_gram)
            if "date" in self.entities_to_anonymize:
                report = self.replace_dates(report, two_gram)
            if "internal_phone_number" in self.entities_to_anonymize:
                report = self.replace_phonenumbers(report, two_gram)
            if "person" in self.entities_to_anonymize:
                report = self.replace_initials(report, two_gram)
            if "location" in self.entities_to_anonymize:
                report = self.replace_cities(report, two_gram)

        for token in report_1_grams:
            if "person" in self.entities_to_anonymize:
                report = self.replace_titles(report, token)
                report = self.replace_cues(report, token)
                report = self.replace_family_names(report, token)
            if "patient_id" in self.entities_to_anonymize:
                report = self.replace_patientnumber(report, token)
            if "z_number" in self.entities_to_anonymize:
                report = self.replace_znumber(report, token)
            if "internal_phone_number" in self.entities_to_anonymize:
                report = self.replace_phonenumbers(report, token)
            if "date" in self.entities_to_anonymize:
                report = self.replace_dates(report, token)
            if "time" in self.entities_to_anonymize:
                report = self.replace_time(report, token)
            if "person" in self.entities_to_anonymize:
                report = self.replace_initials(report, token)
            if "person" in self.entities_to_anonymize:
                report = self.replace_firstnames(report, token)
            if "location" in self.entities_to_anonymize:
                report = self.replace_cities(report, token)

        # Replace intermediate labels by final labels (i.e. full person names)
        report = self.replace_intermediate_labels(report)

        # Revert left-over intermediate labels to original token
        report = self.revert_intermediate_labels(report)

        # Change all tokens from <LABEL_[NUM]> to <LABEL>
        # Generate annotations
        report = self.postprocess_report(report)

        return report

    @staticmethod
    def process_family_names(family_names):
        """
        PRE-PROCESSING FAMILY NAMES

        Function for pre-processing family names. Names are lowercased and commas as removed.
        Alternative names are generated by moving chunks like "van der" etc. to the beginning of each name.
        Names that consist of a single letter are removed.

        """

        temp_family_names = []
        for i in family_names:
            i = i.lower().split(",")
            if len(i) > 1:
                name1 = " ".join([i[1].strip(), i[0].strip()])
                temp_family_names.append(name1)
                name2 = " ".join([i[0].strip(), i[1].strip()])
                temp_family_names.append(name2)
            else:
                name = i[0]
                temp_family_names.append(name)
        family_names = temp_family_names
        # Remove letter accents and names which consist of a single letter
        family_names = set(
            [unidecode(x) for x in family_names if len(unidecode(x)) > 1]
        )

        return family_names

    @staticmethod
    def tokenize_reports(report):
        """
        TOKENIZER AND N-GRAM GENERATOR

        Tokenize report and convert tokens into n-grams, namely 2-grams, 3-grams, 4-grams, 5-grams, and 6-grams.

        """

        # Tokenize each line with delimiters: , ; : ( ) ! ? .
        tokenized_sentences = []
        for sentence in report.split("\n"):
            tokenized_sentence = " ".join(re.split("[,;:()!?.]", sentence)).split()
            tokenized_sentences.append(tokenized_sentence)

        # Generate n-grams for each line and merge together as a single list of tuples
        report_1_grams = [
            token
            for tokenized_sentence in tokenized_sentences
            for token in ngrams(tokenized_sentence, 1)
        ]
        report_2_grams = [
            token
            for tokenized_sentence in tokenized_sentences
            for token in ngrams(tokenized_sentence, 2)
        ]
        report_3_grams = [
            token
            for tokenized_sentence in tokenized_sentences
            for token in ngrams(tokenized_sentence, 3)
        ]
        report_4_grams = [
            token
            for tokenized_sentence in tokenized_sentences
            for token in ngrams(tokenized_sentence, 4)
        ]
        report_5_grams = [
            token
            for tokenized_sentence in tokenized_sentences
            for token in ngrams(tokenized_sentence, 5)
        ]

        return (
            report_1_grams,
            report_2_grams,
            report_3_grams,
            report_4_grams,
            report_5_grams,
        )

    @staticmethod
    def clean_report(report_temp):
        # Remove double spaces to ensure proper replacements
        report_temp = re.sub(" +", " ", report_temp)
        # Remove whitespace around newlines
        report_temp = re.sub(" *\n *", "\n", report_temp)
        # Remove whitespace, whitespace, etc. at start document
        newlines = re.search("\n+", report_temp)
        if newlines is not None:
            if newlines.start() == 0:
                report_temp = re.sub(newlines.group(), "", report_temp, 1)
        report_temp = re.sub(" *\n *", "\n", report_temp)
        # Remove white space around hyphens and slashes to address faulty grammar like Hendrix -Verberne
        report_temp = re.sub(" *- *", "-", report_temp)
        report_temp = re.sub(" */ *", "/", report_temp)

        return report_temp

    """
    REGEX FUNCTIONS
    
    Regular expressions for replacing names, dates, etc. in the reports. 
    
    """

    # Replace string in report_temp (not attached to hyphen or slashes or underscores)
    def replace_text(self, report_temp, input_text, t_id):
        input_text = str(input_text)
        matches = re.findall(
            r"\b"
            + "(?<!/)(?<!-)(?<!_)"
            + re.escape(input_text)
            + r"\b"
            + "(?!/)(?!-)(?!_)",
            report_temp,
        )
        if matches:
            for match in matches:
                t_id = re.sub("\$|([0-9]+)", str(self.counter), t_id)
                report_temp = re.sub(
                    r"\b"
                    + "(?<!/)(?<!-)(?<!_)"
                    + re.escape(match)
                    + r"\b"
                    + "(?!/)(?!-)(?!_)",
                    t_id,
                    report_temp,
                    1,
                )
                self.replace_dict[t_id] = match
                self.counter += 1

        return report_temp

    # Replace string in report_temp with no regex escape at all
    def replace_text_no_escape(self, report_temp, input_text, t_id):
        input_text = str(input_text)
        matches = re.findall(re.escape(input_text), report_temp)
        if matches:
            for match in matches:
                t_id = re.sub("\$|([0-9]+)", str(self.counter), t_id)
                report_temp = re.sub(re.escape(match), t_id, report_temp, 1)
                self.replace_dict[t_id] = match
                self.counter += 1

        return report_temp

    # Replace string in report_temp with no regex escape at the end (for strings that end with a dot)
    def replace_text_end_dot(self, report_temp, input_text, t_id):
        input_text = str(input_text)
        matches = re.findall(r"\b" + re.escape(input_text), report_temp)
        if matches:
            for match in matches:
                t_id = re.sub("\$|([0-9]+)", str(self.counter), t_id)
                report_temp = re.sub(r"\b" + re.escape(match), t_id, report_temp, 1)
                self.replace_dict[t_id] = match
                self.counter += 1

        return report_temp

    # When ALL hits need to be replaced at once, which is the case for initials and titles
    def replace_text_all_end_dot(self, report_temp, input_text, t_id):
        input_text = str(input_text)
        matches = re.findall(r"\b" + re.escape(input_text), report_temp)
        if matches:
            for match in matches:
                t_id = re.sub("\$|([0-9]+)", str(self.counter), t_id)
                report_temp = re.sub(r"\b" + re.escape(match), t_id, report_temp, 1)
                self.replace_dict[t_id] = match
                self.counter += 1

        return report_temp

    # Replace initials that are not preceded or followed by a hyphen
    def replace_text_no_hyphen(self, report_temp, input_text, t_id):
        input_text = str(input_text)
        exception_regex = "(?<!" + ".)(?<!".join(self.whitelist_abbreviated) + ".)"

        # Check whether input string ends with a dot
        if input_text[-1] == ".":
            matches = re.findall(
                r"\b"
                + "(?<!-)"
                + "{}".format(exception_regex)
                + re.escape(input_text)
                + "(?!-)",
                report_temp,
            )
            if matches:
                for match in matches:
                    t_id = re.sub("\$|([0-9]+)", str(self.counter), t_id)
                    report_temp = re.sub(
                        r"\b"
                        + "(?<!-)"
                        + "{}".format(exception_regex)
                        + re.escape(match)
                        + "(?!-)",
                        t_id,
                        report_temp,
                        1,
                    )
                    self.replace_dict[t_id] = match
                    self.counter += 1
        else:
            matches = re.findall(
                r"\b"
                + "(?<!-)"
                + "{}".format(exception_regex)
                + re.escape(input_text)
                + r"\b"
                + "(?!-)",
                report_temp,
            )
            if matches:
                for match in matches:
                    t_id = re.sub("\$|([0-9]+)", str(self.counter), t_id)
                    report_temp = re.sub(
                        r"\b"
                        + "(?<!-)"
                        + "{}".format(exception_regex)
                        + re.escape(match)
                        + r"\b"
                        + "(?!-)",
                        t_id,
                        report_temp,
                        1,
                    )
                    self.replace_dict[t_id] = match
                    self.counter += 1

        return report_temp

    # Replace string in report_temp (not attached to hyphen) with exceptions (like "leeftijd")
    def replace_text_with_exceptions(
        self,
        report_temp,
        input_text,
        lookbehind_exception_strings,
        lookahead_exception_strings,
        t_id,
        recist=False,
    ):
        input_text = str(input_text)
        lookbehind_exception_string_capitals = [
            s.upper() for s in lookbehind_exception_strings if s.isalpha()
        ]
        lookahead_exception_string_capitals = [
            s.upper() for s in lookahead_exception_strings if s.isalpha()
        ]

        full_lookbehind_exception_string = ""
        for exception_string in (
            lookbehind_exception_strings + lookbehind_exception_string_capitals
        ):
            full_lookbehind_exception_string += "(?<!{} )".format(exception_string)
            full_lookbehind_exception_string += "(?<!{} [(])".format(exception_string)
        full_lookahead_exception_string = ""
        for exception_string in (
            lookahead_exception_strings + lookahead_exception_string_capitals
        ):
            full_lookahead_exception_string += "(?! {})".format(exception_string)

        begin_table_index = len(report_temp)
        end_table_index = -1
        if recist and self.recist_table_delimiters["begin_word"] in report_temp.lower():
            begin_table_index = report_temp.lower().index(
                self.recist_table_delimiters["begin_word"]
            )
            if self.recist_table_delimiters["end_word"] in report_temp.lower():
                end_table_index = report_temp.lower().index(
                    self.recist_table_delimiters["end_word"]
                )
            else:
                end_table_index = len(report_temp) - 1

        matches = re.finditer(
            r"\b"
            + "(?<!/)(?<!-)(?<!_)"
            + full_lookbehind_exception_string
            + re.escape(input_text)
            + full_lookahead_exception_string
            + "(?!/)(?!-)(?!_)"
            + r"\b",
            report_temp,
        )
        matches = list((match.start(), match.end(), match.group()) for match in matches)

        if matches:
            for start_idx, end_idx, match_value in matches:
                if start_idx > end_table_index or end_idx < begin_table_index:
                    t_id = re.sub("\$|([0-9]+)", str(self.counter), t_id)
                    report_temp = re.sub(
                        r"\b"
                        + "(?<!/)(?<!-)(?<!_)"
                        + full_lookbehind_exception_string
                        + re.escape(match_value)
                        + full_lookahead_exception_string
                        + "(?!/)(?!-)(?!_)"
                        + r"\b",
                        t_id,
                        report_temp,
                        1,
                    )
                    self.replace_dict[t_id] = match_value
                    self.counter += 1

        return report_temp

    # Replace person names
    def replace_text_persons(self, report_temp, pattern, t_id):
        pattern = str(pattern)
        matches = re.findall(pattern, report_temp)
        if matches:
            for match in matches:
                t_id = re.sub("\$|([0-9]+)", str(self.counter), t_id)
                report_temp = re.sub(re.escape(match), t_id, report_temp, 1)
                self.replace_dict[t_id] = match
                self.counter += 1

        return report_temp

    """
    RADBOUDUMC SPECIFIC FUNCTIONS
    
    Radboudumc reports contain report status sections and addendums with difficult to anonymize functions. 
    The functions below are designed to detect these sections and process them before running the other 
    anonymizaiton functions. 
    
    """

    def remove_report_status(self, report):
        """
        In Radboudumc reports, there is a special case with the following pattern:
        - Some report
        - Two newlines (but there can be more)
        - Name with initials attached to it (like "HENDRIXABC")
        - "Verslagstatus" followed by name

        """

        # Add newlines if the report contains less than 4 lines
        # Because we are going to create groups of 4 lines each
        report_lines = report.split("\n")
        while len(report_lines) < 4:
            report_lines.append("")

        # Generate groups of 4 lines each and check for the earlier described pattern
        pointer = 0
        for i in range(len(report_lines) - 3):
            line_4_gram = report_lines[pointer : 4 + pointer]
            original_line_4_gram = line_4_gram
            line_4_gram = [s.strip() for s in line_4_gram]
            if (
                line_4_gram[0] == ""
                and line_4_gram[1] == ""
                and "verslagstatus" in line_4_gram[3].lower()
            ):
                if len(line_4_gram[2]) > 0:
                    if line_4_gram[2][0].isupper():
                        t_id = "<GEAUTORISEERD_{}>".format(self.counter)
                        report = re.sub("\n".join(original_line_4_gram), t_id, report)
                        self.replace_dict[t_id] = "\n".join(original_line_4_gram)
                        self.counter += 1
            pointer += 1

        return report

    def remove_name_addendum(self, report):
        """
        Some Radboudumc reports have an addendum header with difficult to anonymize names, such as "Sar, J.M. van der"
        Replace entire addendum header with tag <PERSOON>.

        """

        report_lines = report.split("\n")
        new_report = []
        for line in report_lines:
            if "ADDENDUM: >>" in line:
                # Take everything between "door" and a newline
                matches = re.findall("(?<=door )[^0-9]+", line)
                if matches:
                    t_id = "<PERSOON_{}>".format(self.counter)
                    line = re.sub(matches[0], t_id, line)
                    self.replace_dict[t_id] = matches[0]
                    self.counter += 1
            new_report.append(line)
        report = "\n".join(new_report)

        return report

    """
    FUNCTIONS FOR ANONYMISING TEXTS
    
    Various functions for replacing privacy-sensitive information with (intermediate) labels.  
    
    label: description
    
    <TITELS_ACHTER_$>       Titles used after names
    <TITELS_VOOR_$>         Titles used before names
    <TITELS_VOOR_ACHTER_$>  Titles used before and after names
    <PERSOON_HINT_VOOR_$>   Cues used before names
    <PERSOON_HINT_ACHTER_$> Cues used after names
    <INITIALEN_$>           Initials
    <ACHTERNAAM_$>          Family names
    <VOORNAAM_$>            First names
    <W_ACHTERNAAM_$>        Whitelist family names
    <RAPPORT_ID_$>          Report IDs
    <DATUM_$>               Dates
    <TIJD_$>                Time of the day
    <TELEFOONNUMMER_$>      Internal phonenumbers
    <ZNUMMER_$>             Z-numbers (Radboudumc specific)
    <PATIENTNUMMER_$>       Patient numbers
    <PLAATS_$>              Dutch city names
    
    $ will be replaced by counter.
    If labels are not part of a name, then these will be reverted back. 
    Timestamps, z-numbers, phonenumbers, and dates are directly replaced.
    
    """

    def replace_family_names(self, report, tokens):

        # Get full string from n-gram
        full_string = " ".join(tokens)

        # Replace family names by unique ID
        # <ACHTERNAAM$> : Family name
        # <W_ACHTERNAAM$> : Whitelisted family name
        # $ will be replaced by counter
        id_1 = "<ACHTERNAAM_$>"
        id_2 = "<W_ACHTERNAAM_$>"

        # check if one of the tokens starts with a capital letter
        capitals_exist = False
        for i in range(len(tokens)):
            if tokens[i][0].isupper():
                capitals_exist = True

        # remove double family names that are separated by "-" or "/" (like Hendrix-Verberne)
        tokens_double = re.split("[-/]", full_string)
        if len(tokens_double) == 2:
            if (
                unidecode(tokens_double[0].lower())
                in self.names["person_names"]["family_names"]
                and unidecode(tokens_double[1].lower())
                in self.names["person_names"]["family_names"]
                and capitals_exist
            ):
                if (
                    unidecode(tokens_double[0].lower()) not in self.whitelist
                    and unidecode(tokens_double[1].lower()) not in self.whitelist
                ):
                    # Check first whether family name is within parentheses
                    new_family_name_parentheses = "({})".format(full_string)
                    report = self.replace_text_no_escape(
                        report, new_family_name_parentheses, id_1
                    )
                    # Replace family name
                    report = self.replace_text(report, full_string, id_1)
                else:
                    # Check first whether family name is within parentheses
                    new_family_name_parentheses = "({})".format(full_string)
                    report = self.replace_text_no_escape(
                        report, new_family_name_parentheses, id_2
                    )
                    # Replace family name
                    report = self.replace_text(report, full_string, id_2)

        # Create additional alternative names (nested list)
        # Convert the abbreviation "vd" and "v/d" into "van der", "van de", and "van den"
        # Convert the abbreviation "v" in "van"
        tokens_van_der = [
            re.sub(r"\b" + "vd" + r"\b", "van der", full_string).split(),
            re.sub(r"\b" + "v/d" + r"\b", "van der", full_string).split(),
        ]
        tokens_van_de = [
            re.sub(r"\b" + "vd" + r"\b", "van de", full_string).split(),
            re.sub(r"\b" + "v/d" + r"\b", "van de", full_string).split(),
        ]
        tokens_van_den = [
            re.sub(r"\b" + "vd" + r"\b", "van den", full_string).split(),
            re.sub(r"\b" + "v/d" + r"\b", "van den", full_string).split(),
        ]
        tokens_van = re.sub(r"\b" + "v" + r"\b", "van", full_string).split()
        all_tokens = [
            full_string.split(),
            tokens_van_der[0],
            tokens_van_der[1],
            tokens_van_de[0],
            tokens_van_de[1],
            tokens_van_den[0],
            tokens_van_den[1],
            tokens_van,
        ]
        all_tokens = [
            list(tokens_alt)
            for tokens_alt in set(tuple(tokens) for tokens in all_tokens)
        ]

        # Replace all family names
        matching_tokens = [
            tokens_alt
            for tokens_alt in all_tokens
            if " ".join(tokens_alt).lower()
            in self.names["person_names"]["family_names"]
        ]
        ngram_in_familylist = any(matching_tokens)
        if ngram_in_familylist and capitals_exist:
            # in case the name is separated by a comma
            # generate list of family names with different comma positions
            new_family_names = []
            if len(tokens) > 1:
                for i in range(len(tokens) - 1):
                    family_name_tokens = [tokens[j] for j in range(len(tokens))]
                    family_name_tokens[i] = family_name_tokens[i] + ","
                    family_name = " ".join(family_name_tokens)
                    new_family_names.append(family_name)
            new_family_names.append(" ".join(tokens))
            # Revert list of family names (start with comma near the end)
            new_family_names = new_family_names[::-1]

            # Replace family name by unique id (in-text replacement)
            for new_family_name in new_family_names:
                if new_family_name.lower() not in self.whitelist:
                    # Check first whether family name is within parentheses
                    new_family_name_parentheses = "({})".format(new_family_name)
                    report = self.replace_text_no_escape(
                        report, new_family_name_parentheses, id_1
                    )
                    # Check whether "v" appears in familyname and replace with dot
                    if new_family_name[0] == "v":
                        new_family_name_dot = "".join(
                            [new_family_name[0], ".", new_family_name[1:]]
                        )
                        report = self.replace_text_no_escape(
                            report, new_family_name_dot, id_1
                        )
                    # Replace family name
                    report = self.replace_text(report, new_family_name, id_1)
                else:
                    # Check first whether family name is within parentheses
                    new_family_name_parentheses = "({})".format(new_family_name)
                    report = self.replace_text_no_escape(
                        report, new_family_name_parentheses, id_2
                    )
                    # Replace family name
                    report = self.replace_text(report, new_family_name, id_2)
        return report

    def replace_initials(self, report, tokens):

        # Replace initials by unique ID <INITIALEN$>
        # $ will be replaced by counter
        id_1 = "<INITIALEN_$>"

        # Check if all tokens are capital letters
        if not any([token for token in tokens if not token.isalpha()]) and not any(
            [token for token in tokens if not token.isupper()]
        ):

            # Check for a group of initials with no delimiters or a single initial
            # Max 5 letters (more than 5 initials is unlikely)
            # Should not be included in the initials exception list
            if len(tokens[0]) < 6 and len(tokens) == 1:
                if tokens[0] not in self.whitelist_abbreviated:
                    # If initials are followed by a dot, but not preceded or followed by a hyphen
                    new_token = tokens[0] + "."
                    report = self.replace_text_no_hyphen(report, new_token, id_1)
                    # If initials are not followed by a dot, and not preceded or followed by a hyphen
                    new_token = tokens[0]
                    report = self.replace_text_no_hyphen(report, new_token, id_1)

            # Check if all tokens consist of a single letter
            if not any([token for token in tokens if len(token) > 1]):
                # If all tokens are separated with a dot and whitespace characters
                new_token = ". ".join(tokens) + "."
                report = self.replace_text_all_end_dot(report, new_token, id_1)
                # If all tokens are separated with a dot
                new_token = ".".join(tokens) + "."
                report = self.replace_text_all_end_dot(report, new_token, id_1)
                # If all tokens are separated with a dot, but the n-gram does not end with a dot
                new_token = ".".join(tokens)
                report = self.replace_text(report, new_token, id_1)
                # If all tokens are separated with a whitespace character
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_1)

        return report

    def replace_dates(self, report, tokens):

        # For all patterns:
        # the year is between 1900 and 2099
        # the day and month are in the day and month list

        # Replace dates by unique ID <DATUM_$>
        # $ Will be replaced by counter
        id_1 = "<DATUM_$>"

        # Look for 4-gram date patterns
        if len(tokens) == 4:

            # Pattern: day of the week (incl. abbreviation) +
            # day of the month (max 2 digits) + month (incl. abbreviation) + year (4 digits)
            if (
                (tokens[3][:2] == "19" or tokens[3][:2] == "20")
                and len(tokens[1]) <= 2
                and len(tokens[3]) == 4
                and tokens[1].isdigit()
                and tokens[3].isdigit()
                and (
                    tokens[0].lower() in self.days
                    or tokens[0].lower() in self.days_abbreviated
                )
                and (
                    tokens[2].lower() in self.months
                    or tokens[2].lower() in self.months_abbreviated
                )
            ):
                # check if the day number is within a valid range
                if 1 <= int(tokens[1]) <= 31:
                    # in case the date contains abbreviations that end with a dot
                    new_token = " ".join(
                        [tokens[0] + ".", tokens[1], tokens[2] + ".", tokens[3]]
                    )
                    report = self.replace_text(report, new_token, id_1)
                    new_token = " ".join(
                        [tokens[0], tokens[1], tokens[2] + ".", tokens[3]]
                    )
                    report = self.replace_text(report, new_token, id_1)
                    new_token = " ".join(
                        [tokens[0] + ".", tokens[1], tokens[2], tokens[3]]
                    )
                    report = self.replace_text(report, new_token, id_1)
                    # in case the date does not contain abbreviations that end with a dot
                    new_token = " ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)

        # Look for 3-gram date patterns
        if len(tokens) == 3:

            # Pattern: day of the month (max 2 digits) + month (incl. abbreviation) + year (4 digits)
            if (
                (tokens[2][:2] == "19" or tokens[2][:2] == "20")
                and len(tokens[0]) <= 2
                and len(tokens[2]) == 4
                and tokens[0].isdigit()
                and tokens[2].isdigit()
                and (
                    tokens[1].lower() in self.months
                    or tokens[1].lower() in self.months_abbreviated
                )
            ):
                # check if the day number is within a valid range
                if 1 <= int(tokens[0]) <= 31:
                    # in case the date contains abbreviations that end with a dot
                    new_token = " ".join([tokens[0], tokens[1] + ".", tokens[2]])
                    report = self.replace_text(report, new_token, id_1)
                    # in case the date does not contain abbreviations that end with a dot
                    new_token = " ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)

            # Pattern: day of the month (max 2 digits) + month (incl. abbreviation) + year (2 digits)
            if (
                len(tokens[0]) <= 2
                and len(tokens[2]) == 2
                and tokens[0].isdigit()
                and tokens[2].isdigit()
                and (
                    tokens[1].lower() in self.months
                    or tokens[1].lower() in self.months_abbreviated
                )
            ):
                # check if the day number is within a valid range
                if 1 <= int(tokens[0]) <= 31:
                    # in case the date contains abbreviations that end with a dot
                    new_token = " ".join([tokens[0], tokens[1] + ".", tokens[2]])
                    report = self.replace_text(report, new_token, id_1)
                    # in case the date does not contain abbreviations that end with a dot
                    new_token = " ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)

            # Pattern: day of the week (incl. abbreviation) +
            # day of the month (max 2 digits) + month (incl. abbreviation)
            if (
                len(tokens[1]) <= 2
                and tokens[1].isdigit()
                and (
                    tokens[0].lower() in self.days
                    or tokens[0].lower() in self.days_abbreviated
                )
                and (
                    tokens[2].lower() in self.months
                    or tokens[2].lower() in self.months_abbreviated
                )
            ):
                # check if the day number is within a valid range
                if 1 <= int(tokens[1]) <= 31:
                    # in case the date contains abbreviations that end with a dot
                    if tokens[2].lower() in self.months_abbreviated:
                        new_token = " ".join(
                            [tokens[0] + ".", tokens[1], tokens[2] + "."]
                        )
                        report = self.replace_text_end_dot(report, new_token, id_1)
                        new_token = " ".join([tokens[0], tokens[1], tokens[2] + "."])
                        report = self.replace_text_end_dot(report, new_token, id_1)
                    new_token = " ".join([tokens[0] + ".", tokens[1], tokens[2]])
                    report = self.replace_text(report, new_token, id_1)
                    # in case the date does not contain abbreviations that end with a dot
                    new_token = " ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)

            # Pattern: day of the month (max 2 digits) + month (max 2 digits) + year (2 digits)
            if (
                len(tokens[0]) <= 2
                and len(tokens[1]) <= 2
                and len(tokens[2]) == 2
                and tokens[0].isdigit()
                and tokens[1].isdigit()
                and tokens[2].isdigit()
            ):
                # check if the day and month number are within a valid range
                if (1 <= int(tokens[0]) <= 31) and (1 <= int(tokens[1]) <= 12):
                    # in case the digits are separated by dots plus whitespace
                    new_token = ". ".join(tokens) + "."
                    report = self.replace_text_end_dot(report, new_token, id_1)
                    new_token = ". ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
                    # in case the digits are separated by dots
                    new_token = ".".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
                    # in case the digits are separated by whitespace
                    new_token = " ".join(tokens)
                    report = self.replace_text_with_exceptions(
                        report,
                        new_token,
                        self.cues["negative_lookbehinds"]["dates"],
                        self.cues["negative_lookaheads"]["dates"],
                        id_1,
                        recist=True,
                    )

            # Pattern: day of the month (max 2 digits) + month (max 2 digits) + year (4 digits)
            if (
                len(tokens[0]) <= 2
                and len(tokens[1]) <= 2
                and len(tokens[2]) == 4
                and tokens[0].isdigit()
                and tokens[1].isdigit()
                and tokens[2].isdigit()
                and (tokens[2][:2] == "19" or tokens[2][:2] == "20")
            ):
                # check if the day and month number are within a valid range
                if (1 <= int(tokens[0]) <= 31) and (1 <= int(tokens[1]) <= 12):
                    # in case the digits are separated by dots plus whitespace
                    new_token = ". ".join(tokens) + "."
                    report = self.replace_text_end_dot(report, new_token, id_1)
                    new_token = ". ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
                    # in case the digits are separated by dots
                    new_token = ".".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
                    # in case the digits are separated by whitespace
                    new_token = " ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)

        # Look for 2-gram date patterns
        if len(tokens) == 2:

            # Pattern: month (incl. abbreviation) + year
            if (
                len(tokens[1]) == 4
                and tokens[1].isdigit()
                and (
                    tokens[0].lower() in self.months
                    or tokens[0].lower() in self.months_abbreviated
                )
                and (tokens[1][:2] == "19" or tokens[1][:2] == "20")
            ):
                # in case the date contains abbreviations that end with a dot
                new_token = " ".join([tokens[0] + ".", tokens[1]])
                report = self.replace_text(report, new_token, id_1)
                # in case the date does not contain abbreviations that end with a dot
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_1)

            # Pattern: day + month (incl. abbreviation)
            if (
                len(tokens[0]) <= 2
                and tokens[0].isdigit()
                and (
                    tokens[1].lower() in self.months
                    or tokens[1].lower() in self.months_abbreviated
                )
            ):

                # check if the day number is within a valid range
                if 1 <= int(tokens[0]) <= 31:
                    # in case the date contains abbreviations that end with a dot
                    if tokens[1].lower() in self.months_abbreviated:
                        new_token = " ".join([tokens[0], tokens[1] + "."])
                        report = self.replace_text_end_dot(report, new_token, id_1)
                    # in case the date does not contain abbreviations that end with a dot
                    new_token = " ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)

            # Pattern: day of the month (max 2 digits) + month (max 2 digits) + year (4 digits),
            # where day and month are separated by a hyphen or slash, and year is separated by a whitespace
            # pattern may not be preceded by exceptions (e.g. leeftijd)
            if "-" in tokens[0]:
                date_text = tokens[0].split("-")
            else:
                date_text = tokens[0].split("/")
            if len(date_text) == 2:
                if (
                    len(date_text[0]) <= 2
                    and len(date_text[1]) <= 2
                    and len(tokens[1]) == 4
                    and date_text[0].isdigit()
                    and date_text[1].isdigit()
                    and tokens[1].isdigit()
                    and (tokens[1][:2] == "19" or tokens[1][:2] == "20")
                ):
                    # check if the day and month number are within a valid range
                    if (1 <= int(date_text[0]) <= 31) and (
                        1 <= int(date_text[1]) <= 12
                    ):
                        new_token = " ".join(tokens)
                        lookbehind_nodigit_exception_list = [
                            pattern
                            for pattern in self.cues["negative_lookbehinds"]["dates"]
                            if not bool(re.search(r"\d", pattern))
                        ]
                        report = self.replace_text_with_exceptions(
                            report,
                            new_token,
                            lookbehind_nodigit_exception_list,
                            [],
                            id_1,
                            recist=True,
                        )

        # Look for 1-gram date patterns
        if len(tokens) == 1:

            if "-" in tokens[0]:
                date_text = tokens[0].split("-")
            else:
                date_text = tokens[0].split("/")

            # Pattern: day of the month (max 2 digits) + month (incl. abbreviation) + year (2 digits),
            # separated by slashes or hyphens
            if len(date_text) == 3:
                if (
                    date_text[0].isdigit()
                    and date_text[2].isdigit()
                    and len(date_text[0]) <= 2
                    and len(date_text[2]) == 2
                    and (
                        date_text[1].lower() in self.months
                        or date_text[1].lower() in self.months_abbreviated
                    )
                ):
                    # check if the day number is within a valid range
                    if 1 <= int(date_text[0]) <= 31:
                        new_token = tokens[0]
                        report = self.replace_text(report, new_token, id_1)

            # Pattern: year (4 digits) + month (max 2 digits) + day of the month (max 2 digits),
            # separated by slashes or hyphens
            if len(date_text) == 3:
                if (
                    len(date_text[0]) == 4
                    and len(date_text[1]) <= 2
                    and len(date_text[2]) <= 2
                    and date_text[0].isdigit()
                    and date_text[1].isdigit()
                    and date_text[2].isdigit()
                    and (date_text[0][:2] == "19" or date_text[0][:2] == "20")
                ):
                    # check if the day and month number are within a valid range
                    if (1 <= int(date_text[2]) <= 31) and (
                        1 <= int(date_text[1]) <= 12
                    ):
                        new_token = tokens[0]
                        report = self.replace_text(report, new_token, id_1)

            # Pattern: day of the month (max 2 digits) + month (incl. abbreviation) + year (4 digits),
            # separated by slashes or hyphens
            if len(date_text) == 3:
                if (
                    date_text[0].isdigit()
                    and date_text[2].isdigit()
                    and len(date_text[0]) <= 2
                    and len(date_text[2]) == 4
                    and (date_text[2][:2] == "19" or date_text[2][:2] == "20")
                    and (
                        date_text[1].lower() in self.months
                        or date_text[1].lower() in self.months_abbreviated
                    )
                ):
                    # check if the day number is within a valid range
                    if 1 <= int(date_text[0]) <= 31:
                        new_token = tokens[0]
                        report = self.replace_text(report, new_token, id_1)

            # Pattern: day of the month (max 2 digits) + month (max 2 digits) + year (4 digits),
            # separated by slashes or hyphens
            if len(date_text) == 3:
                if (
                    len(date_text[0]) <= 2
                    and len(date_text[1]) <= 2
                    and len(date_text[2]) == 4
                    and date_text[0].isdigit()
                    and date_text[1].isdigit()
                    and date_text[2].isdigit()
                    and (date_text[2][:2] == "19" or date_text[2][:2] == "20")
                ):
                    # check if the day and month number are within a valid range
                    if (1 <= int(date_text[0]) <= 31) and (
                        1 <= int(date_text[1]) <= 12
                    ):
                        new_token = tokens[0]
                        report = self.replace_text(report, new_token, id_1)

            # Pattern: day of the month (max 2 digits) + month (max 2 digits) + year (2 digits),
            # separated by slashes or hyphens
            if len(date_text) == 3:
                if (
                    len(date_text[0]) <= 2
                    and len(date_text[1]) <= 2
                    and len(date_text[2]) == 2
                    and date_text[0].isdigit()
                    and date_text[1].isdigit()
                    and date_text[2].isdigit()
                ):
                    # check if the day and month number are within a valid range
                    if (1 <= int(date_text[0]) <= 31) and (
                        1 <= int(date_text[1]) <= 12
                    ):
                        new_token = tokens[0]
                        report = self.replace_text(report, new_token, id_1)

            # Pattern: day of the month (max 2 digits) + year (4 digits), separated by slashes or hyphens
            # But not preceded by exceptions(e.g. leeftijd)
            if len(date_text) == 2:
                if (
                    len(date_text[0]) <= 2
                    and len(date_text[1]) == 4
                    and date_text[0].isdigit()
                    and date_text[1].isdigit()
                    and (date_text[1][:2] == "19" or date_text[1][:2] == "20")
                ):
                    # check if the day number is within a valid range
                    if 1 <= int(date_text[0]) <= 31:
                        new_token = tokens[0]
                        report = self.replace_text_with_exceptions(
                            report,
                            new_token,
                            self.cues["negative_lookbehinds"]["dates"],
                            self.cues["negative_lookaheads"]["dates"],
                            id_1,
                            recist=True,
                        )

            # Pattern: day of the month (max 2 digits) + month (max 2 digits), separated by slashes or hyphens
            # But not preceded by exceptions(e.g. leeftijd)
            if len(date_text) == 2:
                if (
                    len(date_text[0]) <= 2
                    and len(date_text[1]) <= 2
                    and date_text[0].isdigit()
                    and date_text[1].isdigit()
                ):
                    # check if the day and month number are within a valid range
                    if (1 <= int(date_text[0]) <= 31) and (
                        1 <= int(date_text[1]) <= 12
                    ):
                        new_token = tokens[0]
                        report = self.replace_text_with_exceptions(
                            report,
                            new_token,
                            self.cues["negative_lookbehinds"]["dates"],
                            self.cues["negative_lookaheads"]["dates"],
                            id_1,
                            recist=True,
                        )

            # Pattern: year (4 digits)
            # Year is not preceded by a forward slash or hyphen
            if (
                len(tokens[0]) == 4
                and tokens[0].isdigit()
                and (tokens[0][:2] == "19" or tokens[0][:2] == "20")
            ):
                new_token = tokens[0]
                report = self.replace_text(report, new_token, id_1)

            # Pattern: months (incl. abbreviations)
            if (
                tokens[0].lower() in self.months
                or tokens[0].lower() in self.months_abbreviated
            ):

                # in case the date contains abbreviations that end with a dot
                if tokens[0].lower() in self.months_abbreviated:
                    new_token = tokens[0] + "."
                    report = self.replace_text_end_dot(report, new_token, id_1)

                # in case the date does not contain abbreviations that end with a dot
                new_token = tokens[0]
                report = self.replace_text(report, new_token, id_1)

        return report

    def replace_titles(self, report, tokens):

        # Get full string from n-gram
        full_string = " ".join(tokens)

        # Replace titles by unique ID
        # <TITELS_VOOR_ACHTER_$> Titles used before and after names (like "radioloog")
        # <TITELS_VOOR_$> : Titles used before names (like "Mr")
        # <TITELS_ACHTER_$> : Titles used after names (like "Msc")
        # $ Will be replaced by counter
        id_1 = "<TITELS_VOOR_ACHTER_$>"
        id_2 = "<TITELS_VOOR_$>"
        id_3 = "<TITELS_ACHTER_$>"

        titles_before = []
        titles_after = []
        titles_full = []
        for key in self.titles["before"]:
            titles_before += self.titles["before"][key]["full"]
            titles_before += self.titles["before"][key]["abbreviated"]
            titles_full += self.titles["before"][key]["full"]
        for key in self.titles["after"]:
            titles_after += self.titles["after"][key]["full"]
            titles_after += self.titles["after"][key]["abbreviated"]
            titles_full += self.titles["after"][key]["full"]

        # Look for 2-gram and 3-gram patterns
        if len(tokens) == 3 or len(tokens) == 2:

            # remove titles commonly used before and after name
            if (
                full_string.lower() in titles_before
                and full_string.lower() in titles_after
            ):

                if full_string.lower() not in titles_full:
                    # in case tokens are separated by dots and white spaces
                    new_token = ". ".join(tokens) + "."
                    report = self.replace_text_end_dot(report, new_token, id_1)
                    # in case tokens are separated by dots
                    new_token = ".".join(tokens) + "."
                    report = self.replace_text_end_dot(report, new_token, id_1)
                # in case tokens are separated by white spaces
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_1)

            # remove titles commonly used before name
            if full_string.lower() in titles_before:
                if full_string.lower() not in titles_full:
                    # in case tokens are separated by dots and white spaces
                    new_token = ". ".join(tokens) + "."
                    report = self.replace_text_end_dot(report, new_token, id_2)
                    # in case tokens are separated by dots
                    new_token = ".".join(tokens) + "."
                    report = self.replace_text_end_dot(report, new_token, id_2)
                # in case tokens are separated by white spaces
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_2)

            # remove titles commonly used before name
            if full_string.lower() in titles_after:
                if full_string.lower() not in titles_full:
                    # in case tokens are separated by dots and white spaces
                    new_token = ". ".join(tokens) + "."
                    report = self.replace_text_end_dot(report, new_token, id_3)
                    # in case tokens are separated by dots
                    new_token = ".".join(tokens) + "."
                    report = self.replace_text_end_dot(report, new_token, id_3)
                # in case tokens are separated by white spaces
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_3)

        if len(tokens) == 1:

            # remove titles used before and after names
            if tokens[0].lower() in titles_before and tokens[0].lower() in titles_after:
                if tokens[0].lower() not in titles_full:
                    # If a title follows with a dot
                    new_token = tokens[0] + "."
                    report = self.replace_text_end_dot(report, new_token, id_1)

                # If a title is enclosed by brackets
                new_token = "({})".format(tokens[0])
                report = self.replace_text_no_escape(report, new_token, id_3)

                # If a title does not follow with a dot
                report = self.replace_text(report, tokens[0], id_1)

            # remove titles used before names
            if tokens[0].lower() in titles_before:
                if tokens[0].lower() not in titles_full:
                    # If a title follows with a dot
                    new_token = tokens[0] + "."
                    report = self.replace_text_end_dot(report, new_token, id_2)
                # If a title does not follow with a dot
                report = self.replace_text(report, tokens[0], id_2)

            # remove titles used after names
            if tokens[0].lower() in titles_after:
                if tokens[0].lower() not in titles_full:
                    # If a title follows with a dot
                    new_token = tokens[0] + "."
                    report = self.replace_text_end_dot(report, new_token, id_3)

                # If a title is enclosed by brackets
                new_token = "({})".format(tokens[0])
                report = self.replace_text_no_escape(report, new_token, id_3)

                # If a title does not follow with a dot
                report = self.replace_text(report, tokens[0], id_3)

        return report

    def replace_phonenumbers(self, report, tokens):

        # Replace phone numbers by unique ID <TELEFOONNUMMER_$>
        # $ Will be replaced by counter
        id_1 = "<TELEFOONNUMMER_$>"

        if len(tokens) == 3:

            # remove beeper numbers preceded by * with spacing
            # the number consists of 4 digits
            # the number is preceded by the word "sein"
            if (
                len(tokens[2]) == 4
                and tokens[2].isdigit()
                and tokens[1] == "*"
                and tokens[0].lower()
                in self.cues["positive_lookbehinds"]["phone_numbers"]
            ):
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_1)

        if len(tokens) == 2:

            # remove beeper numbers preceded by * without spacing
            # the number consists of 4 digits
            # the number is preceded by the word "sein"
            if (
                len(tokens[1]) == 5
                and tokens[1][-4:].isdigit()
                and tokens[1][0] == "*"
                and tokens[0].lower()
                in self.cues["positive_lookbehinds"]["phone_numbers"]
            ):
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_1)

            # remove beeper numbers
            # the number consists of 4 digits
            # the number is preceded by the word "sein", "nummer", "zijn", "haar", etc.
            if (
                len(tokens[1]) == 4
                and tokens[1].isdigit()
                and tokens[0].lower()
                in self.cues["positive_lookbehinds"]["phone_numbers"]
            ):
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_1)

        if len(tokens) == 1:

            token = tokens[0]
            # the number consists of 5 digits
            if len(token) == 5 and token.isdigit():
                report = self.replace_text(report, token, id_1)

        return report

    def replace_time(self, report, tokens):

        # Replace time of the day by unique ID <TIJD_$>
        # $ Will be replaced by counter
        id_1 = "<TIJD_$>"

        if len(tokens) == 3:

            # remove numeric times with hours, minutes and seconds, e.g. "14:12:00"
            # hours, minutes, and seconds consist of 2 digits
            if (
                len(tokens[0]) <= 2
                and len(tokens[1]) == 2
                and len(tokens[2]) == 2
                and tokens[0].isdigit()
                and tokens[1].isdigit()
                and tokens[2].isdigit()
            ):
                # check if the hours, minutes and seconds are within a valid range
                if (
                    (0 <= int(tokens[0]) <= 23)
                    and (0 <= int(tokens[1]) <= 59)
                    and (0 <= int(tokens[2]) <= 59)
                ):
                    # in case the digits are separated by ":"
                    new_token = ":".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
                    # in case the digits are separated by dots"
                    new_token = ".".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
                    # in case the digits are separated by white spaces
                    new_token = " ".join(tokens)
                    report = self.replace_text_with_exceptions(
                        report,
                        new_token,
                        self.cues["negative_lookbehinds"]["times"],
                        self.cues["negative_lookaheads"]["times"],
                        id_1,
                        recist=True,
                    )

        if len(tokens) == 2:

            # remove numeric times with hours and minutes (e.g. "11:00")
            # hours consist of 1 or 2 digits
            # minutes consist of 2 digits
            if (
                len(tokens[0]) <= 2
                and len(tokens[1]) == 2
                and tokens[0].isdigit()
                and tokens[1].isdigit()
            ):
                # check if the hours and minutes are within a valid range
                if (0 <= int(tokens[0]) <= 23) and (0 <= int(tokens[1]) <= 59):
                    # in case the digits are separated by ":"
                    new_token = ":".join(tokens) + " uur"
                    report = self.replace_text(report, new_token, id_1)
                    new_token = ":".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
                    # in case the digits are separated by dots
                    new_token = ".".join(tokens) + " uur"
                    report = self.replace_text(report, new_token, id_1)
                    new_token = ".".join(tokens)
                    report = self.replace_text_with_exceptions(
                        report,
                        new_token,
                        self.cues["negative_lookbehinds"]["times"],
                        self.cues["negative_lookaheads"]["times"],
                        id_1,
                        recist=True,
                    )
                    # in case the digits are separated by white spaces
                    new_token = " ".join(tokens) + " uur"
                    report = self.replace_text(report, new_token, id_1)

            # remove numeric times with hours and minutes with "u" (e.g. "11:00u")
            # hours consist of 1 or 2 digits
            # minutes consist of 2 digits
            if (
                len(tokens[0]) <= 2
                and len(tokens[1]) == 3
                and tokens[0].isdigit()
                and tokens[1][:-1].isdigit()
                and tokens[1][-1] == "u"
            ):
                # check if the hours and minutes are within a valid range
                if (0 <= int(tokens[0]) <= 23) and (0 <= int(tokens[1][:-1]) <= 59):
                    # in case the digits are separated by ":"
                    new_token = ":".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
                    # in case the digits are separated by dots
                    new_token = ".".join(tokens)
                    report = self.replace_text(report, new_token, id_1)

            # Remove semi-numeric time notations (e.g. "12 uur")
            # hours consist of 1 or 2 digits
            # hours are followed by the word "uur"
            if (
                len(tokens[0]) == 4
                and tokens[0].isdigit()
                and tokens[1].lower() == "uur"
            ):
                # check if the time is within a valid range
                if (1 <= int(str(tokens[0])[:2]) <= 23) and (
                    0 <= int(str(tokens[0])[2:]) <= 59
                ):
                    new_token = " ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)
            elif (
                len(tokens[0]) <= 2
                and tokens[0].isdigit()
                and tokens[1].lower() == "uur"
            ):
                # check if the time is within a valid range
                if 1 <= int(tokens[0]) <= 23:
                    new_token = " ".join(tokens)
                    report = self.replace_text(report, new_token, id_1)

        if len(tokens) == 1:

            # remove time like "15u15"
            # the tags consists of 5 characters
            # the middle character is a "u
            # the first two and last two characters are digits
            new_tokens = re.split("(u)", tokens[0])
            if len(new_tokens) == 3:
                if (
                    new_tokens[0].isdigit()
                    and new_tokens[1] == "u"
                    and new_tokens[2].isdigit()
                    and len(new_tokens[2]) == 2
                    and len(new_tokens[0]) <= 2
                ):
                    # check if the hours and minutes are within a valid range
                    if (0 <= int(new_tokens[0]) <= 23) and (
                        0 <= int(new_tokens[2]) <= 59
                    ):
                        new_token = tokens[0]
                        report = self.replace_text(report, new_token, id_1)

        return report

    def replace_cues(self, report, tokens):

        # Get full string from n-gram
        full_string = " ".join(tokens)

        # Replace cues by unique ID <PERSOON_HINT_VOOR_$> or <PERSOON_HINT_ACHTER_$>
        # $ Will be replaced by counter
        id_1 = "<PERSOON_HINT_VOOR_$>"
        id_2 = "<PERSOON_HINT_ACHTER_$>"

        cues_before = self.cues["positive_lookbehinds"]["person_names"]
        cues_after = self.cues["positive_lookaheads"]["person_names"]

        if len(tokens) > 1:

            # Identify cues before names
            if full_string.lower() in cues_before:
                # in case parts of the title are separated by white spaces
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_1)

            # Identify cues after names
            if full_string.lower() in cues_after:
                # in case parts of the title are separated by white spaces
                new_token = " ".join(tokens)
                report = self.replace_text(report, new_token, id_2)

        if len(tokens) == 1:

            # Identify cues before names
            if full_string.lower() in cues_before:
                report = self.replace_text(report, tokens[0], id_1)

            # Identify cues after names
            if full_string.lower() in cues_after:
                report = self.replace_text(report, tokens[0], id_2)

        return report

    def replace_patientnumber(self, report, tokens):

        # Replace patient numbers by unique ID <PATIENTNUMMER_$>
        # $ Will be replaced by counter
        id_1 = "<PATIENTNUMMER_$>"
        token = tokens[0]
        # the number consists of 7 digits
        if len(token) == 7 and token.isdigit():
            report = self.replace_text(report, token, id_1)

        return report

    def replace_znumber(self, report, tokens):

        # Replace z-numbers by unique ID <Z_NUMMER_$>
        # $ Will be replaced by counter
        id_1 = "<ZNUMMER_$>"
        token = tokens[0]
        # the number contains no more than 6 digits
        # the number starts with a "z"
        if len(token) == 7 and token[0].lower() == "z" and token[1:].isdigit():
            report = self.replace_text(report, token, id_1)

        return report

    def replace_firstnames(self, report, tokens):

        # Replace first names by unique ID <VOORNAAM_$>
        # $ Will be replaced by counter
        id_1 = "<VOORNAAM_$>"
        token = tokens[0]
        # the name is in the first name dictionary
        # (only) the first letter of the first name is capitalized
        # the name is not in the white list
        # the name is not in the days or months list
        if (
            token.lower() in self.names["person_names"]["first_names"]
            and token[0].isupper()
            and token.lower() not in self.whitelist
            and token.lower() not in self.days
            and token.lower() not in self.months
            and token.lower() not in self.months_abbreviated
        ):
            # in case the first name is in parentheses
            new_token = "({})".format(token)
            report = self.replace_text_no_escape(report, new_token, id_1)
            # in case the first name is not in parentheses
            report = self.replace_text(report, token, id_1)

        return report

    def replace_reportid(self, report, tokens):

        # Replace report IDs by unique ID <REPORT_ID_$>
        # $ Will be replaced by counter
        id_1 = "<RAPPORT_ID_$>"

        # Get full string from n-gram
        full_string = " ".join(tokens)

        # Re-split n-gram with hyphen as delimiter
        tokens = re.split("[-\s]", full_string)

        # Replace pattern like: "T 14-15616"
        if len(tokens) == 3:

            if (
                tokens[0] == "T"
                and tokens[1].isdigit()
                and len(tokens[1]) == 2
                and tokens[2].isdigit()
                and len(tokens[2]) == 5
            ):
                report = self.replace_text(report, full_string, id_1)

        return report

    def replace_cities(self, report, tokens):

        # Replace report IDs by unique ID <PLAATS_$>
        # $ Will be replaced by counter
        id_1 = "<PLAATS_$>"

        city_name = " ".join(tokens[1:])
        if (
            city_name.lower() in self.locations["dutch_cities"]
            and any([i for i in tokens[1:] if i[0].isupper()])
            and tokens[0] in self.cues["positive_lookbehinds"]["locations"]
        ):
            report = self.replace_text(report, city_name, id_1)

        return report

    """
    FUNCTIONS FOR REPLACING INTERMEDIATE LABELS WITH FINAL LABEL
    
    Various functions for replacing intermediate labels with the name placeholder <PERSOON_$>
    
    label: description
    
    <TITELS_ACHTER_$>       Titles used after names
    <TITELS_VOOR_$>         Titles used before names
    <TITELS_VOOR_ACHTER_$>  Titles used before and after names
    <PERSOON_HINT_VOOR_$>   Cues used before names
    <PERSOON_HINT_ACHTER_$> Cues used after names
    <INITIALEN_$>           Initials
    <ACHTERNAAM_$>          Family names
    <VOORNAAM_$>            First names
    <W_ACHTERNAAM_$>        Whitelist family names
    
    $ will be replaced by counter.
    If labels are not part of a name, then these will be reverted back. 
    
    """

    def replace_intermediate_labels(self, report):

        id_1 = "ACHTERNAAM"
        id_2 = "W_ACHTERNAAM"

        for t_id in [id_1, id_2]:
            # Six-gram patterns

            # in case the family name is preceded by 3 titles and initials, and followed by title.
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 3 titles and initials, and followed by title with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 3 titles (before/after) and initials,
            # and followed by title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 3 titles (before/after) and initials,
            # and followed by title (before/after) with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")

            # Five-gram patterns

            # in case the family name is preceded by initials and 3 titles.
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 4 titles
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles and initials, and followed by title
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles and initials, and followed by title with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles and initials, and followed by title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles and initials,
            # and followed by title (before/after) with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 3 titles, and followed by initials
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 3 titles, and followed by initials with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")

            # Four-gram patterns

            # in case the family name is preceded by initials and two titles
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by three titles
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles, and followed by initials
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles, and followed by initials with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles, and followed by 1 title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles, and followed by 1 title (before/after) with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles, and followed by 1 title
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles, and followed by 1 title with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for title and family name, followed by initials and first name
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<VOORNAAM_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for title and family name, followed by initials and first name, with a comma after family name
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<VOORNAAM_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by initials and title
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after), and followed by initials and title
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after),
            # and followed by initials and title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by initials and title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by initials and title.
            # Comma after family name.
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after), and followed by initials and title.
            # Comma after family name.
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after),
            # and followed by initials and title (before/after). Comma after family name.
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by initials and title (before/after).
            # Comma after family name.
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")

            # Three-gram patterns

            # in case the family name is preceded by 1 title and initials
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title and initials with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<INITIALEN_[0-9]+>,",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # 1 title followed by two family names
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after) and initials
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after) and initials with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<INITIALEN_[0-9]+>,",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # 1 title (before/after) followed by two family names
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 2 titles
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by initials
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by initials. Comma after family name.
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by initials, and followed by 1 title
            pattern = " ?".join(
                [
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by initials, and followed by 1 title with comma
            pattern = " ?".join(
                [
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by 1 title
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by 1 title with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for first name and family name combination, followed by 1 title
            pattern = " ?".join(
                [
                    "<VOORNAAM_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for mismatched first name and family name combination, followed by 1 title
            pattern = " ?".join(
                [
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for first name and family name combination, followed by 1 title with comma
            pattern = " ?".join(
                [
                    "<VOORNAAM_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for mismatched first name and family name combination, followed by 1 title with comma
            pattern = " ?".join(
                [
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for first name and family name combination, preceded by 1 title
            pattern = " ?".join(
                [
                    "<TITELS_ACHTER_[0-9]+>",
                    "<VOORNAAM_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for mismatched first name and family name combination, preceded by 1 title
            pattern = " ?".join(
                [
                    "<TITELS_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after), and followed by initials
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<INITIALEN_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after), and followed by initials.
            # Comma after family name.
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<INITIALEN_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by initials, and followed by 1 title (before/after)
            pattern = " ?".join(
                [
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by initials, and followed by 1 title (before/after) with comma
            pattern = " ?".join(
                [
                    "<INITIALEN_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by 1 title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after), and followed by 1 title with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title, and followed by 1 title (before/after) with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after), and followed by 1 title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by 1 title (before/after),
            # and followed by 1 title (before/after) with comma
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for first name and family name combination, followed by 1 title (before/after)
            pattern = " ?".join(
                [
                    "<VOORNAAM_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for mismatched first name and family name combination, followed by 1 title (before/after)
            pattern = " ?".join(
                [
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for first name and family name combination, followed by 1 title (before/after) with comma
            pattern = " ?".join(
                [
                    "<VOORNAAM_[0-9]+>",
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for mismatched first name and family name combination, followed by 1 title (before/after) with comma
            pattern = " ?".join(
                [
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for first name and family name combination, preceded by 1 title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<VOORNAAM_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for mismatched first name and family name combination, preceded by 1 title (before/after)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for two family names, followed by 1 title (before/after) (in case a first name is also a family name)
            pattern = " ?".join(
                [
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for two family names, followed by 1 title (before/after) with comma
            # (in case a first name is also a family name)
            pattern = " ?".join(
                [
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>,".format(t_id),
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for two family names, preceded by 1 title (before/after) (in case a first name is also a family name)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # In case family name is preceded by initials and first name
            pattern = " ?".join(
                ["<INITIALEN_[0-9]+>", "<VOORNAAM_[0-9]+>", "<{}_[0-9]+>".format(t_id)]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # In case family name is preceded by initials and mismatched first name
            pattern = " ?".join(
                [
                    "<INITIALEN_[0-9]+>",
                    "<ACHTERNAAM_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # In case two family names (true family name + whitelisted family name) is preceded by 1 title
            # (in case a first name is also a family name)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<W_ACHTERNAAM_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # In case two family names (true family name + whitelisted family name) is preceded by
            # 1 title (before/after)(in case a first name is also a family name)
            pattern = " ?".join(
                [
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<{}_[0-9]+>".format(t_id),
                    "<W_ACHTERNAAM_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # In case two family names (true family name + whitelisted family name) is followed by
            # 1 title (before/after)(in case a first name is also a family name)
            pattern = " ?".join(
                [
                    "<{}_[0-9]+>".format(t_id),
                    "<W_ACHTERNAAM_[0-9]+>",
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # In case two family names (true family name + whitelisted family name) is followed by
            # 1 title with comma (before/after)(in case a first name is also a family name)
            pattern = " ?".join(
                [
                    "<{}_[0-9]+>".format(t_id),
                    "<W_ACHTERNAAM_[0-9]+>,",
                    "<TITELS_VOOR_ACHTER_[0-9]+>",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Detect misspelled names
            pattern = "(?<=<PERSOON_HINT_VOOR_[0-9]{5}> <TITELS_VOOR_ACHTER_[0-9]{5}> )[A-Z]{1}[a-z]+"
            report = self.replace_text_persons(report, pattern, "<TEMP_PERSOON_$>")
            pattern = "(?<=<PERSOON_HINT_VOOR_[0-9]{5}>: <TITELS_VOOR_ACHTER_[0-9]{5}> )[A-Z]{1}[a-z]+"
            report = self.replace_text_persons(report, pattern, "<TEMP_PERSOON_$>")
            pattern = ".{,2}".join(
                [
                    "<PERSOON_HINT_VOOR_[0-9]+>",
                    "(<TITELS_VOOR_ACHTER_[0-9]+>",
                    "<TEMP_PERSOON_[0-9]+>)",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            pattern = "(?<=<PERSOON_HINT_VOOR_[0-9]{5}> <TITELS_VOOR_[0-9]{5}> )[A-Z]{1}[a-z]+"
            report = self.replace_text_persons(report, pattern, "<TEMP_PERSOON_$>")
            pattern = "(?<=<PERSOON_HINT_VOOR_[0-9]{5}>: <TITELS_VOOR_[0-9]{5}> )[A-Z]{1}[a-z]+"
            report = self.replace_text_persons(report, pattern, "<TEMP_PERSOON_$>")
            pattern = ".{,2}".join(
                [
                    "<PERSOON_HINT_VOOR_[0-9]+>",
                    "(<TITELS_VOOR_[0-9]+>",
                    "<TEMP_PERSOON_[0-9]+>)",
                ]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")

            # Two-gram patterns

            # in case the family name is preceded by only initials
            # (initials + whitelisted family name should be ignored)
            pattern = " ?".join(["<INITIALEN_[0-9]+>", "<ACHTERNAAM_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by only initials, with comma
            # (initials + whitelisted family name should be ignored)
            pattern = " ?".join(["<INITIALEN_[0-9]+>,", "<ACHTERNAAM_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by one title
            pattern = " ?".join(["<TITELS_VOOR_[0-9]+>", "<{}_[0-9]+>".format(t_id)])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by one title and comma
            pattern = " ?".join(["<TITELS_VOOR_[0-9]+>,", "<{}_[0-9]+>".format(t_id)])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is followed by one title
            pattern = " ?".join(["<{}_[0-9]+>".format(t_id), "<TITELS_ACHTER_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is followed by one title with comma
            pattern = " ?".join(["<{}_[0-9]+>,".format(t_id), "<TITELS_ACHTER_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by one title
            pattern = " ?".join(
                ["<TITELS_VOOR_ACHTER_[0-9]+>", "<{}_[0-9]+>".format(t_id)]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by one title and comma
            pattern = " ?".join(
                ["<TITELS_VOOR_ACHTER_[0-9]+>,", "<{}_[0-9]+>".format(t_id)]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is followed by one title (before/after)
            # (whitelisted family name should be ignored)
            pattern = " ?".join(["<ACHTERNAAM_[0-9]+>", "<TITELS_VOOR_ACHTER_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is followed by one title
            # (whitelisted family name should be ignored)
            pattern = " ?".join(["<ACHTERNAAM_[0-9]+>", "<TITELS_ACHTER_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is followed by one title with comma
            pattern = " ?".join(
                ["<{}_[0-9]+>,".format(t_id), "<TITELS_VOOR_ACHTER_[0-9]+>"]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is followed by initials
            # (initials + whitelisted family name should be ignored)
            pattern = " ?".join(["<ACHTERNAAM_[0-9]+>", "<INITIALEN_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is followed by initials with comma
            # (initials + whitelisted family name should be ignored)
            pattern = " ?".join(["<ACHTERNAAM_[0-9]+>,", "<INITIALEN_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for two consequent family names (initials are sometimes recognized as family names;
            # ignore whitelisted family names)
            pattern = " ?".join(["<ACHTERNAAM_[0-9]+>", "<ACHTERNAAM_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for two consequent family names with dot in between
            # (initials are sometimes recognized as family names; ignore whitelisted family names)
            pattern = " ?".join(["<ACHTERNAAM_[0-9]+>.", "<ACHTERNAAM_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for two consequent family names with comma in between
            # (initials are sometimes recognized as family names; ignore whitelisted family names)
            pattern = " ?".join(["<ACHTERNAAM_[0-9]+>,", "<ACHTERNAAM_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for first name and family name combination
            pattern = " ?".join(["<VOORNAAM_[0-9]+>", "<ACHTERNAAM_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # Look for family name followed by first name with comma
            pattern = " ?".join(["<ACHTERNAAM_[0-9]+>,", "<VOORNAAM_[0-9]+>"])
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # in case the family name is preceded by one cue
            pattern = " ?".join(
                ["<PERSOON_HINT_VOOR_[0-9]+>", "(<{}_[0-9]+>)".format(t_id)]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # In case the family name is followed by one cue, no dot
            pattern = ".?".join(
                ["(<ACHTERNAAM_[0-9]+>)", "<PERSOON_HINT_ACHTER_[0-9]+>"]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")
            # In case the family name is followed by one cue, with dot
            pattern = ".?".join(
                ["(<ACHTERNAAM_[0-9]+>)", ".<PERSOON_HINT_ACHTER_[0-9]+>"]
            )
            report = self.replace_text_persons(report, pattern, "<PERSOON_$>")

        return report

    """
    POSTPROCESSING: GENERATE FINAL REPORT AND REPLACEMENT DICTIONARY
    
    Change all tokens from <LABEL_[NUM]> to <LABEL> and prepare final replacement dictionary. 
    
    """

    def postprocess_report(self, report):

        # Reverse replacement operations to obtain text span annotations
        # Add text, labels, and text span annotations to annotation dictionary
        annotation_dict = {"text": self.original_report}
        report_copy = report
        pattern = "<[A-Z_]+[0-9]+>"
        matches = re.findall(pattern, report_copy)
        labels = []
        for match in matches:
            if "PERSOON" in match:
                span_start = report_copy.find(match)
                span_end = span_start + len(self.replace_dict[match])
                report_copy = report_copy.replace(match, self.replace_dict[match])
                sub_matches = [
                    i for i in re.findall(pattern, report_copy) if i not in matches
                ]
                # A person tag usually consists of two or more intermediate tags
                if any(sub_matches):
                    for sub_match in sub_matches:
                        span_end = report_copy.find(sub_match) + len(
                            self.replace_dict[sub_match]
                        )
                        report_copy = report_copy.replace(
                            sub_match, self.replace_dict[sub_match]
                        )
                labels.append([span_start, span_end, "<PERSOON>"])
            else:
                span_start = report_copy.find(match)
                span_end = span_start + len(self.replace_dict[match])
                report_copy = report_copy.replace(match, self.replace_dict[match])
                label = "_".join(match.split("_")[:-1]) + ">"
                labels.append([span_start, span_end, label])

        annotation_dict["labels"] = labels

        self.annotations.append(annotation_dict)

        # Clean tags in the report
        phi_keys = self.replace_dict.keys()
        for key in phi_keys:
            new_key = "_".join(key.split("_")[:-1]) + ">"
            report = re.sub(key, new_key, report)

        # Reset replace_dict for every report
        self.replace_dict = {}

        return report

    # Add filenames to annotation metadata
    def add_metadata(self, index, meta_dict):
        annotation_dict = self.annotations[index]
        meta_dict["rra_version"] = self.version
        annotation_dict["meta"] = meta_dict
        self.annotations[index] = annotation_dict

    """
    FUNCTION FOR REVERTING INTERMEDIATE LABELS
    
    All intermediate labels that are related to a person (titles, etc.) but were not replaced by the previous function, 
    are reverted back to their original state. 
    
    """

    def revert_intermediate_labels(self, report):

        matches_initials = re.findall("<INITIALEN_[0-9]+>", report)
        matches_titles_before = re.findall("<TITELS_VOOR_[0-9]+>", report)
        matches_titles_after = re.findall("<TITELS_ACHTER_[0-9]+>", report)
        matches_titles_before_and_after = re.findall(
            "<TITELS_VOOR_ACHTER_[0-9]+>", report
        )
        matches_names = re.findall("<ACHTERNAAM_[0-9]+>", report)
        matches_cues_before = re.findall("<PERSOON_HINT_VOOR_[0-9]+>", report)
        matches_cues_after = re.findall("<PERSOON_HINT_ACHTER_[0-9]+>", report)
        matches_firstnames = re.findall("<VOORNAAM_[0-9]+>", report)
        matches_whitenames = re.findall("<W_ACHTERNAAM_[0-9]+>", report)

        if matches_initials:
            for match in matches_initials:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)
        if matches_titles_before:
            for match in matches_titles_before:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)
        if matches_titles_after:
            for match in matches_titles_after:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)
        if matches_titles_before_and_after:
            for match in matches_titles_before_and_after:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)
        if matches_names:
            for match in matches_names:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)
        if matches_cues_before:
            for match in matches_cues_before:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)
        if matches_cues_after:
            for match in matches_cues_after:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)
        if matches_firstnames:
            for match in matches_firstnames:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)
        if matches_whitenames:
            for match in matches_whitenames:
                original_string = self.replace_dict[match]
                report = re.sub(match, original_string, report, 1)

        return report

    """
    FUNCTIONS FOR SAVING RESULTS
    
    Functions for saving the anonymized reports and generating test reports. 
    
    """

    @staticmethod
    def save_reports(reports, output_path):
        output_path = Path(output_path)
        # Save reports as a single json lines file or as separate txt files in output directory
        if output_path.suffix == ".jsonl":
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with jsonlines.open(output_path, "w") as writer:
                writer.write_all(reports)
        elif output_path.is_dir():
            exceptions_dir = output_path / "exceptions"
            exceptions_dir.mkdir(exist_ok=True, parents=True)
            for index, report in enumerate(tqdm(reports, total=len(reports))):
                # Write difficult cases to other folder
                if report["meta"]["flagged"]:
                    exception_path = (
                        exceptions_dir / f"{report['meta']['filename']}.txt"
                    )
                    with open(exception_path, "w", encoding="utf-8") as file:
                        file.write(report["text"])
                # Write all other cases to main output folder
                else:
                    main_path = output_path / f"{report['meta']['filename']}.txt"
                    with open(main_path, "w", encoding="utf-8") as file:
                        file.write(report["text"])
        else:
            raise Exception(
                "The specified output path is invalid: please specify a path to (1) a directory or "
                "(2) to a .jsonl file. Abort program."
            )

    def generate_testreport(
        self, testset_name, reports, original_reports, targets, tags, patterns
    ):
        filestring = ""
        success = False
        errors_found = False

        filestring += (
            "Test report created on {}".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
            + "\n"
        )
        filestring += (
            "Generated with Radiology Report Anonymizer (RRA) {}".format(self.version)
            + "\n"
        )
        filestring += "Results from file: {}".format(testset_name) + "\n\n"

        for test_index, report in enumerate(tqdm(reports, total=len(reports))):
            original_report = original_reports[test_index]
            target = targets[test_index]
            tag = tags[test_index]
            pattern = patterns[test_index]

            if report != target:
                errors_found = True
                if tag in report:
                    filestring += "CSV row no {}".format(test_index + 2) + "\n"
                    filestring += (
                        "ERROR: Replaced text with correct token, but string lengths do not match."
                        + "\n"
                    )
                    filestring += "PATTERN: {}".format(pattern) + "\n"
                    filestring += "INPUT STRING: {}".format(original_report) + "\n"
                    filestring += "OUTPUT STRING: {}".format(report) + "\n"
                    filestring += "TARGET STRING: {}".format(target) + "\n\n"
                elif report == original_report:
                    filestring += "CSV row no {}".format(test_index + 2) + "\n"
                    filestring += "ERROR: Failed to anonymize string." + "\n"
                    filestring += "PATTERN: {}".format(pattern) + "\n"
                    filestring += "INPUT STRING: {}".format(original_report) + "\n"
                    filestring += "OUTPUT STRING: {}".format(report) + "\n"
                    filestring += "TARGET STRING: {}".format(target) + "\n\n"
                else:
                    filestring += "CSV row no {}".format(test_index + 2) + "\n"
                    filestring += "ERROR: Replaced text with wrong token." + "\n"
                    filestring += "PATTERN: {}".format(pattern) + "\n"
                    filestring += "INPUT STRING: {}".format(original_report) + "\n"
                    filestring += "OUTPUT STRING: {}".format(report) + "\n"
                    filestring += "TARGET STRING: {}".format(target) + "\n\n"

        if not errors_found:
            success = True
            filestring += "No errors are found."

        return filestring, success
