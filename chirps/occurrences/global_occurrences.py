import sqlite3
import argparse
import logging

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

from chirps.utils import init_default_logger

BIRD_DEFINITIONS_TABLE = "bird_definition_items"
WORLD_OCCURRENCES_TABLE = "world_occurrence_items"
LOCALIZATION_TABLE = "localization_items"


class GlobalOccurrences(CLIChirp, ChirpNode):
    data_path = "data/occurrences.db"
    db = None

    def get_key_by_scientific_name(self, scientific_name: str):
        cursor = self.db.cursor()
        query = f"SELECT gbif_species_key FROM {BIRD_DEFINITIONS_TABLE} WHERE LOWER(species_name) = LOWER(?)"
        params = [scientific_name]

        cursor.execute(query, params)

        result = cursor.fetchone()

        if result is None:
            logging.error(
                f"No matching scientific name found for '{scientific_name}'.")
            return None  # No matching scientific name found

        gbif_species_key = result[0]
        logging.info(
            f"Found gbif_species_key: {gbif_species_key} for scientific name: {scientific_name}")
        return gbif_species_key

    def get_localized_name(self, gbif_species_key: int, language_code: str):
        cursor = self.db.cursor()
        query = f"SELECT localized_name FROM {LOCALIZATION_TABLE} WHERE gbif_key = ? AND LOWER(language_code) = LOWER(?)"
        params = [gbif_species_key, language_code]

        cursor.execute(query, params)

        result = cursor.fetchone()

        if result is None:
            logging.warning(
                f"No localized name found for gbif_species_key: {gbif_species_key} and language_code: {language_code}.")
            return None  # No localized name found

        localized_name = result[0]
        logging.info(
            f"Found localized_name: {localized_name} for gbif_species_key: {gbif_species_key} and language_code: {language_code}")
        return localized_name

    def process(self, input_data: dict) -> dict:
        self.load_database()
        
        # get ID from scientific name from BIRD_DEFINITIONS_TABLE
        scientific_name = input_data.get("scientific_name")
        if not scientific_name:
            logging.error("No scientific name provided in input_data.")
            return None

        gbif_species_key = self.get_key_by_scientific_name(scientific_name)
        if gbif_species_key is None:
            return None

        # Query the BIRD_DEFINITIONS_TABLE to get the ID for the given scientific name case insensitive
        # optionally filter by country_code and month if provided in input_data
        country_code = input_data.get("country_code")
        month = input_data.get("month")

        cursor = self.db.cursor()
        query = f"SELECT gbif_species_key FROM {BIRD_DEFINITIONS_TABLE} WHERE LOWER(species_name) = LOWER(?)"
        params = [scientific_name]

        cursor.execute(query, params)

        result = cursor.fetchone()

        if result is None:
            logging.error(
                f"No matching scientific name found for '{scientific_name}'.")
            return None  # No matching scientific name found

        # get column with name gbif_species_key
        gbif_species_key = result[0]
        logging.info(
            f"Found gbif_species_key: {gbif_species_key} for scientific name: {scientific_name}")

        # Query the WORLD_OCCURRENCES_TABLE to get occurrences for the given gbif_species_key and sum count column
        query = f"SELECT SUM(count) FROM {WORLD_OCCURRENCES_TABLE} WHERE gbif_species_key = ?"
        params = [gbif_species_key]

        if country_code:
            query += " AND LOWER(country_code) = LOWER(?)"
            params.append(country_code)

        if month:
            query += " AND month = ?"
            params.append(month)

        cursor.execute(query, params)

        occurrences_result = cursor.fetchone()

        if occurrences_result is None or occurrences_result[0] is None:
            logging.error(
                f"No occurrences found for gbif_species_key: {gbif_species_key}")
            return None  # No occurrences found for the given gbif_species_key

        total_occurrences = occurrences_result[0]

        target_language_code = input_data.get("language", "en")
        localized_name = self.get_localized_name(
            gbif_species_key, target_language_code)
        if localized_name:
            logging.info(
                f"Localized name for gbif_species_key {gbif_species_key} in language '{target_language_code}': {localized_name}")
        else:
            logging.warning(
                f"No localized name found for gbif_species_key {gbif_species_key} in language '{target_language_code}'.")

        return {"scientific_name": scientific_name, "localized_name": localized_name, "total_occurrences": total_occurrences}

    def configure(self, input_config: dict):
        self.data_path = input_config.get("data_path", self.data_path)

    def load_database(self):
        if self.db is None:
            self.db = sqlite3.connect(self.data_path)

    def parse_args(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Multilanguage IOC Taxonomy Chirp Node")

        parser.add_argument("--scientific_name", type=str,
                            help="Scientific name to search for in the taxonomy")
        parser.add_argument("--country_code", type=str,
                            help="Country code to search for occurrences")
        parser.add_argument("--month", type=str,
                            help="Month to search for occurrences")
        parser.add_argument("--data_path", type=str, default=self.data_path,
                            help="Path to the SQLite database file (default: data/occurrences.db)")
        parser.add_argument("--language_code", type=str, default="en",
                            help="Language code to get localized name (default: en)")

        return parser

    def process_cli(self, args) -> None:
        self.configure({
            "data_path": args.data_path
        })

        if args.scientific_name:
            results = self.process({"scientific_name": args.scientific_name,
                                   "country_code": args.country_code, "month": args.month, "language": args.language_code})
            if results is None:
                print(
                    f"No results found for scientific name '{args.scientific_name}'.")
            else:
                print(
                    f"Results for scientific name '{args.scientific_name}': {results}")
        else:
            print("Please provide a scientific name to search for.")


if __name__ == "__main__":
    init_default_logger()
    node = GlobalOccurrences()
    parser = node.parse_args()
    args = parser.parse_args()
    node.process_cli(args)
