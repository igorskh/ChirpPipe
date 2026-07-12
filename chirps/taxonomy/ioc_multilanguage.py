
import argparse

from chirps.chirp_node import ChirpNode
from chirps.cli_chirp import CLIChirp

import pandas as pd

from chirps.utils import init_default_logger


class IOCMultilanguageTaxonomy(CLIChirp, ChirpNode):
    data_path = "data/Multiling IOC 15.2.xlsx"
    df = None

    def process(self, input_data: dict) -> dict:
        pass

    def configure(self, input_config: dict):
        self.data_path = input_config.get("data_path", self.data_path)

    def load_df(self):
        if self.df is None:
            self.df = pd.read_excel(self.data_path, sheet_name="List")

    def parse_args(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Multilanguage IOC Taxonomy Chirp Node")

        parser.add_argument("--query", type=str,
                            help="Query string to search for in the taxonomy")
        parser.add_argument("--data_path", type=str, default=self.data_path,
                            help="Path to the Multilanguage IOC Excel file (default: data/Multiling IOC 15.2.xlsx)")
        parser.add_argument("--language", type=str, default="English",
                            help="Language to filter the taxonomy (default: English)")
        parser.add_argument("--list_languages", action="store_true",
                            help="List available languages in the taxonomy")

        return parser

    def process_cli(self, args) -> None:
        self.configure({
            "data_path": args.data_path
        })

        self.load_df()

        if args.list_languages:
            languages = self.get_languages()
            print("Available languages in the taxonomy:")
            for lang in languages:
                print(f"- {lang}")
        else:
            results = self.search_taxonomy(args.query, args.language)
            if results is None or results.empty:
                print(
                    f"No results found for query '{args.query}' in language '{args.language}'.")
            else:
                print(
                    f"Results for query '{args.query}' in language '{args.language}':")
                for index, row in results.iterrows():
                    print(f"Scientific Name: {row['IOC_15.2']}")
                    print(f"{args.language}: {row[args.language]}")

    def get_languages(self) -> list:
        self.load_df()
        # Exclude the first column (assumed to be 'Taxon')
        return list(self.df.columns[4:])

    def search_taxonomy(self, query: str, language: str = "English") -> pd.DataFrame:
        self.load_df()
        if language not in self.df.columns:
            print(
                f"Warning: Language '{language}' not found. Defaulting to English.")
            return None
        
        search_columns = self.df.columns[3:]
        # exclude the target language column from the search
        # search_columns = [col for col in search_columns if col != language]
        
        # fuzzy search across all columns except the target language
        mask = self.df[search_columns].apply(lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1)

        filtered_df = self.df[mask][["IOC_15.2", language]]

        return filtered_df


if __name__ == "__main__":
    init_default_logger()

    node = IOCMultilanguageTaxonomy()
    parser = node.parse_args()
    args = parser.parse_args()
    node.process_cli(args)
