# -*- coding: utf-8 -*-
# pylint: disable=relative-beyond-top-level
"""Parsing functions for the oxidation state mining project"""

import concurrent.futures
import re
from collections import defaultdict
from typing import Tuple

import numpy as np
from ccdc import io  # pylint: disable=import-error
from numeral import roman2int
from tqdm import tqdm

from .utils import SymbolNameDict


class GetOxStatesCSD:
    """Main parsing class"""

    def __init__(self, cds_ids: list) -> None:
        """Parses CSD structures for oxidation states

        Args:
            cds_ids (list): list of CSD database identifiers

        Returns:
            None
        """
        # Set up dictionaries and regex
        self.symbol_name_dict = SymbolNameDict().get_symbol_name_dict()
        self.name_symbol_dict = {v: k for k, v in self.symbol_name_dict.items()}
        symbol_regex = '|'.join(list(self.symbol_name_dict.values()))
        self.symbol_regex = re.compile(symbol_regex)
        self.regex = re.compile('((?:{})\\([iv0]+\\))'.format(symbol_regex), re.IGNORECASE)
        self.not_ox_regex = re.compile('((?:{})[^\\(]*$)'.format(symbol_regex), re.IGNORECASE)
        self.negative_regex = re.compile('((?:{})\\(-[1234567890]+\\))'.format(symbol_regex), re.IGNORECASE)
        self.csd_ids = cds_ids
        self.csd_reader = io.EntryReader('CSD')

    def get_symbol_ox_number(self, parsed_string: str) -> Tuple[str, int]:
        """Splits a parser hit into symbol and ox nuber and returns
        latter as a integer

        Args:
            parsed_string (str): regex match of the form metalname(romanoxidationstate)

        Returns:
            str: symbol
            int: oxidation number

        """
        name, roman = parsed_string.strip(')').split('(')
        if roman != '0':
            return self.name_symbol_dict[name.lower()], roman2int(roman)
        else:
            return self.name_symbol_dict[name.lower()], int(0)

    def get_symbol_negative_ox_number(self, parsed_string: str) -> Tuple[str, int]:
        name, roman = parsed_string.strip(')').split('(')
        return self.name_symbol_dict[name.lower()], int(roman)

    def get_symbol_nan(self, parsed_string: str) -> Tuple[str, int]:
        name = self.symbol_regex.findall(parsed_string)[0]
        return self.name_symbol_dict[name.lower()], np.nan

    def parse_name(self, chemical_name_string: str) -> dict:
        """Takes the chemical name string from the CSD database and returns,
        if it finds it, a dictionary with the oxidation states for the metals

        Args:
            chemical_name_string (str): full chemical name

        Returns:
            dict: dictionary of  symbol: oxidation states (list)

        """

        oxidation_state_dict = defaultdict(list)

        matches = re.findall(self.regex, chemical_name_string)
        for match in matches:
            symbol, oxidation_int = self.get_symbol_ox_number(match)
            oxidation_state_dict[symbol].append(oxidation_int)

        no_ox_matches = re.findall(self.not_ox_regex, chemical_name_string)
        for match in no_ox_matches:
            symbol, oxidation_int = self.get_symbol_nan(match)
            oxidation_state_dict[symbol].append(oxidation_int)

        matches = re.findall(self.negative_regex, chemical_name_string)
        for match in matches:
            symbol, oxidation_int = self.get_symbol_negative_ox_number(match)
            oxidation_state_dict[symbol].append(oxidation_int)

        return dict(oxidation_state_dict)

    def parse_csd_entry(self, database_id: str) -> dict:
        """Looks up a CSD id and runs the parsing

        Args:
            database_id (str): CSD database identifier

        Returns:
            dict: symbol - oxidation state dictionary

        Exception:
            returns empy dict upon exception (if it cannot find the structure in the database)

        """
        try:
            entry_object = self.csd_reader.entry(database_id)  # pylint:disable=no-member
            name = entry_object.chemical_name
            return self.parse_name(name)
        except Exception:  # pylint: disable=broad-except
            return {}

    def run_parsing(self, njobs=4):
        """Runs (concurrent) parsing over the list of database identifiers.

        Args:
            njobs (int): maximum number of parallel workers

        Returns:
            dict: nested dictionary  with {'id': {'symbol': [oxidation states]}}

        """
        results_dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=njobs) as executor:
            for database_id, result in tqdm(
                    zip(self.csd_ids, executor.map(self.parse_csd_entry, self.csd_ids)),
                    total=len(self.csd_ids),
            ):
                results_dict[database_id] = result
        return results_dict
