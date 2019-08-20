# -*- coding: utf-8 -*-
"""
Some general utility functions for the oxidation state mining project
"""
from __future__ import absolute_import
import os
import json
from pathlib import Path
from pymatgen.core import Element


class SymbolNameDict():
    """
    Parses the periodic table json and returns  a dictionary with symbol: longname
    """

    def __init__(self):
        with open(os.path.join(Path(__file__).absolute().parent, 'data', 'periodic_table.json'),
                  'r') as periodic_table_file:
            self.pt_data = json.load(periodic_table_file)
        self.symbol_name_dict = {}

    def get_symbol_name_dict(self, only_metal=True):
        """
        Iterates over keys and returns the symbol: name dict.
        """
        for key, value in self.pt_data.iterdict():
            if only_metal:
                if Element(key).is_metal:
                    self.symbol_name_dict[key] = value['Name'].lower()
            else:
                self.symbol_name_dict[key] = value['Name'].lower()

        return self.symbol_name_dict
