# -*- coding: utf-8 -*-
"""
Get some structures and labels for solids.
I probably should have used one simple GET request instead of querying multiple times, but I'll go with it now,
it is not too slow
"""

import os
from itertools import product

import pandas as pd
from pymatgen import MPRester
from tqdm import tqdm

mp_api = MPRester(os.getenv('MP_API_KEY', None))  # pylint:disable=invalid-name

# Select metals and anions that are of interest for us
anions_dict = {  # pylint:disable=invalid-name
    'I': -1,
    'Cl': -1,
    'Br': -1,
    'F': -1,
    'O': -2,
    'S': -2,
    'N': -3,
}
metals = [  # pylint:disable=invalid-name
    'Li',
    'Na',
    'K',
    'Rb',
    'Cs',
    'Be',
    'Mg',
    'Ca',
    'Sr',
    'Ba',
    'Sc',
    'Ti',
    'V',
    'Cr',
    'Mn',
    'Fe',
    'Co',
    'Ni',
    'Cu',
    'Zn',
    'Y',
    'Zr',
    'Nb',
    'Mo',
    'Tc',
    'Ru',
    'Rh',
    'Pd',
    'Ag',
    'Cd',
    'Hf',
    'Ta',
    'W',
    'Re',
    'Os',
    'Ir',
    'Pt',
    'Au',
    'Hg',
    'B',
    'Al',
    'Ga',
    'In',
    'Tl',
    'Sn',
    'Pb',
    'Bi',
    'La',
    'Ce',
    'Pr',
    'Eu',
    'Gd',
    'Tb',
    'Dy',
    'Ho',
    'Er',
    'Tm',
    'U',
    'Pu',
]

anions = list(anions_dict.keys())  # pylint:disable=invalid-name


def check_stable(entry_id):
    """Check if energy is at hull minimum"""
    return mp_api.get_data(entry_id, prop='e_above_hull')[0]['e_above_hull'] == 0


def get_binary_combinations(  # pylint:disable=dangerous-default-value, redefined-outer-name
        metals: list = metals, anions: list = anions) -> list:
    """Create list of entries of binary compounds with the metals and anions we defined which are stable"""
    combinations = list(product(metals, anions))
    entries = []
    for combination in tqdm(combinations):
        combination_entries = mp_api.get_entries_in_chemsys(combination)
        for combination_entry in combination_entries:
            if check_stable(combination_entry.entry_id) and (len(combination_entry.as_dict()['composition']) > 1):
                entries.append(combination_entry.entry_id)
    return entries


def _check_consistency_ox_state(formula, oxidationstate, metal, anion):
    """Check if oxidation state is integer and makes sense in terms of charge neutrality"""
    check0 = formula[metal] * oxidationstate + formula[anion] * anions_dict[anion] == 0
    check1 = (oxidationstate).is_integer()
    return check0 * check1


def _figure_out_oxidation_state(formula, metal, anion):
    """Use charge neutrality, chemical formula and charge of anion to guess oxidation state"""
    negative_charge = formula[anion] * anions_dict[anion]
    positive_charge = -1 * negative_charge
    oxidation_state_guess = positive_charge / formula[metal]

    return float(oxidation_state_guess)


def calculate_metal_oxidation_state(formula: dict, metal: str, anion: str):
    """This returns the metal oxidation state for a composition dict"""
    # first check if first or second group, always set them to +1 and +2, respectively and see
    oxidationstate = None
    if metal in ['Li', 'Na', 'K', 'Rb', 'Cs']:
        if _check_consistency_ox_state(formula, 1.0, metal, anion):
            oxidationstate = 1.0
    elif metal in ['Be', 'Mg', 'Ca', 'Sr', 'Ba']:
        if _check_consistency_ox_state(formula, 2.0, metal, anion):
            oxidationstate = 2.0
    else:
        guess = _figure_out_oxidation_state(formula, metal, anion)
        print(guess)
        if _check_consistency_ox_state(formula, guess, metal, anion):
            oxidationstate = guess

    return oxidationstate


def which_is_the_metal(  # pylint:disable=dangerous-default-value
    formula,
    metals=metals,  # pylint:disable=redefined-outer-name
    anions=anions,  # pylint:disable=redefined-outer-name
):
    """Return metal, anion to have quicker access to the relevant keys of the formula dictionary"""
    metal = None
    anion = None
    for k in formula.keys():
        if k in metals:
            metal = k
        elif k in anions:
            anion = k

    return metal, anion


def collect_for_id(entry_id, outdir='mp_structures'):
    """Run the collections for one materials project id"""
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outdict = {}
    outdict['material_id'] = entry_id
    s = mp_api.get_structure_by_material_id(entry_id)  # pylint:disable=invalid-name
    formula_dict = dict(s.composition.get_el_amt_dict())  # it returns a defaultdict
    metal, anion = which_is_the_metal(formula_dict)
    outdict['metal'] = metal
    outdict['anion'] = anion
    formula_string = s.formula
    outdict['formula'] = formula_string
    oxidationstate = calculate_metal_oxidation_state(formula_dict, metal, anion)
    outdict['oxidationstate'] = oxidationstate
    name = entry_id + formula_string.replace(' ', '_')
    outdict['name'] = name
    if oxidationstate is not None:
        s.to(filename=os.path.join(outdir, name + '.cif'))
    return outdict


def collect_entries():
    """Runs the whole thing"""
    print('*** Starting collect entries for all binary combinations and check if they are stable ***')
    entries = get_binary_combinations()
    print('*** Now iterating over all the entries to find out oxidation states ***')
    results = []
    for entry in entries:
        outdict = collect_for_id(entry)
        results.append(outdict)
        print(f"Worked on {outdict['name']}")

    print('*** Finished datacollection ***')
    print('found {} materials'.format(len(results)))

    df = pd.DataFrame(results)  # pylint:disable=invalid-name
    df.to_csv('mp_parsing_results.csv')


if __name__ == '__main__':
    collect_entries()
