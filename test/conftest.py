# -*- coding: utf-8 -*-
import numpy as np
import pytest


@pytest.fixture(scope='module')
def get_oxidationstate_dict():
    names = [
        'Tetracarbonyl-(h6-1,2,4,5-tetramethylbenzene)-vanadium(i) hexacarbonyl-vanadium',
        '(m-carbonyl)-([(phenylphosphanediyl)di(2,1-phenylene)]bis[di(propan-2-yl)phosphane])-tris[1,2-bis(methoxy)ethane]-sodium-cobalt(-1)',
    ]
    expected_result = [{'V': [1, np.nan]}, {'Co': [-1]}]

    return names, expected_result
