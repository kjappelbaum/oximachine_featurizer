# -*- coding: utf-8 -*-
import pickle


def load_pickle(filename):
    with open(filename, "rb") as handle:
        result = pickle.load(handle)
    return result
