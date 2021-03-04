import os
import pandas as pd


def sortTuple(tup):

    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):

        for j in range(0, lst - i - 1):
            if tup[j][1] > tup[j + 1][1]:
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup


def country_codes():
    # takes custom data file made from probability file
    names_data = pd.read_csv("country_names.csv")

    country_codes_dict = {}
    country_codes_dict["Origin"] = "ORG"
    for index, row in names_data.iterrows():

        country_codes_dict[row["NAME"]] = row["ISO3"]
    country_codes_dict["Taiwan"] = "TWN"
    return country_codes_dict


def convertTuple(tup):
    text = " - ".join(map(str, tup))
    return text


def truncate(f, n):
    """Truncates/pads a float f to n decimal places without rounding"""
    s = "%.12f" % f
    i, p, d = s.partition(".")
    return ".".join([i, (d + "0" * n)[:n]])


def get_pandemic_data_files(filepath, attr_set, iteration, attr_list):

    parFolder = str(attr_list[attr_set])
    iterFolder = "run_" + str(iteration)

    odFilepath = os.path.join(filepath, parFolder, iterFolder, "origin_destination.csv")
    input_data = pd.read_csv(odFilepath)
    input_data["Year"] = input_data["TS"].astype(str).str[:4]
    input_data["Year"] = input_data["Year"].astype(int)
    odFilepath2 = os.path.join(
        filepath, parFolder, iterFolder, "pandemic_output_aggregated.csv"
    )
    probability_data = pd.read_csv(
        odFilepath2, index_col=0, header=0
    )  # This is the input for probabilities, aggregated to year.

    return input_data, probability_data
