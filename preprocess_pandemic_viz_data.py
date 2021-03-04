import pandas as pd
import pickle
import os
from ast import literal_eval
import numpy as np
from statistics import mean, median, mode
from network_utility_functions import country_codes


def generate_viz_data(filepath):

    header_path = os.path.join(filepath, "header.csv")
    header = pd.read_csv(header_path)
    country_codes_dict = country_codes()

    attr_list = literal_eval(
        header[header.attributes.str.contains("run_prefix")].values[0, 2]
    )
    carto_data_dict = carto_data(filepath, header, attr_list, country_codes_dict)
    # carto_data_dict = {"cat": "dog"}
    filename = os.path.join(filepath, "viz_data", "carto_data.p")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(carto_data_dict, f)


"""
def carto_labels(filepath, header, attr_list, country_codes_dict)
"""


def carto_data(filepath, header, attr_list, country_codes_dict):
    map_data_type = [
        "fi_mean",
        "fi_mode",
        "fi_std",
        "fi_min",
        "ri_mean",
        "ri_range",
        "fi_prop",
        "fi_prop50",
    ]

    carto_data_dict = {}
    attr_num = 0
    for attr in attr_list:
        print(attr)
        print(
            "Generating Cartographic Data for:  "
            + attr
            + " "
            + str(attr_num + 1)
            + " of "
            + str(len(attr_list))
        )

        carto_data_dict[attr] = {}
        num_it = literal_eval(
            header[header.attributes.str.contains("num_runs")].values[0, 2]
        )[attr_num]
        for data_type in map_data_type:
            # number of iterations for a given attribtue

            carto_data_dict[attr][data_type] = {"ISO": [], "data": []}

        carto_data_dict = carto_generate_data(
            data_type,
            header,
            filepath,
            attr_list,
            attr_num,
            country_codes_dict,
            num_it,
            carto_data_dict,
        )
        attr_num = attr_num + 1

    return carto_data_dict


def carto_generate_data(
    data_type,
    header,
    filepath,
    attr_list,
    attr_num,
    country_codes_dict,
    num_it,
    carto_data_dict,
):
    proportion_runs_dict = {}
    first_intros_dict = {}
    re_intros_dict = {}
    temp_od_data, temp_probability_data = get_pandemic_data_files(
        filepath, attr_num, 0, attr_list
    )

    for index, row in temp_probability_data.iterrows():
        proportion_runs_dict[row["NAME"]] = 0
        first_intros_dict[row["NAME"]] = []
        re_intros_dict[row["NAME"]] = []

    for i in range(num_it):
        print(str(i) + " of " + str(num_it))
        od_data, probability_data = get_pandemic_data_files(
            filepath, attr_num, i, attr_list
        )
        for country in proportion_runs_dict.keys():
            # od_data = od_data.loc[od_data["Year"] <= year_selection_slider]
            if country in list(od_data["Destination"]):
                # proportion
                proportion_runs_dict[country] = proportion_runs_dict[country] + 1

                # first intros
                first_intro = min(
                    od_data.loc[od_data["Destination"] == country]["Year"]
                )
                first_intros_dict[country].append(first_intro)

                # re-intros
                num_reintros = (
                    len(od_data.loc[od_data["Destination"] == country]["Year"]) - 1
                )

                re_intros_dict[country].append(num_reintros)

    for country in proportion_runs_dict:
        try:
            # PROPORTION
            # Proportion All
            carto_data_dict[attr_list[attr_num]]["fi_prop"]["ISO"].append(
                country_codes_dict[country]
            )
            carto_data_dict[attr_list[attr_num]]["fi_prop"]["data"].append(
                proportion_runs_dict[country] / num_it
            )
            # Proportion > 50
            if proportion_runs_dict[country] / num_it >= 0.5:
                carto_data_dict[attr_list[attr_num]]["fi_prop50"]["ISO"].append(
                    country_codes_dict[country]
                )
                carto_data_dict[attr_list[attr_num]]["fi_prop50"]["data"].append(
                    proportion_runs_dict[country] / num_it
                )

            carto_data_dict[attr_list[attr_num]]["fi_prop"]["ISO"].append("PRI")
            carto_data_dict[attr_list[attr_num]]["fi_prop"]["data"].append(
                proportion_runs_dict["United States"] / num_it
            )
            carto_data_dict[attr_list[attr_num]]["fi_prop50"]["ISO"].append("PRI")
            carto_data_dict[attr_list[attr_num]]["fi_prop50"]["data"].append(
                proportion_runs_dict["United States"] / num_it
            )

            if first_intros_dict[country] != []:
                # FIRST INTROS
                # Mean
                carto_data_dict[attr_list[attr_num]]["fi_mean"]["ISO"].append(
                    country_codes_dict[country]
                )
                carto_data_dict[attr_list[attr_num]]["fi_mean"]["data"].append(
                    mean(first_intros_dict[country])
                )
                # Mode

                carto_data_dict[attr_list[attr_num]]["fi_mode"]["ISO"].append(
                    country_codes_dict[country]
                )
                carto_data_dict[attr_list[attr_num]]["fi_mode"]["data"].append(
                    mode(first_intros_dict[country])
                )

                # Standard Deviation
                carto_data_dict[attr_list[attr_num]]["fi_std"]["ISO"].append(
                    country_codes_dict[country]
                )
                carto_data_dict[attr_list[attr_num]]["fi_std"]["data"].append(
                    np.std(first_intros_dict[country])
                )
                # Minimum
                carto_data_dict[attr_list[attr_num]]["fi_min"]["ISO"].append(
                    country_codes_dict[country]
                )
                carto_data_dict[attr_list[attr_num]]["fi_min"]["data"].append(
                    min(first_intros_dict[country])
                )

                if first_intros_dict["United States"] != []:
                    # Mean
                    carto_data_dict[attr_list[attr_num]]["fi_mean"]["ISO"].append("PRI")
                    carto_data_dict[attr_list[attr_num]]["fi_mean"]["data"].append(
                        mean(first_intros_dict["United States"])
                    )
                    # Mode
                    carto_data_dict[attr_list[attr_num]]["fi_mode"]["ISO"].append("PRI")
                    carto_data_dict[attr_list[attr_num]]["fi_mode"]["data"].append(
                        mode(first_intros_dict["United States"])
                    )

                    # SD
                    carto_data_dict[attr_list[attr_num]]["fi_std"]["ISO"].append("PRI")
                    carto_data_dict[attr_list[attr_num]]["fi_std"]["data"].append(
                        np.std(first_intros_dict["United States"])
                    )
                    # Minimum
                    carto_data_dict[attr_list[attr_num]]["fi_min"]["ISO"].append("PRI")
                    carto_data_dict[attr_list[attr_num]]["fi_min"]["data"].append(
                        min(first_intros_dict["United States"])
                    )

                # RE-INTROS
                # Mean Num Reintros
                carto_data_dict[attr_list[attr_num]]["ri_mean"]["ISO"].append(
                    country_codes_dict[country]
                )
                carto_data_dict[attr_list[attr_num]]["ri_mean"]["data"].append(
                    mean(re_intros_dict[country])
                )
                # Range Num Reintros
                carto_data_dict[attr_list[attr_num]]["ri_mean"]["ISO"].append(
                    country_codes_dict[country]
                )
                carto_data_dict[attr_list[attr_num]]["fi_range"]["data"].append(
                    max(data_dict[country]) - min(data_dict[country])
                )
                if re_intros_dict["United States"] != []:
                    # Mean
                    carto_data_dict[attr_list[attr_num]]["ri_mean"]["ISO"].append("PRI")
                    carto_data_dict[attr_list[attr_num]]["ri_mean"]["data"].append(
                        mean(first_intros_dict["United States"])
                    )
                    # Mode
                    carto_data_dict[attr_list[attr_num]]["ri_mode"]["ISO"].append("PRI")
                    carto_data_dict[attr_list[attr_num]]["ri_mode"]["data"].append(
                        mode(first_intros_dict["United States"])
                    )
        except KeyError:
            continue

    return carto_data_dict


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


"""
def aggregate_graph_data(filepath):



    header_path = os.path.join(filepath, "header.csv")
    header = pd.read_csv(header_path)
    attr_list = literal_eval(
        header[header.attributes.str.contains("run_prefix")].values[0, 2]
    )


    years = []
    intros = []
        for i in range(len(attr_list)):
            if attr_list[i] in attributes_selected:
                run_iterations = literal_eval(
                    header[header.attributes.str.contains("num_runs")].values[0, 2]
                )
                run_iterations = run_iterations[i]

                for n in range(run_iterations):
                    parFolder = str(attr_list[i])
                    iterFolder = "run_" + str(n)
                    odFilepath = os.path.join(
                        filepath, parFolder, iterFolder, "origin_destination.csv"
                    )
                    all_intros_dict = {}
                    od_data = pd.read_csv(odFilepath)
                    countries_list = literal_eval(
                        header[
                            header.attributes.str.contains("starting_countries")
                        ].values[0, 2]
                    )
                    for index, row in od_data.iterrows():

                        year = int(str(row["TS"])[:4])

                        month = int(str(row["TS"])[4:6])
                        date = datetime(year=year, month=month, day=1)
                        if data_selected == "intros":
                            if date in all_intros_dict:
                                all_intros_dict[date] = all_intros_dict[date] + 1
                            else:
                                all_intros_dict[date] = 1
                        else:
                            dest = str(row["Destination"])
                            if dest not in countries_list:
                                countries_list.append(dest)
                            all_intros_dict[date] = len(countries_list)

                    if view == "all":
                        alpha = 0.3
                    else:
                        alpha = 1

                    all_intros_dict = sorted(all_intros_dict.items())
                    years, intros = zip(*all_intros_dict)
                    sl = False
                    if n == 0:
                        sl = True
                    fig.add_trace(
                        go.Scatter(
                            x=years,
                            y=intros,
                            line_color=colors[i],
                            opacity=alpha,
                            showlegend=sl,
                            name=attr_list[i],
                        )
                    )

    if view == "avg" or view == "all":
        for i in range(len(attr_list)):
            if attr_list[i] in attributes_selected:
                run_iterations = literal_eval(
                    header[header.attributes.str.contains("num_runs")].values[0, 2]
                )
                run_iterations = run_iterations[i]
                unique_dates = []
                date_values = {}

                for n in range(run_iterations):
                    parFolder = str(attr_list[i])
                    iterFolder = "run_" + str(n)
                    odFilepath = os.path.join(
                        filepath, parFolder, iterFolder, "origin_destination.csv"
                    )
                    run_intros_dict = {}
                    od_data = pd.read_csv(odFilepath)
                    countries_list = literal_eval(
                        header[
                            header.attributes.str.contains("starting_countries")
                        ].values[0, 2]
                    )

                    for index, row in od_data.iterrows():

                        year = int(str(row["TS"])[:4])
                        month = int(str(row["TS"])[4:6])
                        date = datetime(year=year, month=month, day=1)
                        if data_selected == "intros":
                            if date in run_intros_dict:
                                run_intros_dict[date] = run_intros_dict[date] + 1
                            else:
                                run_intros_dict[date] = 1
                        else:
                            dest = str(row["Destination"])
                            if dest not in countries_list:
                                countries_list.append(dest)
                            run_intros_dict[date] = len(countries_list)

                    # all_intros_dict = sorted(all_intros_dict.items())
                    # years, intros = zip(*all_intros_dict)
                    for key in run_intros_dict:
                        if key not in date_values:
                            date_values[key] = [run_intros_dict[key]]
                        else:
                            date_values[key].append(run_intros_dict[key])
                for key in date_values:
                    date_values[key] = mean(date_values[key])

                date_values = sorted(date_values.items())
                years, intros = zip(*date_values)
                if data_selected == "countries":
                    intros = list(intros)

                    for g in range(len(intros)):
                        if g > 0:
                            if intros[g] < intros[g - 1]:
                                intros[g] = intros[g - 1]
"""

