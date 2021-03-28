import sys

import os
import pandas as pd

# from colour import Color
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import csv
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from network_utility_functions import *
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from addEdge import addEdge

from generate_trees import *
import random
import collections
from datetime import datetime
from statistics import mean, median, mode
from ast import literal_eval
from dash.exceptions import PreventUpdate
from collections import defaultdict
import seaborn as sns
import json
import plotly.io as pio
from cartography_utility_functions import *
import pickle
import preprocess_pandemic_viz_data
import time

# import pydot
# from networkx.drawing.nx_pydot import graphviz_layout
#  #These are a tremendous hassle to install and require a remakrable number\
# Of dependencies - ideally will not be used


# filepath = input("Enter path to folder containing header.csv and subfolders of run data  :    ")
filepath = r"H:\Shared drives\Pandemic Data\slf_model\outputs\slf_scenarios"
pio.templates.default = "none"


header_path = os.path.join(filepath, "header.csv")
header = pd.read_csv(header_path)

# country_names_file = pd.read_csv(
#    "iso3_un.csv", index_col=0
# )  # crosswalk file for 3 -letter iso / country names. No need to change.
attr_list = literal_eval(
    header[header.attributes.str.contains("run_prefix")].values[0, 2]
)
od_initialize, prob_initialize = get_pandemic_data_files(filepath, 0, 0, attr_list)
country_codes_dict = country_codes()

viz_data_path = os.path.join(filepath, "summary_data")

summary_data = pickle.load(open(os.path.join(viz_data_path, "summary_data.p"), "rb"))
app = dash.Dash(__name__, assets_folder="assets")
app.config.suppress_callback_exceptions = True
app.title = "Pandemics Dashboard"
server = app.server
reset_click_tracker = 0
tab1 = html.Div(
    [
        html.Div(
            [
                dcc.Markdown("""***Attribute Set***"""),
                dcc.Dropdown(
                    id="attr_dropdown",
                    persistence=True,
                    options=[
                        {"label": attr_list[i], "value": i}
                        for i in range(len(attr_list))
                    ],
                    style={
                        "color": "#212121",
                        "background-color": "#212121",
                    },
                    value=0,
                ),
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            [
                dcc.Markdown("""***Iteration***"""),
                dcc.Slider(id="run_slider", persistence=True, min=0, step=1, value=0),
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            [
                dcc.Markdown("""***Year***"""),
                dcc.Slider(id="year_slider_individual", persistence=True, step=1),
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            [
                html.Button("Reset", id="reset_btn", n_clicks=0),
                dcc.Markdown("""***Layout***"""),
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            style={"fontColor": "white"},
            children=dcc.Markdown("""***Path to United States***"""),
        ),
        html.Div(
            dcc.RadioItems(
                id="uspath",
                persistence=True,
                className="radiobutton-group",
                options=[
                    {"label": "On ", "value": "on"},
                    {"label": "Off", "value": "off"},
                ],
                value="off",
                labelStyle={
                    "display": "inline-block",
                },
            )
        ),
        html.Div(
            style={"fontColor": "white"},
            children=dcc.Markdown("""***All Introductions***"""),
        ),
        html.Table(
            [
                html.Tr(["Total Countries", html.Td(id="total_countries_o")]),
                html.Tr(["Total Introductions", html.Td(id="total_intros_o")]),
                html.Tr(
                    ["Avg Reintroductions (by pathway) ", html.Td(id="avg_reintros_o")]
                ),
                html.Tr(
                    [
                        "Median Reintroductions (by pathway)",
                        html.Td(id="median_reintros_o"),
                    ]
                ),
                html.Tr(
                    [
                        "Avg Country in degrees (Introductions) ",
                        html.Td(id="avg_indegrees_o"),
                    ]
                ),
                html.Tr(
                    [
                        "Avg Country out degrees (Transmission)",
                        html.Td(id="avg_outdegrees_o"),
                    ]
                ),
                html.Tr(
                    [
                        "Top Countries by in degrees (most susceptible) ",
                        html.Td(id="top_indegrees_o"),
                    ]
                ),
                html.Tr(
                    [
                        "Top Countries by out degrees (most infectious)",
                        html.Td(id="top_outdegrees_o"),
                    ]
                ),
            ]
        ),
    ]
)

tab2 = html.Div(
    [
        html.Div(
            [
                dcc.RadioItems(
                    id="aggregate_view_select",
                    className="radiobutton-group",
                    options=[
                        {"label": "Averages", "value": "avg"},
                        {"label": "All (slow 30s per attr)", "value": "all"},
                        {"label": "Runs Only (slow 30s per attr)", "value": "runs"},
                    ],
                    value="avg",
                    labelStyle={
                        "display": "inline-block",
                    },
                )
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            [
                dcc.RadioItems(
                    id="aggregate_data_select",
                    className="radiobutton-group",
                    options=[
                        {
                            "label": "Introductions/ Timestep",
                            "value": "num_introductions",
                        },
                        {"label": "Total Countries", "value": "num_countries"},
                    ],
                    value="num_introductions",
                    labelStyle={
                        "display": "inline-block",
                    },
                )
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            [
                dcc.Checklist(
                    id="aggregate_attr_list",
                    options=[{"label": i, "value": i} for i in attr_list],
                    value=[attr for attr in attr_list],
                    labelStyle={"display": "inline-block"},
                )
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
    ]
)

tab4 = html.Div(
    [
        html.Div(
            [
                dcc.Markdown("""***Data Selection***"""),
                dcc.RadioItems(
                    id="data_type_map",
                    className="radiobutton-group",
                    options=[
                        {"label": "Year of First Introduction", "value": "first"},
                        {"label": "Reintroductions", "value": "reintro"},
                        {"label": "Individual Runs", "value": "ind"},
                    ],
                    value="first",
                    labelStyle={
                        "display": "inline-block",
                    },
                ),
                dcc.RadioItems(
                    id="view_options_map",
                    className="radiobutton-group",
                    labelStyle={"display": "inline-block"},
                ),
                dcc.Markdown("""***Attribute Set***"""),
                dcc.Dropdown(
                    id="attr_dropdown_map",
                    options=[
                        {"label": attr_list[i], "value": i}
                        for i in range(len(attr_list))
                    ],
                    style={
                        "color": "#212121",
                        "background-color": "#212121",
                    },
                    value=0,
                ),
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            id="map_it_slider_container",
            children=[
                dcc.Markdown("""***Iteration***"""),
                dcc.Slider(id="run_slider_map", min=0, step=1, value=0),
            ],
            style={"display": "block"},
        ),
        html.Div(
            id="map_year_slider_container",
            children=[
                dcc.Markdown("""***Year***"""),
                dcc.Slider(id="year_slider_map", step=1),
            ],
            style={"display": "block"},
        ),
    ],
    style={"padding": "0px 20px 20px 20px"},
)


def multi_tree_attributes(attr_list):
    if len(attr_list) < 4:
        return [attr for attr in attr_list]
    else:
        return [attr for attr in attr_list[:2]]


multi_year_min = int(
    min(literal_eval(header[header.attributes.str.contains("start_year")].values[0, 2]))
)

multi_year_max = int(
    max(literal_eval(header[header.attributes.str.contains("stop_year")].values[0, 2]))
)

tab3 = html.Div(
    [
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            [
                dcc.Checklist(
                    id="multi_tree_attr_list",
                    options=[{"label": i, "value": i} for i in attr_list],
                    value=multi_tree_attributes(attr_list),
                    labelStyle={"display": "inline-block"},
                )
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            [
                dcc.Markdown("""***Year***"""),
                dcc.Slider(
                    id="year_slider_multi",
                    min=multi_year_min,
                    max=multi_year_max,
                    step=1,
                    value=multi_year_max,
                    marks=dict(
                        (int(i), str(i)) for i in range(multi_year_min, multi_year_max)
                    ),
                ),
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Div(
            [
                dcc.RadioItems(
                    id="uspath_multi",
                    className="radiobutton-group",
                    options=[
                        {"label": "On ", "value": "on"},
                        {"label": "Off", "value": "off"},
                    ],
                    value="on",
                    labelStyle={
                        "display": "inline-block",
                    },
                )
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
    ]
)

app.layout = html.Div(
    [  # HTML layout of the app and slider info. See dash documentation for more
        html.Div(
            style={"backgroundColor": "#19191a", "fontColor": "white"},
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="tab_4",
                    className="custom-tabs",
                    children=[
                        dcc.Tab(
                            id="tab-4",
                            label="Geographic",
                            value="tab_4",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            children=[html.Div(id="graph-4")],
                        ),
                        dcc.Tab(
                            id="tab-1",
                            label="Network",
                            value="tab_1",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            children=[
                                dcc.Tabs(
                                    id="subtabs1",
                                    value="individual",
                                    className="custom-tabs",
                                    children=[
                                        dcc.Tab(
                                            label="First Introductions: Individual Runs",
                                            id="subtab1",
                                            value="individual",
                                            children=html.Div(id="graph-1"),
                                            className="custom-tab",
                                            selected_className="custom-tab--selected",
                                        ),
                                        dcc.Tab(
                                            label="First Introductions: Multi",
                                            id="subtab2",
                                            value="multi",
                                            children=html.Div(id="graph-3"),
                                            className="custom-tab",
                                            selected_className="custom-tab--selected",
                                        ),
                                    ],
                                )
                            ],
                        ),
                        dcc.Tab(
                            id="tab-2",
                            label="Temporal",
                            value="tab_2",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            children=[html.Div(id="graph-2")],
                        ),
                    ],
                    colors={
                        "border": "#2e1e18",
                        "primary": "orange",
                        "background": "#1f1c1a",
                    },
                ),
                html.Div(id="tabs-content"),
            ],
        )
    ]
)


@app.callback(
    Output("tabs-content", "children"),
    [
        dash.dependencies.Input("tabs", "value"),
        dash.dependencies.Input("subtabs1", "value"),
    ],
)
def render_content(tab, subtab):
    if tab == "tab_1":
        if subtab == "multi":
            return tab3

        return tab1
    elif tab == "tab_2":
        return tab2
    elif tab == "tab_3":
        return tab3
    elif tab == "tab_4":
        return tab4


@app.callback(
    [
        Output(component_id="run_slider", component_property="max"),
        Output(component_id="run_slider", component_property="marks"),
        Output(component_id="year_slider_individual", component_property="min"),
        Output(component_id="year_slider_individual", component_property="max"),
        Output(component_id="year_slider_individual", component_property="value"),
        Output(component_id="year_slider_individual", component_property="marks"),
    ],
    [dash.dependencies.Input("attr_dropdown", "value")],
)
def select_attr(attr_num):

    # INDIVIDUAL SLIDERS
    num_it = literal_eval(
        header[header.attributes.str.contains("num_runs")].values[0, 2]
    )
    start_y = literal_eval(
        header[header.attributes.str.contains("start_year")].values[0, 2]
    )
    stop_y = literal_eval(
        header[header.attributes.str.contains("stop_year")].values[0, 2]
    )

    num_it = num_it[attr_num]
    start_y = int(start_y[attr_num])
    stop_y = int(stop_y[attr_num])
    if num_it > 100:
        run_slider_max = 100
    else:
        run_slider_max = num_it
    return list(
        (
            run_slider_max,
            dict((int(i), str(i)) for i in range(num_it)),
            start_y,
            stop_y,
            stop_y,
            dict((int(i), str(i)) for i in range(start_y, stop_y)),
        )
    )


@app.callback(
    [
        Output(component_id="run_slider_map", component_property="max"),
        Output(component_id="run_slider_map", component_property="marks"),
        Output(component_id="year_slider_map", component_property="min"),
        Output(component_id="year_slider_map", component_property="max"),
        Output(component_id="year_slider_map", component_property="value"),
        Output(component_id="year_slider_map", component_property="marks"),
    ],
    [dash.dependencies.Input("attr_dropdown_map", "value")],
)
def select_attr_map(attr_num):

    # INDIVIDUAL SLIDERS
    num_it = literal_eval(
        header[header.attributes.str.contains("num_runs")].values[0, 2]
    )
    start_y = literal_eval(
        header[header.attributes.str.contains("start_year")].values[0, 2]
    )
    stop_y = literal_eval(
        header[header.attributes.str.contains("stop_year")].values[0, 2]
    )

    num_it = num_it[attr_num]
    start_y = int(start_y[attr_num])
    stop_y = int(stop_y[attr_num])

    return list(
        (
            num_it,
            dict((int(i), str(i)) for i in range(num_it)),
            start_y,
            stop_y,
            stop_y,
            dict((int(i), str(i)) for i in range(start_y, stop_y)),
        )
    )


@app.callback(
    Output("graph-2", "children"),
    [
        Input("aggregate_view_select", "value"),
        Input("aggregate_attr_list", "value"),
        Input("aggregate_data_select", "value"),
    ],
)
def update_graph_aggregate(view, attributes_selected, data_selected):
    years = []
    intros = []
    fig = go.Figure(data=go.Scatter(x=years, y=intros, line_color="#d6861e"))
    colors = sns.color_palette("colorblind", 10).as_hex()
    colors_dict = {}
    for i in range(len(attr_list)):
        colors_dict[attr_list[i]] = colors[i]

    if view == "all" or view == "runs":

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
        print(attributes_selected)
        for attr in attributes_selected:
            # print(i)
            print(attr)
            dates = summary_data[attr]["aggregate"][data_selected]["dates"]
            data = summary_data[attr]["aggregate"][data_selected]["data"]
            if data_selected == "num_introductions":
                title = "Average Number of Introductions per Time Step : " + attr
            elif data_selected == "num_countries":
                title = "Average Number of Introductions per Time Step : " + attr
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=data,
                    line_color=colors_dict[attr],
                    line_width=5,
                    name=attr,
                )
            )

    fig.update_layout(
        height=850,  # sets fig size - could potentially be adaptive
        showlegend=True,
        plot_bgcolor="#19191a",
        paper_bgcolor="#19191a",
        yaxis=dict(color="white"),
        xaxis=dict(color="white"),
        title=title,
        title_font_color="white",
    )

    return dcc.Graph(figure=fig)


@app.callback(  # currently all info fed into the same callback - may change in the future if faster layout speeds needed
    [
        Output("graph-1", "children"),
        Output("total_countries_o", "children"),
        Output("total_intros_o", "children"),
        Output("avg_reintros_o", "children"),
        Output("median_reintros_o", "children"),
        Output("avg_indegrees_o", "children"),
        Output("avg_outdegrees_o", "children"),
        Output("top_indegrees_o", "children"),
        Output("top_outdegrees_o", "children"),
    ],
    [
        Input("year_slider_individual", "value"),
        Input("attr_dropdown", "value"),
        Input("run_slider", "value"),
        # Input('graphic', 'clickData'),
        Input("uspath", "value"),
        Input("attr_dropdown", "value"),
    ],
)
def update_graph_individual(
    year_selection_slider, attr_selection, run_slider, uspath, attribute_selected_single
):

    ########## CHANGE THESE FILES WITH NEW DATA #################

    if year_selection_slider == None:
        raise PreventUpdate

    # GET DATA
    od_data, probability_data = get_pandemic_data_files(
        filepath, attribute_selected_single, run_slider, attr_list
    )

    starting_countries = literal_eval(
        header[header.attributes.str.contains("starting_countries")].values[0, 2]
    )
    starting_countries = starting_countries[attribute_selected_single]
    country_selection = "Origin"

    # Gernerate Networks
    G, H, total_intros_dict, introduction_tally = generate_networks(
        starting_countries, od_data, year_selection_slider
    )

    master_node_intros = nx.get_node_attributes(H, "num_introductions")
    master_node_intros[country_selection] = 9999  # allows coloring of root node of tree
    tree = nx.bfs_tree(
        H, country_selection
    )  # constructs tree from H - first introductions graph using breadth first search

    ###### SUMMARY STATS #######

    num_introduced_countries = len(G.nodes())
    reintros = nx.get_node_attributes(H, "num_introductions").values()
    avg_reintros = mean(reintros)
    median_reintros = median(reintros)
    in_degs = nx.get_node_attributes(G, "in_deg")
    out_degs = nx.get_node_attributes(G, "out_deg")
    in_degs_tuples = [(k, v) for k, v in in_degs.items()]
    out_degs_tuples = [(k, v) for k, v in out_degs.items()]
    sort_in_deg = sortTuple(in_degs_tuples)
    sort_out_deg = sortTuple(out_degs_tuples)
    if len(sort_in_deg) > 2:
        top_in_deg = (
            convertTuple(sort_in_deg[-1])
            + " | "
            + convertTuple(sort_in_deg[-2])
            + " | "
            + convertTuple(sort_in_deg[-3])
        )
    else:
        top_in_deg = "Tree Too Small"
    if len(sort_out_deg) > 2:
        top_out_deg = (
            convertTuple(sort_out_deg[-1])
            + " | "
            + convertTuple(sort_out_deg[-2])
            + " | "
            + convertTuple(sort_out_deg[-3])
        )
    else:
        top_out_deg = "Tree Too Small"
    avg_indegree = mean(in_degs.values())
    avg_outdegree = mean(out_degs.values())

    ####### LAYOUT AND DISPLAY ######

    """
    if layout_opts == 'twopi':
        pos = hierarchy_pos(tree, country_selection, width = 7, leaf_vs_root_factor= 0.9)
        pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in pos.items()}
    
        arrowangle = 9 #sets the arrow angle, used in the addEdge call. Edges look nice on radial graphs, but not on trees
"""

    fig, dummy_edge_list = draw_network(
        tree,
        G,
        H,
        country_selection,
        year_selection_slider,
        probability_data,
        total_intros_dict,
        master_node_intros,
        country_codes_dict,
        uspath,
    )

    return (
        dcc.Graph(figure=fig),
        num_introduced_countries,
        introduction_tally,
        truncate(avg_reintros, 2),
        truncate(median_reintros, 2),
        truncate(avg_indegree, 2),
        truncate(avg_outdegree, 2),
        top_in_deg,
        top_out_deg,
    )


@app.callback(
    Output("graph-3", "children"),
    [
        Input("multi_tree_attr_list", "value"),
        Input("year_slider_multi", "value"),
        Input("uspath_multi", "value"),
    ],
)
def multi_tree_graph(selected_attr_list, year_selection_slider, uspath):

    attr_dict = {}
    for i in range(len(attr_list)):
        attr_dict[attr_list[i]] = i

    for i in selected_attr_list:
        ncol = 4
        if (
            literal_eval(
                header[header.attributes.str.contains("num_runs")].values[0, 2]
            )[attr_dict[i]]
            < 4
        ):
            ncol = literal_eval(
                header[header.attributes.str.contains("num_runs")].values[0, 2]
            )[attr_dict[i]]
    fig = make_subplots(
        rows=len(selected_attr_list),
        cols=ncol,
        print_grid=False,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for attr_i in range(len(selected_attr_list)):

        for col in range(ncol):

            od_data, probability_data = get_pandemic_data_files(
                filepath, attr_dict[selected_attr_list[attr_i]], col, attr_list
            )
            starting_countries = literal_eval(
                header[header.attributes.str.contains("starting_countries")].values[
                    0, 2
                ]
            )
            starting_countries = starting_countries[
                attr_dict[selected_attr_list[attr_i]]
            ]
            country_selection = "Origin"

            # Gernerate Networks
            G, H, total_intros_dict, introduction_tally = generate_networks(
                starting_countries, od_data, year_selection_slider
            )

            master_node_intros = nx.get_node_attributes(H, "num_introductions")
            master_node_intros[
                country_selection
            ] = 9999  # allows coloring of root node of tree
            tree = nx.bfs_tree(
                H, country_selection
            )  # constructs tree from H - first introductions graph using breadth first search
            sub_fig, edge_trace_list = draw_network(
                tree,
                G,
                H,
                country_selection,
                year_selection_slider,
                probability_data,
                total_intros_dict,
                master_node_intros,
                country_codes_dict,
                uspath,
            )
            for trace in edge_trace_list:
                fig.add_trace(trace, attr_i + 1, col + 1)
                fig.append_trace(sub_fig["data"][0], attr_i + 1, col + 1)

            dat = go.Scatter()
            sub_fig = go.Figure(data=dat)
            fig.append_trace(sub_fig["data"][0], attr_i + 1, col + 1)

            rowtitle = " "
            if col + 1 == 1:
                rowtitle = selected_attr_list[attr_i]
            fig.update_yaxes(row=attr_i + 1, col=col + 1, showgrid=False, visible=False)
            fig.update_xaxes(
                row=attr_i + 1,
                col=col + 1,
                showgrid=False,
                visible=True,
                title_text=rowtitle,
                color="white",
            )

        # need dict to get position of each attr in header file

        fig.update_layout(
            height=900,  # sets fig size - could potentially be adaptive
            showlegend=False,
            # annotations= annotations, #shows iSO annotations
            plot_bgcolor="#19191a",
            paper_bgcolor="#19191a",
            titlefont_size=16,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0, pad=0),
        )

    return dcc.Graph(figure=fig)


@app.callback(
    [
        Output(component_id="map_it_slider_container", component_property="style"),
        Output(component_id="view_options_map", component_property="options"),
        Output(component_id="view_options_map", component_property="value"),
    ],
    [Input(component_id="data_type_map", component_property="value")],
)
def show_hide_element(selected_data):
    if selected_data == "ind":
        return [
            {"display": "block"},
            [{"label": "Probability of Introduction", "value": "intro"}],
            "intro",
        ]
    if selected_data == "first":
        return [
            {"display": "none"},
            [
                {"label": "Mode", "value": "fi_mode"},
                {"label": "St. Dev.", "value": "fi_std"},
                {"label": "Minimum", "value": "fi_min"},
                {"label": "Mean", "value": "fi_mean"},
                {"label": "Prop. Runs w/ an Introduction", "value": "fi_prop"},
                {"label": "Prop. Runs w/ an Introduction > 50%", "value": "fi_prop50"},
            ],
            "fi_prop50",
        ]

    if selected_data == "reintro":
        return [
            {"display": "none"},
            [
                {"label": "Mean Num. Reintros", "value": "ri_mean"},
                {"label": "Range of Num. Reintros", "value": "ri_range"},
            ],
            "ri_mean",
        ]


@app.callback(
    Output(component_id="map_year_slider_container", component_property="style"),
    [
        Input(component_id="view_options_map", component_property="value"),
        Input(component_id="data_type_map", component_property="value"),
    ],
)
def show_hide_year_slider_map(view_opts, data_type):
    if view_opts == "prop" or data_type == "ind":
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output("graph-4", "children"),
    [
        Input("attr_dropdown_map", "value"),
        Input("year_slider_map", "value"),
        Input("run_slider_map", "value"),
        Input("view_options_map", "value"),
        Input("data_type_map", "value"),
    ],
)
def update_map(attr_num, year_selection_slider, iteration, view_options, data_type):
    if year_selection_slider == None:
        raise PreventUpdate
    num_it = literal_eval(
        header[header.attributes.str.contains("num_runs")].values[0, 2]
    )[attr_num]
    starting_countries = literal_eval(
        header[header.attributes.str.contains("starting_countries")].values[0, 2]
    )[attr_num]
    if data_type == "first":
        if (
            view_options == "fi_prop" or view_options == "fi_prop50"
        ):  # for proportion of runs w/introduction
            iso_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                view_options
            ]["ISO"]
            data_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                view_options
            ]["data"]
            if view_options == "fi_prop":
                title_text = "Proportion of Runs With Introduction" + str(
                    attr_list[attr_num]
                )
                colorscale = [
                    "#19191a",
                    "#542206",
                    "#742f10",
                    "#913c0d",
                    "#bd4300",
                    "#ff2400",
                ]
            if view_options == "fi_prop50":
                title_text = "Proportion of Runs With Introduction > 50%" + str(
                    attr_list[attr_num]
                )
                colorscale = ["#542206", "#742f10", "#913c0d", "#bd4300", "#ff2400"]

            reverse_colorscale = False
            labels = []
            for i in iso_column:
                labels.append(
                    summary_data[attr_list[attr_num]]["cartographic"]["labels"][i]
                )

            fig = draw_vector_map(
                title_text,
                iso_column,
                data_column,
                colorscale,
                reverse_colorscale,
                labels,
            )

            return dcc.Graph(figure=fig)
            # DONE - need 2 graph it
        if (
            view_options == "fi_mode"
            or view_options == "fi_std"
            or view_options == "fi_min"
            or view_options == "fi_mean"
        ):

            # create dict of all countries

            colorscale = "redor_r"
            iso_column = []
            data_column = []

            if view_options == "fi_mode":
                title_text = "Mode of First Introduction Year : " + str(
                    attr_list[attr_num]
                )
                colorscale = [
                    "#200019",
                    "#350e1d",
                    "#5e1524",
                    "#891718",
                    "#b81300",
                    "#ff0000",
                ]
                reverse_colors = True
                iso_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                    view_options
                ]["ISO"]
                data_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                    view_options
                ]["data"]
            if view_options == "fi_min":
                title_text = "Minimum of First Introduction Year : " + str(
                    attr_list[attr_num]
                )
                colorscale = [
                    "#200000",
                    "#350e00",
                    "#5e1500",
                    "#891700",
                    "#b81d00",
                    "#ff3100",
                ]
                reverse_colors = True
                iso_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                    view_options
                ]["ISO"]
                data_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                    view_options
                ]["data"]

            if view_options == "fi_std":
                title_text = "Standard Deviation of First Introduction Year : " + str(
                    attr_list[attr_num]
                )
                colorscale = "ylorrd_r"
                reverse_colors = True
                iso_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                    view_options
                ]["ISO"]
                data_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                    view_options
                ]["data"]

            if view_options == "fi_mean":
                title_text = "Mean of First Introduction Year : " + str(
                    attr_list[attr_num]
                )
                colorscale = [
                    "#200019",
                    "#350e1d",
                    "#5e1524",
                    "#89172c",
                    "#b81d37",
                    "#ff3144",
                ]
                reverse_colors = True
                iso_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                    view_options
                ]["ISO"]
                data_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
                    view_options
                ]["data"]
            labels = []
            for i in iso_column:
                labels.append(
                    summary_data[attr_list[attr_num]]["cartographic"]["labels"][i]
                )

            fig = draw_vector_map(
                title_text,
                iso_column,
                data_column,
                colorscale,
                reverse_colors,
                labels,
            )
            return dcc.Graph(figure=fig)

    if data_type == "reintro":

        # create dict of all countries

        iso_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
            view_options
        ]["ISO"]
        data_column = summary_data[attr_list[attr_num]]["cartographic"]["data"][
            view_options
        ]["data"]

        colorscale = "blugrn"
        title_text = "placeholder"
        hover_text = []
        if view_options == "ri_mean":
            colorscale = "darkmint_r"
            reverse_colors = False

        if view_options == "ri_range":
            title_text = "Range in Number of Reintroductions:  " + str(
                attr_list[attr_num]
            )
            colorscale = [
                "#002d24",
                "#104a5e",
                "#1b7795",
                "#299cb5",
                "#23c2c3",
                "#00ffcc",
            ]
            reverse_colors = False

        labels = []
        for i in iso_column:
            labels.append(
                summary_data[attr_list[attr_num]]["cartographic"]["labels"][i]
            )
        fig = draw_vector_map(
            title_text, iso_column, data_column, colorscale, reverse_colors, labels
        )
        return dcc.Graph(figure=fig)
    if data_type == "ind":
        od_data, probability_data = get_pandemic_data_files(
            filepath, attr_num, iteration, attr_list
        )
        prob_select = "Agg Prob Intro "
        prob_select = prob_select + str(year_selection_slider)
        presence_select = "Presence "
        presence_select = presence_select + str(year_selection_slider)
        probability_column = list(probability_data[prob_select])
        iso_column = list(probability_data["ISO3"])
        ######### Countries not covered in data:
        # 110m Natural Earth political product, not the UN

        # Somaliland - lumped with somalia

        # probability_column.append(probability_data.loc[probability_data['ISO3'] == 'SOM'][prob_select].item())
        # iso_column.append('SOM')

        probability_column.append(
            probability_data.loc[probability_data["ISO3"] == "USA"][prob_select].item()
        )
        iso_column.append("PRI")

        title_text = "Probability of Introduction: " + str(attr_list[attr_num])
        colorscale = ["#19191a", "#492071", "#7f1ba1", "#a90ba3", "#ce038c", "#f40d70"]
        reverse_colorscale = False

        for i in iso_column:
            print(i)
            labels.append(
                summary_data[attr_list[attr_num]]["cartographic"]["labels"][i]
            )

        fig = draw_vector_map(
            title_text, iso_column, data_column, colorscale, reverse_colorscale, labels
        )
        return dcc.Graph(figure=fig)


"""
@app.callback(

        Output('graphic', 'clickData'),
        [Input('reset_btn', 'n_clicks')])
def update(reset):
    return None
"""

if __name__ == "__main__":
    app.run_server(debug=True)
