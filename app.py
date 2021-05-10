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
from dash.dependencies import Input, Output, State
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
from summary_network_drawing import *
import dash_cytoscape as cyto
import matplotlib as mpl
import matplotlib.cm as cm
from cytoscape_stylesheets import *

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
cyto.load_extra_layouts()  # cytoscape force directed layouts
summary_data = pickle.load(open(os.path.join(viz_data_path, "summary_data.p"), "rb"))
app = dash.Dash(__name__, assets_folder="assets", suppress_callback_exceptions=True)

app.title = "Pandemics Dashboard"
server = app.server
reset_click_tracker = 0
fi_ind_tab_content = html.Div(
    [
        html.Div(
            [
                dcc.Markdown("""***Scenario:***"""),
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

temporal_tab_content = html.Div(
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

geographic_tab_content = html.Div(
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
                        # {"label": "Individual Runs", "value": "ind"},
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
                dcc.Markdown("""***Scenario:***"""),
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


ri_network_tab_content = html.Div(
    [
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            [
                dcc.Store(id="elements_storage"),
                dcc.Store(id="graph_storage"),
                dcc.Store(id="degree_cent_storage"),
                dcc.Store(id="graph_is_focused_storage"),
                dcc.Markdown("""***Scenario:***"""),
                dcc.Dropdown(
                    id="ri_network_attr_dropdown",
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
                dcc.Markdown("""***Display Style:***"""),
                dcc.RadioItems(
                    id="ri_network_highlight_style",
                    className="radiobutton-group",
                    options=[
                        {
                            "label": "Focus - Click to Expand Connected Nodes",
                            "value": "focus_expand",
                        },
                        {
                            "label": "Highlight - Show Country in Context ",
                            "value": "highlight",
                        },
                    ],
                    value="focus_expand",
                    labelStyle={
                        "display": "inline-block",
                    },
                ),
                html.Hr(),
                dcc.Markdown("""***Node Color:***"""),
                dcc.RadioItems(
                    id="ri_network_node_color",
                    className="radiobutton-group",
                    options=[
                        {"label": "None ", "value": "no_color"},
                        {"label": "Centrality", "value": "centrality"},
                    ],
                    value="centrality",
                    labelStyle={
                        "display": "inline-block",
                    },
                ),
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
        html.Hr(),
        html.Div(
            [
                dcc.Markdown("""***Filter Edges***"""),
                html.Div(id="edge_slider_value_container", style={"margin-top": 20}),
                dcc.Slider(
                    id="ri_edge_weight_slider",
                    min=0,
                    max=3,
                    step=0.01,
                    updatemode="drag",
                    value=0,
                ),
                dcc.Markdown("""***Filter Nodes ***"""),
                dcc.Markdown("""Proportion of Runs (uncertainty) > :"""),
                dcc.Slider(
                    id="ri_network_prop_slider",
                    min=0,
                    max=1,
                    step=0.05,
                    value=0,
                    marks=dict([(i / 10, str(i / 10)) for i in range(0, 10)]),
                ),
                dcc.Markdown("""Degree Centrality (importance) > :"""),
                dcc.Slider(id="ri_degree_centrality_slider", min=0, step=0.01, value=0),
            ],
            style={"padding": "0px 20px 20px 20px"},
        ),
    ]
)


@app.callback(
    Output("edge_slider_value_container", "children"),
    Input("ri_edge_weight_slider", "value"),
)
def edge_slider_weight(value):
    transformed = 10 ** value
    return "Edges > :  {:0.0f}".format(transformed)


@app.callback(
    [
        Output("ri_degree_centrality_slider", "max"),
        Output("ri_degree_centrality_slider", "marks"),
    ],
    Input("ri_network_attr_dropdown", "value"),
)
def populate_centraility_max(attr_num):

    max_degree = summary_data[attr_list[attr_num]]["network"]["max_degree_cent"]

    degree_marks = dict(
        [(i / 20, str(i / 20)) for i in range(0, int(max_degree * 20.0))]
    )

    return max_degree, degree_marks


"""
@app.callback(


    [],
    []
)
"""


@app.callback(
    [
        Output("elements_storage", "data"),
        Output("graph_storage", "data"),
        Output("degree_cent_storage", "data"),
    ],
    [
        dash.dependencies.Input("ri_network_attr_dropdown", "value"),
        dash.dependencies.Input("ri_network_prop_slider", "value"),
        dash.dependencies.Input("ri_degree_centrality_slider", "value"),
        dash.dependencies.Input("ri_edge_weight_slider", "value"),
    ],
)
def all_intros_network_graph(
    attr_num, prop_slider, deg_centrality_slider, edge_weight_slider
):
    G = summary_data[attr_list[attr_num]]["network"]["diGraph"]
    starting_countries = literal_eval(
        header[header.attributes.str.contains("starting_countries")].values[0, 2]
    )[attr_num]
    max_node_intros = summary_data[attr_list[attr_num]]["network"]["max_node_intros"]

    graph_countries_list = starting_countries  # the list of selected countires to build a subgraph from. All starting countries are included
    prop_dict = summary_data[attr_list[attr_num]]["network"]["prop_dict"]
    degree_centrality = summary_data[attr_list[attr_num]]["network"]["degree_cent"]
    for country in G.nodes():
        if (
            country not in starting_countries
            and prop_dict[country] > prop_slider
            and degree_centrality[country] > deg_centrality_slider
        ):
            graph_countries_list.append(country)
    G = G.subgraph(graph_countries_list)
    graph_edges_list = []
    for edge in G.edges():
        if G[edge[0]][edge[1]]["num_intros"] > edge_weight_slider ** 10:
            graph_edges_list.append(edge)

    G = nx.edge_subgraph(G, graph_edges_list)
    elements = generate_cytoscape_elements(
        G, country_codes_dict, degree_centrality, starting_countries
    )
    node_link = nx.node_link_data(G)

    return elements, node_link, degree_centrality

    '''
    default_stylesheet = [
        {
            "selector": "node",
            "style": {
                # "width": "mapData(size, 0, 100, 20, 60)",
                # "height": "mapData(size, 0, 100, 20, 60)",
                "content": "data(label)",
                "font-size": "12px",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": "mapData(total_intros, 0, max_node_intros, #2375fa,  green)",
            },
        },
    ]

    concentric_layout = {"name": "concentric", "startAngle": 0}
    """
    clockwise: true, 
    minNodeSpacing: 10, 
    avoidOverlap: true, 
    nodeDimensionsIncludeLabels: false, 
     
    'concentric': function (node) { return node.degree();},
    levelWidth: function (nodes) { return nodes.maxDegree() / 4;},
    animate: false, 
    animationDuration: 500, 
    animationEasing: undefined, 
    animateFilter: function (node, i) { return true; }, 
    ready: undefined, 
    stop: undefined, 
    transform: function (node, position) { return position; } 
    }

    """
    circle_layout = {"name": "circle", "radius": 250, "startAngle": 0}
    '''


@app.callback(
    [
        Output("cytoscape", "stylesheet"),
        Output("cytoscape", "elements"),
        Output("cytoscape", "layout"),
        Output("cytoscape", "style"),
        Output("graph_is_focused_storage", "data"),
    ],
    [
        Input("cytoscape", "tapNode"),
        Input("cytoscape", "tapEdge"),
        Input("cytoscape", "selectedNodeData"),
        Input("cytoscape", "selectedEdgeData"),
        dash.dependencies.Input("ri_network_attr_dropdown", "value"),
        Input("ri_network_highlight_style", "value"),
        Input("elements_storage", "data"),
        Input("graph_storage", "data"),
        Input("degree_cent_storage", "data"),
        dash.dependencies.Input("ri_network_node_color", "value"),
    ],
    [
        State("cytoscape", "stylesheet"),
        State("cytoscape", "elements"),
        State("cytoscape", "layout"),
        State("cytoscape", "style"),
        State("cytoscape", "tapNode"),
        State("graph_is_focused_storage", "data"),
    ],
)
def generate_stylesheet(
    node,
    edge,
    node_data,
    edge_data,
    attr_num,
    highlight_style,
    elements_master,
    node_link,
    degree_centrality,
    color_selected,
    stylesheet_state,
    elements_state,
    layout_state,
    style_state,
    node_state,
    graph_is_focused,
):

    max_edge_intros_log = math.log(
        summary_data[attr_list[attr_num]]["network"]["max_edge_intros"]
    )

    if node_data == None and edge_data == None:  # on startup
        default_stylesheet = default_concentric_style(
            max_edge_intros_log, color_selected
        )
        style = {
            "width": "105%",
            "height": "900px",
            "backgroundColor": "#2e2e2c",
            "minZoom": 0.2,
            "maxZoom": 1,
        }
        layout = {"name": "concentric", "startAngle": 0, "fit": True}

        return default_stylesheet, elements_master, layout, style, "unfocused"

    if (
        (node_data == [] or node_data == None) and edge_data != [] and edge_data != None
    ):  # Clicked Edge - graph propagated in other stylesheet

        return stylesheet_state, elements_state, layout_state, style_state, "unfocused"

    style = {
        "width": "105%",
        "height": "800px",
        "backgroundColor": "#2e2e2c",
        "minZoom": 0.2,
        "maxZoom": 1,
    }

    max_edge_intros = summary_data[attr_list[attr_num]]["network"]["max_edge_intros"]
    layout = {"name": "concentric", "startAngle": 0, "fit": True}

    if highlight_style == "highlight":
        default_stylesheet = default_concentric_style(
            max_edge_intros_log, color_selected
        )
        if node_data == []:
            return default_stylesheet, elements_master, layout, style, "unfocused"
        if not node:
            return default_stylesheet, elements_master, layout, styl, "unfocused"

        stylesheet = concentric_highlight_node(
            node,
            math.log(max_edge_intros),
        )

        return stylesheet, elements_master, layout, style, "unfocused"

    elif highlight_style == "focus" or highlight_style == "focus_expand":

        default_stylesheet = default_concentric_style(
            max_edge_intros_log, color_selected
        )

        if node_data == [] or node_data == None:
            return default_stylesheet, elements_master, layout, style, "unfocused"
        if not node:
            return default_stylesheet, elements_master, layout, style, "unfocused"

        if graph_is_focused != "unfocused" and highlight_style == "focus_expand":

            degree_centrality = summary_data[attr_list[attr_num]]["network"][
                "degree_cent"
            ]
            G = nx.node_link_graph(node_link)

            # print("ELEMENTS STATE", elements_state)
            # print(node.keys())
            root_source_countries = list(G.predecessors(graph_is_focused))
            new_elements = concentric_focus_elements(
                G,
                country_codes_dict,
                node["edgesData"],
                expand_on=node_data[0]["id"],
                existing_elements=elements_state,
            )  # NEED G< CCD< DEDREE CENT IN THIS FUNCTION
            focus_elements = new_elements + elements_state

            stylesheet = concentric_highlight_node(
                node,
                math.log(
                    summary_data[attr_list[attr_num]]["network"]["max_edge_intros"]
                ),
            )
            # print("FOCUS_ELEMENTS", focus_elements)
            new_stylesheet = expand_focus_stylesheet(
                new_elements,
                math.log(
                    summary_data[attr_list[attr_num]]["network"]["max_edge_intros"]
                ),
                graph_is_focused,
                node,
                root_source_countries=root_source_countries,
            )

            stylesheet = stylesheet_state + new_stylesheet
            return stylesheet, focus_elements, layout_state, style, graph_is_focused

        layout = {
            "name": "cola",
            "animate": False,
            "animationDuration": 200,
            "fit": False,
            "minZoom": 0.2,
            "maxZoom": 1,
        }
        """
        layout = {
            "name": "concentric",
            "startAngle": 0,
            "fit": False,
            "spacingFactor": 2,
            "animate": True,
            "animationDuration": 300,
            
        }


        """
        # "minNodeSpacing": 80
        G = nx.node_link_graph(node_link)
        focus_elements = concentric_focus_elements(
            G,
            country_codes_dict,
            node["edgesData"],
            expand_on=None,
            existing_elements=[],
        )  # NEED G< CCD< DEDREE CENT IN THIS FUNCTION
        stylesheet = concentric_highlight_node(
            node,
            math.log(summary_data[attr_list[attr_num]]["network"]["max_edge_intros"]),
        )
        focused_node = node_data[0]["id"]
        return stylesheet, focus_elements, layout, style, focused_node

    # else:
    # return default_stylesheet, elements_master, layout, style, "unfocused"


@app.callback(
    [
        Output("network_summary_histogram_1", "children"),
        Output("network_summary_histogram_2", "children"),
    ],
    [
        Input("cytoscape", "tapNode"),
        Input("cytoscape", "tapEdge"),
        Input("cytoscape", "selectedNodeData"),
        Input("cytoscape", "selectedEdgeData"),
        Input("graph_storage", "data"),
    ],
)
def summary_network_edge_histogram(node, edge, node_data, edge_data, node_link):
    G = nx.node_link_graph(node_link)

    if node_data is None and edge_data is None:
        return [], []
    if (node_data == [] or node_data == None) and edge_data != [] and edge_data != None:

        source = edge_data[0]["source"]
        target = edge_data[0]["target"]
        years = G[source][target]["years"]
        trace = go.Histogram(
            x=years,
            xbins=dict(start=min(years), size=1, end=max(years)),
            marker=dict(color="#8fc6cc"),
        )
        layout = go.Layout(title=str("Introductions From " + source + " to " + target))

        fig = go.Figure(data=trace, layout=layout)
        fig.update_layout(
            xaxis=dict(tickmode="linear", dtick=1, color="white"),
            yaxis=dict(tickmode="auto", color="white"),
            height=450,
            paper_bgcolor="#2e2e2c",
            plot_bgcolor="#2e2e2c",
            title_font=dict(color="white"),
        )
        return dcc.Graph(figure=fig), []

    elif node_data != [] or node is None:

        # Imports
        try:
            target = node_data[0]["id"]
        except IndexError:
            return [], []
        intro_data = G.nodes[target]["intro_years"]
        export_data = G.nodes[target]["export_years"]
        intro_trace = go.Histogram(
            x=intro_data,
            xbins=dict(start=min(intro_data), size=1, end=max(intro_data)),
            marker=dict(color="#de5716"),
        )
        intro_layout = go.Layout(title=str("All Introductions to " + target))

        intro_fig = go.Figure(data=intro_trace, layout=intro_layout)
        intro_fig.update_layout(
            xaxis=dict(tickmode="linear", dtick=1, color="white"),
            yaxis=dict(tickmode="auto", color="white"),
            paper_bgcolor="#2e2e2c",
            plot_bgcolor="#2e2e2c",
            title_font=dict(color="white"),
        )
        if export_data == []:
            return dcc.Graph(figure=intro_fig), []
        export_trace = go.Histogram(
            x=export_data,
            xbins=dict(start=min(export_data), size=1, end=max(export_data)),
            marker=dict(color="#8338EC"),
        )
        export_layout = go.Layout(title=str("All Exports from " + target))

        export_fig = go.Figure(data=[export_trace], layout=export_layout)
        export_fig.update_layout(
            xaxis=dict(tickmode="linear", dtick=1, color="white"),
            yaxis=dict(tickmode="auto", color="white"),
            paper_bgcolor="#2e2e2c",
            plot_bgcolor="#2e2e2c",
            title_font=dict(color="white"),
        )

        return dcc.Graph(figure=intro_fig), dcc.Graph(figure=export_fig)

    else:
        return [], []


# callback here to take attr from dropdown and pass it to the degree centrality slider as the max centrality measure
fi_multi_tab_content = html.Div(
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
                    value="geographic_tab",
                    className="custom-tabs",
                    children=[
                        dcc.Tab(
                            id="tab-1",
                            label="Geographic",
                            value="geographic_tab",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            children=[html.Div(id="graph-4")],
                        ),
                        dcc.Tab(
                            id="tab-2",
                            label="Network",
                            value="network_tab",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            children=[
                                dcc.Tabs(
                                    id="network_tabs",
                                    value="fi-multi",
                                    className="custom-tabs",
                                    children=[
                                        dcc.Tab(
                                            label="First Introductions",
                                            id="fi_subtab",
                                            value="first_intros",
                                            className="custom-tab",
                                            selected_className="custom-tab--selected",
                                            children=[
                                                dcc.Tabs(
                                                    id="fi_network_tabs",
                                                    value="fi_individual",
                                                    className="custom-tabs",
                                                    children=[
                                                        dcc.Tab(
                                                            label="First Introductions: Individual Runs",
                                                            id="fi_subsubtab1",
                                                            value="fi_individual",
                                                            children=html.Div(
                                                                id="fi_ind_network_graph"
                                                            ),
                                                            className="custom-tab",
                                                            selected_className="custom-tab--selected",
                                                        ),
                                                        dcc.Tab(
                                                            label="First Introductions: Multiple Runs",
                                                            id="fi_subsubtab2",
                                                            value="fi_multi",
                                                            children=html.Div(
                                                                id="fi_multi_network_graph"
                                                            ),
                                                            className="custom-tab",
                                                            selected_className="custom-tab--selected",
                                                        ),
                                                    ],
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            id="ri_subtab",
                                            label="All Introductions",
                                            value="fi-multi",
                                            children=[
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                cyto.Cytoscape(
                                                                    id="cytoscape",
                                                                    minZoom=0.5,
                                                                    maxZoom=2,
                                                                    elements=[],
                                                                    style={
                                                                        "width": "100%",
                                                                        "height": "920px",
                                                                        "backgroundColor": "#2e2e2c",
                                                                    },
                                                                    layout={
                                                                        "name": "preset"
                                                                    },
                                                                    stylesheet=[],
                                                                )
                                                            ],
                                                            className="nine columns",
                                                        ),
                                                        html.Div(
                                                            id="network_summary_histogram_1",
                                                            className="three columns",
                                                        ),
                                                        html.Div(
                                                            id="network_summary_histogram_2",
                                                            className="three columns",
                                                        ),
                                                    ]
                                                )
                                            ],
                                            className="custom-tab",
                                            selected_className="custom-tab--selected",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Tab(
                            id="tab-3",
                            label="Temporal",
                            value="temporal_tab",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                            children=[html.Div(id="temporal_graph")],
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
    ],
)


@app.callback(
    Output("tabs-content", "children"),
    [
        dash.dependencies.Input("tabs", "value"),
        dash.dependencies.Input("network_tabs", "value"),
        dash.dependencies.Input("fi_network_tabs", "value"),
    ],
)
def render_content(tab, network_tab, fi_content):
    if tab == "network_tab":
        if network_tab == "first_intros":
            if fi_content == "fi_individual":
                return fi_ind_tab_content
            elif fi_content == "fi_multi":
                return fi_multi_tab_content

        else:
            return ri_network_tab_content
    elif tab == "temporal_tab":
        return temporal_tab_content

    elif tab == "geographic_tab":
        return geographic_tab_content


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
    Output("temporal_graph", "children"),
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
        for attr in attributes_selected:
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
        Output("fi_ind_network_graph", "children"),
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
    Output("fi_multi_network_graph", "children"),
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
def show_hide_element_map(selected_data):
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
    starting_countries_iso = []
    starting_countries_labels = []
    for country in starting_countries:
        starting_countries_iso.append(country_codes_dict[country])
        starting_countries_labels.append(
            summary_data[attr_list[attr_num]]["cartographic"]["labels"][
                country_codes_dict[country]
            ]
        )
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
                try:
                    labels.append(
                        summary_data[attr_list[attr_num]]["cartographic"]["labels"][i]
                    )
                except KeyError:
                    labels.append("KeyError")

            fig = draw_vector_map(
                title_text,
                iso_column,
                data_column,
                colorscale,
                reverse_colorscale,
                labels,
                starting_countries_iso,
                starting_countries_labels,
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
                # print("ISO:")
                # print(iso_column)
                print(summary_data[attr_list[attr_num]]["cartographic"]["labels"])
                for i in iso_column:
                    try:
                        labels.append(
                            summary_data[attr_list[attr_num]]["cartographic"]["labels"][
                                i
                            ]
                        )
                    except KeyError:
                        labels.append("KeyError")

                fig = draw_vector_map(
                    title_text,
                    iso_column,
                    data_column,
                    colorscale,
                    reverse_colors,
                    labels,
                    starting_countries_iso,
                    starting_countries_labels,
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
            title_text,
            iso_column,
            data_column,
            colorscale,
            reverse_colors,
            labels,
            starting_countries_iso,
            starting_countries_labels,
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
            starting_countries_iso,
            starting_countries_labels,
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
