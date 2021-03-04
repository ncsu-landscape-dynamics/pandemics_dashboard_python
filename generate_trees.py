import networkx as nx
import numpy as np
from hierarchy_pos import hierarchy_pos
import plotly.graph_objects as go
from addEdge import addEdge
import pandas as pd
import os

here = os.path.dirname(os.path.abspath(__file__))


def generate_networks(emergent_countries, input_data, year_selection_slider):
    total_intros_dict = {}
    G = nx.DiGraph()  # Intial graph , holds all connections
    introduction_tally = 0
    for (
        country
    ) in emergent_countries:  # initializes origin and the native range countries in G()
        G.add_edge("Origin", country, year=0, num_introductions=1)
        G.nodes[country]["year_introduced"] = 0
        G.nodes[country]["introduced_from"] = "Origin"
        G.nodes[country]["fullname"] = country
        G.nodes[country]["num_introductions"] = 1
        G.nodes[country]["in_deg"] = 0
        G.nodes[country]["out_deg"] = 1
        total_intros_dict[country] = "Native to this Country"
    G.nodes["Origin"]["year_introduced"] = 0
    G.nodes["Origin"]["introduced_from"] = "none"
    G.nodes["Origin"]["fullname"] = "Origin"
    G.nodes["Origin"]["num_introductions"] = 0
    G.nodes["Origin"]["in_deg"] = 0
    G.nodes["Origin"]["out_deg"] = 1
    H = G.copy()  # secondary graph, stores only the first introductions
    total_intros_dict["Origin"] = "The Home Range of the Species"
    for index, row in input_data.iterrows():
        if int(row["Year"]) <= year_selection_slider:
            introduction_tally = introduction_tally + 1
            org = row["Origin"]
            dest = row["Destination"]

            if G.has_node(org):
                if "out_deg" in G.nodes[org]:
                    G.nodes[org]["out_deg"] = G.nodes[org]["out_deg"] + 1
                else:
                    G.nodes[org]["out_deg"] = 1
            if G.has_node(dest):
                if "in_deg" in G.nodes[dest]:
                    G.nodes[dest]["in_deg"] = G.nodes[dest]["in_deg"] + 1
                else:
                    G.nodes[dest]["in_deg"] = 1
            if not G.has_node(org):
                G.nodes[org]["out_deg"] = 1
            if not G.has_node(dest):
                total_intros_dict[dest] = org + " " + str(row["Year"])
                G.nodes[org]["in_deg"] = 1

            if G.has_node(dest):

                if H.nodes[dest]["num_introductions"] < 11:
                    total_intros_dict[dest] = (
                        total_intros_dict[dest] + "<br>" + org + " " + str(row["Year"])
                    )

            if G.has_edge(org, dest):
                G.edges[org, dest]["num_introductions"] = (
                    G.edges[org, dest]["num_introductions"] + 1
                )
            if H.has_node(dest):
                H.nodes[dest]["num_introductions"] = (
                    H.nodes[dest]["num_introductions"] + 1
                )

            if H.has_node(dest) == False:
                H.add_edge(org, dest, year=row["Year"], num_introductions=1)
                H.nodes[dest]["year_introduced"] = row["Year"]
                H.nodes[dest]["introduced_from"] = row["Origin"]
                H.nodes[dest]["fullname"] = row["Destination"]
                H.nodes[dest]["num_introductions"] = 1

            if not G.has_edge(org, dest):
                G.add_edge(org, dest, year=row["Year"], num_introductions=1)
                if "year_introduced" not in G.nodes[dest]:
                    G.nodes[dest]["year_introduced"] = row["Year"]
                    G.nodes[dest]["introduced_from"] = row["Origin"]

    for node in G.nodes():
        if H.nodes[node]["num_introductions"] >= 11:
            total_intros_dict[node] = (
                total_intros_dict[node]
                + "<br> "
                + str(H.nodes[node]["num_introductions"])
                + " more"
            )
    return G, H, total_intros_dict, introduction_tally


def country_codes():
    # takes custom data file made from probability file
    names_data = pd.read_csv("country_names.csv")

    country_codes_dict = {}
    country_codes_dict["Origin"] = "ORG"
    for index, row in names_data.iterrows():

        country_codes_dict[row["NAME"]] = row["ISO3"]
    country_codes_dict["Taiwan"] = "TWN"
    return country_codes_dict


def draw_network(
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
):
    pos = hierarchy_pos(tree, country_selection, leaf_vs_root_factor=0.6)
    arrowangle = 0
    H = tree
    nx.set_node_attributes(H, pos, "pos")
    selected_edges = list(H.edges())
    master_years = nx.get_edge_attributes(G, "year")
    master_intros = nx.get_edge_attributes(G, "num_introductions")

    edge_label_text = []
    edge_color_list = []

    for edge in selected_edges:  # writes and applies text for each edge
        edge_text = str(master_years[edge])
        total_intros = str(master_intros[edge])
        edge_text = (
            "First introduction: "
            + edge_text
            + " | Total introductions: "
            + total_intros
        )
        for i in range(9):
            edge_label_text.append(edge_text)

    edge_x = []
    edge_y = []
    edge_trace_list = []
    path_edges = []

    # This allows tracing of the path from the root node to the US , if it appears in the graph. Throws an error if it does not appear
    if "United States" in tree and year_selection_slider >= int(
        G.nodes["United States"]["year_introduced"]
    ):
        path = nx.shortest_path(tree, source=country_selection, target="United States")

        for node in range(len(path) - 1):
            path_edges.append((path[node], path[node + 1]))

    # the following for loop selects colors for each edge, including coloring the path to the US
    for edge in selected_edges:
        start = H.nodes[edge[0]]["pos"]
        if edge[0] != "Origin":
            end = H.nodes[edge[1]]["pos"]
            edge_x_pos = []
            edge_y_pos = []
            # edge_x, edge_y = addEdge(start, end, edge_x, edge_y, 1, 'end', .02, 6, 40)
            edge_x_pos, edge_y_pos = addEdge(
                start, end, edge_x_pos, edge_y_pos, 1, "end", 0.02, arrowangle, 30
            )
            edge_x.extend(edge_x_pos)
            edge_y.extend(edge_y_pos)
            edge_text = str(master_years[edge])
            total_intros = str(master_intros[edge])
            edge_text = (
                "First introduction: "
                + edge_text
                + " | Total introductions: "
                + total_intros
            )
            edge_text_list = []
            for i in range(
                9
            ):  # each label needs to be duplicated 9 times for each of the 9 points of the drawn arrows from addEdge()
                edge_text_list.append(edge_text)

            if edge in path_edges and uspath == "on":
                edge_color = "green"
                edge_weight = 8
            elif master_intros[edge] >= 4:
                edge_color = "white"
                edge_weight = 8

            elif master_intros[edge] == 3:
                edge_color = "#AAA8AA"
                edge_weight = 8
            elif master_intros[edge] == 2:
                edge_color = "#7B787C"
                edge_weight = 6.5
            else:
                edge_color = "#4C484E"
                edge_weight = 5

            trace = go.Scatter(  # creates a trace for each edge, appends to list to be drawn later)
                x=edge_x_pos,
                y=edge_y_pos,
                line=dict(width=edge_weight, color=edge_color),
                hoverinfo="text",
                text=(edge_text_list),
                mode="lines",
            )
            edge_trace_list.append(trace)

    node_annotations = []
    # for node in H.nodes():
    total_prob_dict = {}
    total_prob_list = []
    # selects probabilites based on the year from the year_selection_slider
    column_to_select = "Agg Prob Intro "
    column_to_select = column_to_select + str(year_selection_slider)
    presence_select = "Presence "
    presence_select = presence_select + str(year_selection_slider)
    selection = []
    for node in H.nodes():
        if node != "Origin":
            namerow = probability_data.loc[probability_data["NAME"] == node]
            pres = list(namerow[presence_select])
            if (
                pres[0] == True
            ):  # currently selecting probabililities the same if pest is introduced or not. Pest presence at the moment initialized from O/D pairs

                dat = list(namerow[column_to_select])
                total_prob_list.append(dat[0])
                H.nodes[node]["nod_col"] = dat[0]
                selection.append(node)

                prob_trimmed = str(dat[0])
                prob_trimmed = prob_trimmed[0:4]
                total_prob_dict[node] = prob_trimmed
            else:

                dat = list(namerow[column_to_select])
                total_prob_list.append(dat[0])
                H.nodes[node]["nod_col"] = dat[0]
                selection.append(node)

                prob_trimmed = str(dat[0])
                prob_trimmed = prob_trimmed[0:4]
                total_prob_dict[node] = prob_trimmed

    # sets the color for node borders
    def SetColorEdge(x):
        if x == 0:
            return "#1E1820"
        elif x == 9999 and layout_opts == "twopi":  # green border for US
            return "green"
        elif x >= 10:
            return "#AAA8AA"
        elif x >= 100:
            return "#7B787C"
        elif x >= 150:
            return "#4C484E"
        else:
            return "#1E1820"

    node_x = []
    node_y = []
    colorlist = []
    node_num_intros = []

    node_anno_text = []
    node_anno_size = []
    node_anno_col = []

    for node in H.nodes():  # node annotations
        if node != "Origin":
            x, y = H.nodes[node]["pos"]
            node_anno_text.append(
                "<b>"
                + node
                + "</b><br> <br>P(intro):  "
                + str(total_prob_dict[node])
                + " <br>Introductions: <br>"
                + total_intros_dict[node]
            )
            node_anno_size.append(25)
            node_anno_col.append("blue")
            node_x.append(x)
            node_y.append(y)
            node_num_intros.append(master_node_intros[node])

    colorscale = [
        [0, "rgba(77, 67, 58, .9)"],
        [0.25, "rgba(132, 82, 33, .8)"],
        [0.5, "rgba(194, 116, 37, .8)"],
        [0.75, "rgba(213, 77, 40, .8)"],
        [1.0, "rgba(196, 37, 37, .8)"],
    ]

    # node traces
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        hovertext=node_anno_text,
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            reversescale=False,
            color=[],
            size=35,
            colorbar=dict(
                thickness=35,
                title="Probability of Introduction",
                titlefont=dict(color="white", size=14),
                tickfont=dict(color="white"),
                xanchor="left",
                titleside="right",
                bgcolor="#19191a",
            ),
            line_width=3,
            line_color=list(map(SetColorEdge, node_num_intros)),
        ),
    )

    node_annotations = []
    annotations = []
    for (
        node
    ) in (
        H.nodes()
    ):  # sets color for node ISO code annotations, which float over each node
        if node != "Origin":
            x, y = H.nodes[node]["pos"]
            node_col = H.nodes[node]["nod_col"]
            node_text = country_codes_dict[node]
            if node_col > 0.65:
                text_col = "white"
            else:
                text_col = "white"

            annotations.append(
                dict(
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    text=node_text,  # node name that will be displayed
                    xanchor="right",
                    xshift=15,
                    font=dict(color=text_col, size=12),
                    showarrow=False,
                    arrowhead=1,
                    ax=-10,
                    ay=-10,
                ),
            )

    node_text = []

    node_trace.marker.color = total_prob_list
    node_trace.text = selection

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            plot_bgcolor="#19191a",
            paper_bgcolor="#19191a",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0, pad=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    for trace in edge_trace_list:
        fig.add_trace(trace)

    fig.update_layout(
        height=850,  # sets fig size - could potentially be adaptive
        showlegend=False,
        annotations=annotations,  # shows iSO annotations
    )
    return fig, edge_trace_list

