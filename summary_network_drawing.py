import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import chart_studio.plotly as py
import plotly
import math
from addEdge import addEdge
import random


def draw_force_directed(G, pos, country_codes_dict):
    x_pos = []
    y_pos = []
    iso_labels = []
    hover_labels = []
    for country in pos:
        x_pos.append(pos[country][0])
        y_pos.append(pos[country][1])
        iso_labels.append(country_codes_dict[country])
        G.nodes[country]["pos"] = pos[country]
        hover_labels.append(country)
    arrowangle = 3

    edge_trace_list = []
    edge_x = []
    edge_y = []
    for edge in G.edges():

        start = G.nodes[edge[0]]["pos"]

        end = G.nodes[edge[1]]["pos"]
        edge_x_pos = []
        edge_y_pos = []
        # edge_x, edge_y = addEdge(start, end, edge_x, edge_y, 1, 'end', .02, 6, 40)
        edge_x_pos, edge_y_pos = addEdge(
            start, end, edge_x_pos, edge_y_pos, 1, None, 0.07, arrowangle, 50
        )
        # edge_x.extend(edge_x_pos)
        # edge_y.extend(edge_y_pos)

        edge_text = edge[0] + " to " + edge[1]

        edge_text_list = []
        for i in range(
            9
        ):  # each label needs to be duplicated 9 times for each of the 9 points of the drawn arrows from addEdge()
            edge_text_list.append(edge_text)

        trace = go.Scatter(  # creates a trace for each edge, appends to list to be drawn later)
            x=edge_x_pos,
            y=edge_y_pos,
            line=dict(width=2, color="white"),
            hoverinfo="text",
            text=(edge_text_list),
            mode="lines",
            opacity=0.5,
        )
        edge_trace_list.append(trace)

    node_trace = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="markers",
        hoverinfo="text",
        hovertext=hover_labels,
        marker=dict(
            showscale=False,
            colorscale="aggrnyl",
            reversescale=False,
            color="blue",
            size=50,
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
            line_color="grey",
        ),
    )

    fig = go.Figure(
        data=[],
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
    fig.add_trace(node_trace)

    annotations = []
    for (
        node
    ) in (
        G.nodes()
    ):  # sets color for node ISO code annotations, which float over each node
        if node != "Origin":
            x, y = G.nodes[node]["pos"]
            node_text = country_codes_dict[node]

            annotations.append(
                dict(
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    text=node_text,  # node name that will be displayed
                    xanchor="right",
                    xshift=15,
                    font=dict(color="white", size=12),
                    showarrow=False,
                    arrowhead=1,
                    ax=-10,
                    ay=-10,
                ),
            )
    fig.update_layout(
        height=950,  # sets fig size - could potentially be adaptive
        showlegend=False,
        annotations=annotations,  # shows iSO annotations
    )
    return fig


def generate_cytoscape_elements(G, country_codes_dict, degree_cent, starting_countries):

    nodes = set()

    cy_edges = []
    cy_nodes = []
    for network_edge in G.edges():
        source = network_edge[0]
        target = network_edge[1]
        if source not in nodes:
            nodes.add(source)
            border_width = 0
            print(starting_countries)
            if source in starting_countries:
                print(source)
                border_width = 4

            cy_nodes.append(
                {
                    "data": {
                        "id": source,
                        "label": country_codes_dict[source],
                        "total_intros": G.nodes[source]["total_intros"],
                        "shell": 1,
                        "border_width": border_width,
                        "centrality_color": G.nodes[source]["centrality_color"],
                    }
                }
            )
        if target not in nodes:
            nodes.add(target)
            cy_nodes.append(
                {
                    "data": {
                        "id": target,
                        "label": country_codes_dict[target],
                        "total_intros": G.nodes[target]["total_intros"],
                        "shell": 1,
                        "border_width": 0,
                        "centrality_color": G.nodes[target]["centrality_color"],
                    }
                }
            )

        cy_edges.append(
            {
                "data": {
                    "id": source + "to" + target,
                    "source": source,
                    "target": target,
                    "num_intros": G[source][target]["num_intros"],
                    "log_intros": G[source][target]["log_intros"],
                }
            }
        )
    elements = cy_nodes + cy_edges
    return elements


def concentric_focus_elements(
    G, country_codes_dict, edge_data, expand_on, existing_elements
):
    nodes = set()
    cy_edges = []
    cy_nodes = []
    if existing_elements != []:
        edges = list(G.in_edges(expand_on)) + list(G.out_edges(expand_on))
        for edge in edges:
            source = edge[0]
            target = edge[1]
            if source not in nodes:
                nodes.add(source)
                cy_nodes.append(
                    {
                        "data": {
                            "id": source,
                            "label": country_codes_dict[source],
                            "total_intros": G.nodes[source]["total_intros"],
                        }
                    }
                )
            if target not in nodes:
                nodes.add(target)
                cy_nodes.append(
                    {
                        "data": {
                            "id": target,
                            "label": country_codes_dict[target],
                            "total_intros": G.nodes[target]["total_intros"],
                        }
                    }
                )

            cy_edges.append(
                {
                    "data": {
                        "id": source + "to" + target,
                        "source": source,
                        "target": target,
                        "num_intros": G[source][target]["num_intros"],
                        "log_intros": G[source][target]["log_intros"],
                    }
                }
            )

            # print(cy_edges)
        elements = cy_edges + cy_nodes
        return elements

    for edge in edge_data:
        source = edge["source"]
        target = edge["target"]
        if source not in nodes:
            nodes.add(source)
            cy_nodes.append(
                {
                    "data": {
                        "id": source,
                        "label": country_codes_dict[source],
                        "total_intros": G.nodes[source]["total_intros"],
                    }
                }
            )
        if target not in nodes:
            nodes.add(target)
            cy_nodes.append(
                {
                    "data": {
                        "id": target,
                        "label": country_codes_dict[target],
                        "total_intros": G.nodes[target]["total_intros"],
                    }
                }
            )

        cy_edges.append(
            {
                "data": {
                    "id": source + "to" + target,
                    "source": source,
                    "target": target,
                    "num_intros": G[source][target]["num_intros"],
                    "log_intros": G[source][target]["log_intros"],
                }
            }
        )

    elements = cy_nodes + cy_edges

    return elements