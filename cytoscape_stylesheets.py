destination_color = "#de5716"  # orange
source_color = "#8338EC"  # Purple
selected_color = "#FF006E"
node_shape = "circle"


def default_concentric_style(max_edge_intros_log, color_selected):
    if color_selected == "no_color":
        node_color = "grey"
    elif color_selected == "centrality":
        node_color = "data(centrality_color)"
    else:
        node_color = "grey"

    return [
        {
            "selector": "node",
            "style": {
                # "width": "mapData(size, 0, 100, 20, 60)",
                # "height": "mapData(size, 0, 100, 20, 60)",
                "content": "data(label)",
                "font-size": "12px",
                "text-valign": "center",
                "text-halign": "center",
                "color": "black",
                "border-color": "#017520",
                "border-width": "data(border_width)",
                "background-color": node_color,
            },
        },
        {
            "selector": "edge",
            "style": {
                "opacity": "mapData(log_intros, 0, "
                + str(max_edge_intros_log)
                + ", 0.3, 0.8)",
                "curve-style": "bezier",
                "width": "mapData(log_intros, 0, "
                + str(max_edge_intros_log)
                + ", 1, 8)",
            },
        },
    ]


def concentric_highlight_node(node, max_edge_intros_log):
    stylesheet = [
        {"selector": "node", "style": {"opacity": 0.3, "shape": node_shape}},
        {
            "selector": "edge",
            "style": {
                "opacity": 0.2,
                "curve-style": "bezier",
            },
        },
        {
            "selector": 'node[id = "{}"]'.format(node["data"]["id"]),
            "style": {
                "background-color": selected_color,
                "opacity": 1,
                "label": "data(label)",
                "color": "#B10DC9",
                "text-opacity": 1,
                "font-size": 12,
                "text-valign": "center",
                "text-halign": "center",
                "color": "black",
                "z-index": 9999,
            },
        },
    ]
    for edge in node["edgesData"]:
        if edge["source"] == node["data"]["id"]:
            stylesheet.append(
                {
                    "selector": 'node[id = "{}"]'.format(edge["target"]),
                    "style": {
                        "background-color": source_color,
                        "opacity": 0.9,
                        "label": "data(label)",
                        "text-valign": "center",
                        "text-halign": "center",
                        "color": "black",
                    },
                }
            )
            stylesheet.append(
                {
                    "selector": 'edge[id= "{}"]'.format(edge["id"]),
                    "style": {
                        "mid-target-arrow-color": source_color,
                        "mid-target-arrow-shape": "triangle",
                        "line-color": source_color,
                        "opacity": 0.5,
                        "z-index": 5000,
                        "text-valign": "center",
                        "text-halign": "center",
                        "color": "black",
                        "width": "mapData(log_intros, 0, "
                        + str(max_edge_intros_log)
                        + ", 2, 24)",
                        # "label": "data(num_intros)",
                    },
                }
            )

        if edge["target"] == node["data"]["id"]:
            stylesheet.append(
                {
                    "selector": 'node[id = "{}"]'.format(edge["source"]),
                    "style": {
                        "background-color": destination_color,
                        "opacity": 0.9,
                        "z-index": 9999,
                        "label": "data(label)",
                        "text-valign": "center",
                        "text-halign": "center",
                        "color": "black",
                    },
                }
            )
            stylesheet.append(
                {
                    "selector": 'edge[id= "{}"]'.format(edge["id"]),
                    "style": {
                        "mid-target-arrow-color": destination_color,
                        "mid-target-arrow-shape": "triangle",
                        "line-color": destination_color,
                        "opacity": 1,
                        "z-index": 5000,
                        "width": "mapData(log_intros, 0, "
                        + str(max_edge_intros_log)
                        + ", 2, 24)",
                    },
                }
            )
    return stylesheet


def expand_focus_stylesheet(
    new_elements, max_edge_intros_log, root_node, node, root_source_countries
):
    stylesheet = []
    for element in new_elements:
        if "target" in element["data"].keys():  # just gives edge elements
            source = element["data"]["source"]
            target = element["data"]["target"]

            if (
                target in root_source_countries
                and target != root_node
                and source != root_node
            ):  # ORANGE COUNTRIES
                stylesheet.append(
                    {
                        "selector": 'node[id = "{}"]'.format(target),
                        "style": {
                            "background-color": destination_color,
                            "opacity": 0.9,
                            "label": "data(label)",
                            "text-valign": "center",
                            "text-halign": "center",
                            "color": "black",
                        },
                    }
                )
                stylesheet.append(
                    {
                        "selector": 'node[id = "{}"]'.format(source),
                        "style": {
                            "background-color": destination_color,
                            "opacity": 0.9,
                            "label": "data(label)",
                            "text-valign": "center",
                            "text-halign": "center",
                            "color": "black",
                        },
                    }
                )

                stylesheet.append(
                    {
                        "selector": 'edge[id= "{}"]'.format(element["data"]["id"]),
                        "style": {
                            "mid-target-arrow-color": destination_color,
                            "mid-target-arrow-shape": "triangle",
                            "line-color": destination_color,
                            "opacity": 0.5,
                            "z-index": 5000,
                            "text-valign": "center",
                            "text-halign": "center",
                            "color": "black",
                            "width": "mapData(log_intros, 0, "
                            + str(max_edge_intros_log)
                            + ", 2, 24)",
                            # "label": "data(num_intros)",
                        },
                    }
                )

            elif (
                source != root_node
                and target != root_node
                and target not in root_source_countries
            ):  # PURPLES
                stylesheet.append(
                    {
                        "selector": 'node[id = "{}"]'.format(target),
                        "style": {
                            "background-color": source_color,
                            "opacity": 0.9,
                            "label": "data(label)",
                            "text-valign": "center",
                            "text-halign": "center",
                            "color": "black",
                        },
                    }
                )
                if source not in root_source_countries:
                    stylesheet.append(
                        {
                            "selector": 'node[id = "{}"]'.format(source),
                            "style": {
                                "background-color": source_color,
                                "opacity": 0.9,
                                "label": "data(label)",
                                "text-valign": "center",
                                "text-halign": "center",
                                "color": "black",
                            },
                        }
                    )
                stylesheet.append(
                    {
                        "selector": 'edge[id= "{}"]'.format(element["data"]["id"]),
                        "style": {
                            "mid-target-arrow-color": source_color,
                            "mid-target-arrow-shape": "triangle",
                            "line-color": source_color,
                            "opacity": 0.5,
                            "z-index": 5000,
                            "text-valign": "center",
                            "text-halign": "center",
                            "color": "black",
                            "width": "mapData(log_intros, 0, "
                            + str(max_edge_intros_log)
                            + ", 2, 24)",
                            # "label": "data(num_intros)",
                        },
                    }
                )

    return stylesheet