follower_color = "red"
following_color = "green"
node_shape = "circle"


def default_concentric_style(max_edge_intros_log):
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
                "color": "White",
                "border-color": "#8ca37f",
                "border-width": "data(border_width)",
                "background-color": "mapData(total_intros, 0, 500, red,  blue)",
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
    stylesheet = [  # unselected, "greyed out"
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
                "background-color": "#B10DC9",
                "border-color": "#8ca37f",
                "border-width": "data(border_width)",
                "border-opacity": 1,
                "opacity": 1,
                "label": "data(label)",
                "color": "#B10DC9",
                "text-opacity": 1,
                "font-size": 12,
                "text-valign": "center",
                "text-halign": "center",
                "color": "White",
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
                        "background-color": following_color,
                        "opacity": 0.9,
                        "label": "data(label)",
                        "text-valign": "center",
                        "text-halign": "center",
                        "color": "White",
                    },
                }
            )
            stylesheet.append(
                {
                    "selector": 'edge[id= "{}"]'.format(edge["id"]),
                    "style": {
                        "mid-target-arrow-color": following_color,
                        "mid-target-arrow-shape": "triangle",
                        "line-color": following_color,
                        "opacity": 0.9,
                        "z-index": 5000,
                        "text-valign": "center",
                        "text-halign": "center",
                        "color": "White",
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
                        "background-color": follower_color,
                        "opacity": 0.9,
                        "z-index": 9999,
                        "label": "data(label)",
                        "text-valign": "center",
                        "text-halign": "center",
                        "color": "White",
                    },
                }
            )
            stylesheet.append(
                {
                    "selector": 'edge[id= "{}"]'.format(edge["id"]),
                    "style": {
                        "mid-target-arrow-color": follower_color,
                        "mid-target-arrow-shape": "triangle",
                        "line-color": follower_color,
                        "opacity": 1,
                        "z-index": 5000,
                        "width": "mapData(log_intros, 0, "
                        + str(max_edge_intros_log)
                        + ", 2, 24)",
                    },
                }
            )
    return stylesheet