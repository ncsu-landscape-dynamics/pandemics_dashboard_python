import plotly.graph_objects as go


def draw_vector_map(
    title,
    iso_column,
    data_column,
    colorscale,
    reverse_colors,
    labels,
    starting_countries_iso,
    starting_countries_labels,
):
    layout = dict(
        title=title,
        geo=dict(showframe=False, projection={"type": "natural earth"}),
        plot_bgcolor="#19191a",
        paper_bgcolor="#19191a",
        geo_bgcolor="#19191a",
    )
    data = go.Choropleth(
        locations=iso_column,
        locationmode="ISO-3",
        z=data_column,
        colorscale=colorscale,
        colorbar={"title": ""},
        reversescale=reverse_colors,
        text=labels,
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        height=1000, font_color="white"
    )  # sets fig size - could potentially be adaptive)
    fig.update_geos(
        projection_type="robinson",
        showocean=True,
        oceancolor="#838383",
        resolution=50,
        showcountries=True,
        countrycolor="grey",
    )

    starting = go.Choropleth(
        locations=starting_countries_iso,
        locationmode="ISO-3",
        z=[1] * len(starting_countries_iso),
        colorscale=[[0, "#017520"], [1, "#017520"]],
        showscale=False,
        text=starting_countries_labels,
    )

    hide_antarctica = go.Choropleth(
        locations=["ATA"],
        locationmode="ISO-3",
        z=[1],
        colorscale=[[0, "#838383"], [1, "#838383"]],
        showscale=False,
    )

    fig.add_trace(starting)
    fig.add_trace(hide_antarctica)
    return fig
