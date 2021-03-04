import plotly.graph_objects as go


def draw_vector_map(title, iso_column, data_column, colorscale, reverse_colors):
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
    return fig

