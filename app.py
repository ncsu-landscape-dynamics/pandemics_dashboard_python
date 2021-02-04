
import sys

import os
import pandas as pd
from colour import Color
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
from statistics import mean, median
from ast import literal_eval
from dash.exceptions import PreventUpdate
from collections import defaultdict
import seaborn as sns 
import json
import plotly.io as pio
pio.templates.default = "none"

#import pydot
#from networkx.drawing.nx_pydot import graphviz_layout

#filepath = input("Enter path to folder containing header.csv and subfolders of run data  :    ")
filepath = r"Q:\Shared drives\APHIS  Projects\Pandemic\Data\slf_model\outputs\time_lags"
header_path = os.path.join(filepath, 'header.csv')
header = pd.read_csv(header_path)

with open('Q:\Shared drives\APHIS  Projects\Pandemic\Data\DASHboard\countries_revised_wHost_filtered_reindex.json') as f:
  geometry = json.load(f)

token = 'pk.eyJ1IjoidGhvbXdvcm0iLCJhIjoiY2s2bDY1ZXJyMDhrbTNqbjB6YWV4ZG91dyJ9.RRKk7tlQbdvJBLWnXwG9QA'


year_list = range(2010,2018)

country_names_file = pd.read_csv('iso3_un.csv', index_col=0) #crosswalk file for 3 -letter iso / country names. No need to change.
attr_list = literal_eval(header[header.attributes.str.contains('run_prefix')].values[0,2])

app = dash.Dash(__name__, assets_folder='assets')
app.config.suppress_callback_exceptions = True
server = app.server
reset_click_tracker = 0
tab1 = html.Div([
       
         
html.Div(
[ dcc.Markdown("""***Attribute Set***"""), 
    dcc.Dropdown(
        id='attr_dropdown',
        
        options = [{'label':attr_list[i], 'value':i} for i in range(len(attr_list))],
        style=
                                    { 
                                      'color': '#212121',
                                      'background-color': '#212121',
                                    } ,
        value = 0
       
        )
])
        ,


html.Div(
[ dcc.Markdown("""***Iteration***"""),
        dcc.Slider(
        id='run_slider',
        min=0,
        step= 1,
        value = 0
        )
        
        
        ]),

        html.Div(
[ dcc.Markdown("""***Year***"""),


        dcc.Slider(
        id='year_slider_individual',
        
        step= 1
        
        )]),
html.Div(
[            html.Button('Reset', id='reset_btn', n_clicks=0),
            dcc.Markdown("""***Layout***"""),
        ]),

html.Div(style={'fontColor':'white'}, children =
            dcc.Markdown("""***Path to United States***"""),  
        ),

dcc.RadioItems(
            id = "uspath",
            className='radiobutton-group',
        options=[
        {'label': 'On ', 'value': 'on'},
        {'label': 'Off', 'value': 'off'},
        
    ],
    value='off',
    labelStyle={'display': 'inline-block',
    }
)  , 

html.Div(style={'fontColor':'white'}, children =
            dcc.Markdown("""***All Introductions***"""),  
        ),



 html.Table([
        html.Tr(["Total Countries", html.Td(id='total_countries_o')]),
        html.Tr(["Total Introductions", html.Td(id='total_intros_o')]),
        html.Tr(["Avg Reintroductions (by pathway) ", html.Td(id='avg_reintros_o')]),
        html.Tr(["Median Reintroductions (by pathway)", html.Td(id='median_reintros_o')]),
        html.Tr(["Avg Country in degrees (Introductions) ", html.Td(id='avg_indegrees_o')]),
        html.Tr(["Avg Country out degrees (Transmission)", html.Td(id='avg_outdegrees_o')]),
        html.Tr(["Top Countries by in degrees (most susceptible) ", html.Td(id='top_indegrees_o')]),
        html.Tr(["Top Countries by out degrees (most infectious)", html.Td(id='top_outdegrees_o')])

     ])
    ])

tab2 = html.Div([ html.Div(
[ 
        
        dcc.RadioItems(
            id = "aggregate_view_select",
            className='radiobutton-group',
        options=[
        {'label': 'Averages', 'value': 'avg'},
        {'label': 'All', 'value': 'all'},
        {'label': 'Runs Only', 'value': 'runs'}
        
    ],
    value='avg',
    labelStyle={'display': 'inline-block',
    }
)]),
html.Div(
[ 
        
        dcc.RadioItems(
            id = "aggregate_data_select",
            className='radiobutton-group',
        options=[
        {'label': 'Introductions/ Timestep', 'value': 'intros'},
        {'label': 'Total Countries', 'value': 'countries'}
        
    ],
    value='intros',
    labelStyle={'display': 'inline-block',
    }
)]),

html.Div(
[ 
 dcc.Checklist(
            id="aggregate_attr_list",
            options=[
                {'label': i, 'value':i} for i in attr_list
            ],
            value=[attr for attr in attr_list],
            labelStyle={"display": "inline-block"},
        )
])
    ])    

tab4 = html.Div([
       
         
html.Div(
[ dcc.Markdown("""***Attribute Set***"""), 
    dcc.Dropdown(
        id='attr_dropdown_map',
        
        options = [{'label':attr_list[i], 'value':i} for i in range(len(attr_list))],
        style=
                                    { 
                                      'color': '#212121',
                                      'background-color': '#212121',
                                    } ,
        value = 0
       
        )
])
        ,


html.Div(
[ dcc.Markdown("""***Iteration***"""),
        dcc.Slider(
        id='run_slider_map',
        min=0,
        step= 1,
        value = 0
        )
        
        
        ]),

        html.Div(
[ dcc.Markdown("""***Year***"""),


        dcc.Slider(
        id='year_slider_map',
        
        step= 1
        
        )])])

def multi_tree_attributes(attr_list):
    if len(attr_list) < 4:
        return [attr for attr in attr_list]
    else:
        return [attr for attr in attr_list[:2]]


multi_year_min  = int(min(literal_eval(header[header.attributes.str.contains('start_year')].values[0,2])))

multi_year_max = int(max(literal_eval(header[header.attributes.str.contains('stop_year')].values[0,2])))

tab3 = html.Div([ html.Div(
[ 
 dcc.Checklist(
            id="multi_tree_attr_list",
            options=[
                {'label': i, 'value':i} for i in attr_list
            ],
            value=multi_tree_attributes(attr_list),
            labelStyle={"display": "inline-block"},
        )

]),
        html.Div(
[ dcc.Markdown("""***Year***"""),


        dcc.Slider(
        id='year_slider_multi',
        min = multi_year_min,
        max = multi_year_max,
        step= 1,
        value = multi_year_min,
        marks = dict((int(i), str(i)) for i in range(multi_year_min, multi_year_max))
        
        )]),
        html.Div([
        dcc.RadioItems(
            id = "uspath_multi",
            className='radiobutton-group',
        options=[
        {'label': 'On ', 'value': 'on'},
        {'label': 'Off', 'value': 'off'},
        
    ],
    value='off',
    labelStyle={'display': 'inline-block',
    }
)])  , 

])
app.layout = html.Div([ # HTML layout of the app and slider info. See dash documentation for more
    html.Div(
style={'backgroundColor': '#19191a', 'fontColor':'white'},
children=[
  
        
    dcc.Tabs(id="tabs", value='tab_1', children=[
        dcc.Tab(id="tab-1", label='Individual Runs/ Network', value='tab_1', children = [ html.Div(id='graph-1')]),
        dcc.Tab(id="tab-2", label='Aggregate', value='tab_2', children = [ html.Div(id='graph-2')]),
        dcc.Tab(id="tab-3", label='Multi-Tree', value='tab_3', children = [ html.Div(id='graph-3')]),
        dcc.Tab(id="tab-4", label='Map', value='tab_4', children = [ html.Div(id='graph-4')])
    ], colors={
        "border": "#2e1e18",
        "primary": "orange",
        "background": "#1f1c1a"
    }),
    html.Div(id='tabs-content',
             children = tab1)

    ])
])

@app.callback(Output('tabs-content', 'children')
    ,
             [dash.dependencies.Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab_1':
        return tab1
    elif tab == 'tab_2':
        return tab2
    elif tab == 'tab_3':
        return tab3
    elif tab == 'tab_4':
        return tab4


@app.callback([Output(component_id= 'run_slider', component_property='max'),
Output(component_id= 'run_slider', component_property= 'marks'),
Output(component_id= 'year_slider_individual', component_property= 'min'),
Output(component_id= 'year_slider_individual', component_property= 'max'),
Output(component_id= 'year_slider_individual', component_property= 'value'),
Output(component_id= 'year_slider_individual', component_property= 'marks')],

[dash.dependencies.Input('attr_dropdown', 'value')])

def select_attr(attr_num):

    #INDIVIDUAL SLIDERS
    num_it = literal_eval(header[header.attributes.str.contains('num_runs')].values[0,2])
    start_y = literal_eval(header[header.attributes.str.contains('start_year')].values[0,2])
    stop_y = literal_eval(header[header.attributes.str.contains('stop_year')].values[0,2])
   
    num_it = num_it[attr_num]
    start_y = int(start_y[attr_num])
    stop_y = int(stop_y[attr_num])
    


    return list((num_it, 
    dict((int(i), 
    str(i)) for i in range(num_it)),
     start_y, 
     stop_y, start_y,
      dict((int(i), str(i)) for i in range(start_y, stop_y)) ))

@app.callback([Output(component_id= 'run_slider_map', component_property='max'),
Output(component_id= 'run_slider_map', component_property= 'marks'),
Output(component_id= 'year_slider_map', component_property= 'min'),
Output(component_id= 'year_slider_map', component_property= 'max'),
Output(component_id= 'year_slider_map', component_property= 'value'),
Output(component_id= 'year_slider_map', component_property= 'marks')],

[dash.dependencies.Input('attr_dropdown_map', 'value')])

def select_attr(attr_num):

    #INDIVIDUAL SLIDERS
    num_it = literal_eval(header[header.attributes.str.contains('num_runs')].values[0,2])
    start_y = literal_eval(header[header.attributes.str.contains('start_year')].values[0,2])
    stop_y = literal_eval(header[header.attributes.str.contains('stop_year')].values[0,2])
   
    num_it = num_it[attr_num]
    start_y = int(start_y[attr_num])
    stop_y = int(stop_y[attr_num])
    


    return list((num_it, 
    dict((int(i), 
    str(i)) for i in range(num_it)),
     start_y, 
     stop_y, start_y,
      dict((int(i), str(i)) for i in range(start_y, stop_y)) ))

    
@app.callback(
    Output('graph-2', 'children'),
    [
        Input('aggregate_view_select', 'value'),
        Input('aggregate_attr_list', 'value'),
        Input('aggregate_data_select', 'value')
    ]
)
def update_graph_aggregate(view, attributes_selected, data_selected):
    years = []
    intros = []
    fig = go.Figure(data=go.Scatter(x=years, y=intros, line_color='#d6861e'))
    colors = sns.color_palette("colorblind", 10).as_hex()
    colors_dict = {}
    for i in range(len(attr_list)):
            colors_dict[attr_list[i]] = colors[i]

    if view == 'all' or view == 'runs': 
        
        for i in range(len(attr_list)):
            if attr_list[i] in attributes_selected:
                run_iterations = literal_eval(header[header.attributes.str.contains('num_runs')].values[0,2])
                run_iterations = run_iterations[i]

                for n in range(run_iterations):
                    parFolder =  str(attr_list[i])
                    iterFolder = "run_" + str(n)
                    odFilepath =  os.path.join(filepath, parFolder, iterFolder, 
                'origin_destination.csv')
                    all_intros_dict = {}
                    od_data = pd.read_csv(odFilepath)
                    countries_list = literal_eval(header[header.attributes.str.contains('starting_countries')].values[0,2])
                    for index, row in od_data.iterrows():

                        year = int(str(row['TS'])[:4])
                            
                        month = int(str(row['TS'])[4:6])
                        date = datetime(year = year, month = month, day = 1)
                        if data_selected == 'intros':
                            if date in all_intros_dict:
                                all_intros_dict[date] = all_intros_dict[date] + 1 
                            else:
                                all_intros_dict[date] = 1
                        else:
                            dest = str(row['Destination'])
                            if dest not in countries_list:
                                countries_list.append(dest)
                            all_intros_dict[date] = len(countries_list)
                        
                
                    if view == 'all':
                        alpha = 0.3
                    else:
                        alpha = 1

                    all_intros_dict = sorted(all_intros_dict.items())
                    years, intros = zip(*all_intros_dict)
                    sl = False
                    if n == 0:
                        sl = True
                    fig.add_trace(go.Scatter(x=years, y=intros, line_color= colors[i] , opacity = alpha , showlegend = sl, name = attr_list[i]))

    if view == 'avg' or view == 'all':
        for i in range(len(attr_list)):
            if attr_list[i] in attributes_selected:
                run_iterations = literal_eval(header[header.attributes.str.contains('num_runs')].values[0,2])
                run_iterations = run_iterations[i]
                unique_dates = []
                date_values = {}

                for n in range(run_iterations):
                    parFolder =  str(attr_list[i])
                    iterFolder = "run_" + str(n)
                    odFilepath =  os.path.join(filepath, parFolder, iterFolder, 
                'origin_destination.csv')
                    run_intros_dict = {}
                    od_data = pd.read_csv(odFilepath)
                    countries_list = literal_eval(header[header.attributes.str.contains('starting_countries')].values[0,2])

                    for index, row in od_data.iterrows():

                        year = int(str(row['TS'])[:4])
                            
                        month = int(str(row['TS'])[4:6])
                        date = datetime(year = year, month = month, day = 1)
                        if data_selected == 'intros':
                            if date in run_intros_dict:
                                run_intros_dict[date] = run_intros_dict[date] + 1 
                            else:
                                run_intros_dict[date] = 1
                        else:
                            dest = str(row['Destination'])
                            if dest not in countries_list:
                                countries_list.append(dest)
                            run_intros_dict[date] = len(countries_list)

                    #all_intros_dict = sorted(all_intros_dict.items())
                    #years, intros = zip(*all_intros_dict)
                    for key in run_intros_dict:
                        if key not in date_values:
                            date_values[key] = [run_intros_dict[key]]
                        else:
                            date_values[key].append(run_intros_dict[key])
                for key in date_values:
                    date_values[key] = mean(date_values[key])

                date_values = sorted(date_values.items())
                years, intros = zip(*date_values)
                if data_selected == 'countries':
                    intros = list(intros)
                    
                    for g in range(len(intros)):
                        if g > 0:
                            if intros[g] < intros[g-1]:
                                intros[g] = intros[g-1]
                                

                fig.add_trace(go.Scatter(x=years, y=intros, line_color= colors[i], line_width = 5, name= attr_list[i]  ))

    fig.update_layout(
            height = 850, #sets fig size - could potentially be adaptive
            showlegend=True,
            plot_bgcolor='#19191a',
            paper_bgcolor = '#19191a',
            yaxis=dict(color="white"),
            xaxis=dict(color="white"))


    return dcc.Graph(figure=fig)






@app.callback( #currently all info fed into the same callback - may change in the future if faster layout speeds needed
   [ Output('graph-1', 'children'),
     Output('total_countries_o', 'children'),
     Output('total_intros_o', 'children'),
     Output('avg_reintros_o', 'children'),
     Output('median_reintros_o', 'children'),
     Output('avg_indegrees_o', 'children'),
     Output('avg_outdegrees_o', 'children'),
     Output('top_indegrees_o', 'children'),
     Output('top_outdegrees_o', 'children')],
    
        [Input('year_slider_individual', 'value'),
        Input('attr_dropdown', 'value'),
        Input('run_slider', 'value'),
        #Input('graphic', 'clickData'),
        Input('uspath','value'),
    
        Input('attr_dropdown', 'value')
        ]
        )
def update_graph_individual( year_selection_slider, attr_selection, run_slider,  uspath, attribute_selected_single):

########## CHANGE THESE FILES WITH NEW DATA #################

            


    if year_selection_slider == None:
        raise PreventUpdate

    #GET DATA 
    od_data, probability_data = get_pandemic_data_files(filepath, attribute_selected_single, run_slider, attr_list)

    emergent_countries = literal_eval(header[header.attributes.str.contains('starting_countries')].values[0,2])
    emergent_countries = emergent_countries[attribute_selected_single]
    country_selection = 'Origin'


    #Gernerate Networks
    country_codes_dict = country_codes(od_data,country_names_file, emergent_countries )
    G, H, total_intros_dict, introduction_tally = generate_networks(emergent_countries, od_data, year_selection_slider)

    master_node_intros = nx.get_node_attributes(H,'num_introductions')
    master_node_intros[country_selection] = 9999 #allows coloring of root node of tree
    tree = nx.bfs_tree(H, country_selection) #constructs tree from H - first introductions graph using breadth first search


    ###### SUMMARY STATS #######
   
    num_introduced_countries = len(G.nodes())
    reintros = nx.get_node_attributes(H, 'num_introductions').values()
    avg_reintros = mean(reintros)
    median_reintros = median(reintros)
    in_degs = nx.get_node_attributes(G, 'in_deg')
    out_degs = nx.get_node_attributes(G, 'out_deg')
    in_degs_tuples =  [(k, v) for k, v in in_degs.items()]
    out_degs_tuples =  [(k, v) for k, v in out_degs.items()]
    sort_in_deg = sortTuple(in_degs_tuples)
    sort_out_deg = sortTuple(out_degs_tuples)
    if len(sort_in_deg) > 2:
        top_in_deg = convertTuple(sort_in_deg[-1]) + " | " +  convertTuple(sort_in_deg[-2]) + " | " + convertTuple(sort_in_deg[-3]) 
    else: 
        top_in_deg = "Tree Too Small"
    if len(sort_out_deg) > 2:
        top_out_deg = convertTuple(sort_out_deg[-1]) + " | " +  convertTuple(sort_out_deg[-2]) + " | " + convertTuple(sort_out_deg[-3]) 
    else: 
        top_out_deg = "Tree Too Small"
    avg_indegree = mean(in_degs.values())
    avg_outdegree = mean(out_degs.values())


    ####### LAYOUT AND DISPLAY ######
    
    '''
    if layout_opts == 'twopi':
        pos = hierarchy_pos(tree, country_selection, width = 7, leaf_vs_root_factor= 0.9)
        pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in pos.items()}
    
        arrowangle = 9 #sets the arrow angle, used in the addEdge call. Edges look nice on radial graphs, but not on trees
'''


    fig, dummy_edge_list = draw_network(tree, G, H , country_selection, year_selection_slider, probability_data, total_intros_dict, master_node_intros, country_codes_dict, uspath)
   
    return dcc.Graph(figure=fig) , num_introduced_countries, introduction_tally, truncate(avg_reintros,2), truncate(median_reintros,2), truncate(avg_indegree,2), truncate(avg_outdegree,2), top_in_deg, top_out_deg

@app.callback(
    Output('graph-3', 'children'),
    [Input('multi_tree_attr_list', 'value'),
    Input('year_slider_multi', 'value'),
    Input('uspath_multi', 'value')]

)
def multi_tree_graph(selected_attr_list, year_selection_slider, uspath):
    
    fig = make_subplots(rows = len(selected_attr_list), cols = 4, print_grid = False, shared_xaxes=True, shared_yaxes=True)

    attr_dict = {}
    for i in range(len(attr_list)):
        attr_dict[attr_list[i]] = i

    for attr_i  in range(len(selected_attr_list)):
        for col in range (4):
            od_data, probability_data = get_pandemic_data_files(filepath, attr_dict[selected_attr_list[attr_i]], col, attr_list)
            emergent_countries = literal_eval(header[header.attributes.str.contains('starting_countries')].values[0,2])
            emergent_countries = emergent_countries[attr_dict[selected_attr_list[attr_i]]]
            country_selection = 'Origin'


            #Gernerate Networks
            country_codes_dict = country_codes(od_data,country_names_file, emergent_countries )
            G, H, total_intros_dict, introduction_tally = generate_networks(emergent_countries, od_data, year_selection_slider)

            master_node_intros = nx.get_node_attributes(H,'num_introductions')
            master_node_intros[country_selection] = 9999 #allows coloring of root node of tree
            tree = nx.bfs_tree(H, country_selection) #constructs tree from H - first introductions graph using breadth first search
            sub_fig, edge_trace_list = draw_network(tree, G, H , country_selection, year_selection_slider, probability_data, total_intros_dict, master_node_intros, country_codes_dict, uspath)
            for trace in edge_trace_list:
                fig.add_trace(trace,  attr_i+1, col+1)

            fig.append_trace(sub_fig['data'][0], attr_i+1, col+1)

            rowtitle = " "
            if col + 1  == 1:
                rowtitle = selected_attr_list[attr_i]
            fig.update_yaxes( row= attr_i + 1, col= col + 1 , showgrid = False, visible = False)
            fig.update_xaxes( row= attr_i + 1, col= col + 1 , showgrid = False, visible = True, title_text = rowtitle, color = 'white')

    #need dict to get position of each attr in header file



    


        fig.update_layout(
        height = 1250, #sets fig size - could potentially be adaptive
        showlegend=False,
        #annotations= annotations, #shows iSO annotations
        plot_bgcolor='#19191a',
                    
                    paper_bgcolor = '#19191a',
                    titlefont_size=16,
                    
                    
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0, pad=0)
                    
    )

    
    
   
    return dcc.Graph(figure=fig)

        
   
@app.callback(
    Output('graph-4', 'children'),
    [
        Input('attr_dropdown_map', 'value'),
        Input('year_slider_map', 'value'),
        Input('run_slider_map', 'value')
    ]
)
def update__map(attr, year_selection_slider, iteration):

    if year_selection_slider == None:
        raise PreventUpdate


    od_data, probability_data = get_pandemic_data_files(filepath, attr, iteration, attr_list )
    prob_select = 'Agg Prob Intro '
    prob_select = prob_select + str(year_selection_slider)
    presence_select = "Presence "
    presence_select = presence_select + str(year_selection_slider)

    probability_column =  probability_data[prob_select]

    print(geometry["features"][0]["properties"])

    '''
    fig = px.choropleth_mapbox(probability_data, geojson=geometry, color=prob_select,
                           locations="ISO3", featureidkey="properties.ISO3",
                           center={"lat": 45.5517, "lon": -73.7073},
                           mapbox_style="carto-positron", zoom=9)
    
'''
    colorscale = [
        [0, 'rgba(77, 67, 58, .9)'],
        [0.25, 'rgba(132, 82, 33, .8)'],
        [0.5, 'rgba(194, 116, 37, .8)'], 
        [0.75, 'rgba(213, 77, 40, .8)'],
        [1.0, 'rgba(196, 37, 37, .8)']
    ]

  
    layout = dict(title='Probability', geo=dict(showframe=False, projection={'type': 'natural earth'}), plot_bgcolor='#19191a',
                    
                    paper_bgcolor = '#19191a', geo_bgcolor="#19191a")

    data = go.Choropleth(locations=probability_data['ISO3'], locationmode='ISO-3', z=probability_data[prob_select], colorscale=colorscale,  colorbar={'title': 'Cases of COVID-19'})

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(height = 1250) #sets fig size - could potentially be adaptive)
    return dcc.Graph(figure=fig)
'''
@app.callback(

        Output('graphic', 'clickData'),
        [Input('reset_btn', 'n_clicks')])
def update(reset):
    return None
'''

if __name__ == '__main__':
    app.run_server(debug=True)

