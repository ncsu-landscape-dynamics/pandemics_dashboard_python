
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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from addEdge import addEdge
from hierarchy_pos import hierarchy_pos
import random
import collections
from datetime import datetime
from statistics import mean, median
from ast import literal_eval
from dash.exceptions import PreventUpdate
from collections import defaultdict
import seaborn as sns 



#import pydot
#from networkx.drawing.nx_pydot import graphviz_layout

filepath = input("Enter path to folder containing header.csv and subfolders of run data  :    ")
#filepath = r"Q:\Shared drives\APHIS  Projects\Pandemic\Data\slf_model\outputs\time_lags"
header_path = os.path.join(filepath, 'header.csv')
header = pd.read_csv(header_path)



year_list = range(2010,2018)

country_names = pd.read_csv('iso3_un.csv', index_col=0) #crosswalk file for 3 -letter iso / country names. No need to change.
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
        id='year_slider',
        
        step= 1
        
        )]),
html.Div(
[            html.Button('Reset', id='reset_btn', n_clicks=0),
            dcc.Markdown("""***Layout***"""),
        ]),
dcc.RadioItems(
            id = "layout_toggle",
        options=[
            {'label': 'Tree ', 'value': 'tree'},
        {'label': 'Radial', 'value': 'twopi'},  
         {'label': 'All Intros', 'value': 'all'}, 
    ],
    value='tree',
    labelStyle={'display': 'inline-block'}
),

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
    


app.layout = html.Div([ # HTML layout of the app and slider info. See dash documentation for more
    html.Div(
style={'backgroundColor': '#19191a', 'fontColor':'white'},
children=[
  
        
    dcc.Tabs(id="tabs", value='tab_1', children=[
        dcc.Tab(id="tab-1", label='Individual Runs/ Network', value='tab_1', children = [ html.Div(id='graph-1')]),
        dcc.Tab(id="tab-2", label='Aggregate', value='tab_2', children = [ html.Div(id='graph-2')]),
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
    print(tab)
    if tab == 'tab_1':
        return tab1
    elif tab == 'tab_2':
        return tab2


@app.callback([Output(component_id= 'run_slider', component_property='max'),
Output(component_id= 'run_slider', component_property= 'marks'),
Output(component_id= 'year_slider', component_property= 'min'),
Output(component_id= 'year_slider', component_property= 'max'),
Output(component_id= 'year_slider', component_property= 'value'),
Output(component_id= 'year_slider', component_property= 'marks')],
#dash.dependencies.Output('start_year', 'children'),
#dash.dependencies.Output('stop_year', 'children'),
#dash.dependencies.Output('starting_countries', 'children'),

[dash.dependencies.Input('attr_dropdown', 'value')])

def select_attr(attr_num):
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
    print(view)
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
                    input_data = pd.read_csv(odFilepath)
                    countries_list = literal_eval(header[header.attributes.str.contains('starting_countries')].values[0,2])
                    for index, row in input_data.iterrows():

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
                    input_data = pd.read_csv(odFilepath)
                    countries_list = literal_eval(header[header.attributes.str.contains('starting_countries')].values[0,2])

                    for index, row in input_data.iterrows():

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
    
        [Input('year_slider', 'value'),
        Input('attr_dropdown', 'value'),
        Input('run_slider', 'value'),
        #Input('graphic', 'clickData'),
        Input('layout_toggle', 'value'),
        Input('uspath','value'),
    
        Input('attr_dropdown', 'value')
        ]
        )
def update_graph_individual( year_selection_slider, attr_selection, run_slider,   layout_opts, uspath, attribute_selected_single):

########## CHANGE THESE FILES WITH NEW DATA #################

            


    if year_selection_slider == None:
        raise PreventUpdate


    parFolder = str(attr_list[attribute_selected_single])
    iterFolder = "run_" + str(run_slider)
   
    odFilepath =  os.path.join(filepath, parFolder, iterFolder, 
        'origin_destination.csv')
    input_data = pd.read_csv(odFilepath)
    input_data['Year'] = input_data['TS'].astype(str).str[:4]

    odFilepath2 =  os.path.join(filepath, parFolder, iterFolder, 
        'pandemic_output_aggregated.csv')
    probability_data = pd.read_csv(odFilepath2, index_col=0, header = 0) # This is the input for probabilities, aggregated to year. 


    emergent_countries = literal_eval(header[header.attributes.str.contains('starting_countries')].values[0,2])
    emergent_countries = emergent_countries[attribute_selected_single]
    source_node = 'Origin' #Origin if multiple emergent countries

    #Creates a list of unique names O/D data

    uniqueorg = input_data['Origin'].unique()
    uniquedest = input_data['Destination'].unique()
    year_list = input_data.Year.unique()


    frames = [uniqueorg, uniquedest]
    country_list = np.concatenate(frames)
    country_list = list(np.unique(country_list))

    for i in emergent_countries:
        if i not in country_list:
            country_list.append(i)
    country_codes_dict = {}
    country_codes_dict['Origin'] = 'ORG'
    #un_codes_dict = {}
    countries_we_dont_have = []
    for country in country_list:
        namerow = country_names.loc[country_names['NAME'] == country] 
        isorow = list(namerow['ISO3'])
        isoname = 'NA'
        if len(isorow) == 1:
                isoname = isorow[0]
        
        #unrow = list(namerow['UN'])

        country_codes_dict[country] = isoname
        #un_codes_dict[country] = unrow[0]

    country_selection = source_node  #sets the default country selection. With multiple origin countries, should be set to "Origin"
    year_selection_click = 0
            
    total_intros_dict = {}
    G=nx.DiGraph() #Intial graph , holds all connections 
    introduction_tally = 0
    for country in emergent_countries: #initializes origin and the native range countries in G()
        G.add_edge("Origin", country, year = 0,num_introductions = 1)
        G.nodes[country]['year_introduced'] = 0
        G.nodes[country]['introduced_from'] = 'Origin'
        G.nodes[country]['fullname'] = country
        G.nodes[country]['num_introductions'] = 1
        G.nodes[country]['in_deg'] = 0
        G.nodes[country]['out_deg'] = 1 
        total_intros_dict[country] = 'Native to this Country'
    G.nodes['Origin']['year_introduced'] = 0
    G.nodes['Origin']['introduced_from'] = "none"
    G.nodes['Origin']['fullname'] = "Origin"
    G.nodes['Origin']['num_introductions'] = 0
    G.nodes['Origin']['in_deg'] = 0
    G.nodes['Origin']['out_deg'] = 1
    H = G.copy() # secondary graph, stores only the first introductions
    total_intros_dict['Origin'] = 'The Home Range of the Species'
    for index, row in input_data.iterrows():
            if int(row["Year"]) <= year_selection_slider:
                introduction_tally = introduction_tally + 1
                org = row["Origin"]
                dest = row["Destination"]
                
                if G.has_node(org):
                    if 'out_deg' in G.nodes[org]:
                        G.nodes[org]['out_deg'] = G.nodes[org]['out_deg'] + 1 
                    else:
                        G.nodes[org]['out_deg'] = 1 
                if G.has_node(dest):
                    if 'in_deg' in G.nodes[dest]:
                        G.nodes[dest]['in_deg'] =G.nodes[dest]['in_deg']  + 1
                    else:
                        G.nodes[dest]['in_deg'] = 1 
                if not G.has_node(org):
                    G.nodes[org]['out_deg'] = 1
                if not G.has_node(dest):
                    total_intros_dict[dest] = org + " " + str(row["Year"])
                    G.nodes[org]['in_deg'] = 1


                
                if G.has_node(dest):
                    
                    if H.nodes[dest]['num_introductions'] < 11:
                        total_intros_dict[dest] = total_intros_dict[dest] + "<br>" + org + " " + str(row["Year"])
        
                if G.has_edge(org,dest):
                    G.edges[org,dest]['num_introductions'] = G.edges[org,dest]['num_introductions'] + 1 
                if  H.has_node(dest):
                    H.nodes[dest]['num_introductions'] = H.nodes[dest]['num_introductions'] + 1


                if  H.has_node(dest) == False:
                        H.add_edge(org, dest, year = row["Year"], num_introductions = 1)
                        H.nodes[dest]['year_introduced'] = row['Year']
                        H.nodes[dest]['introduced_from'] = row['Origin']
                        H.nodes[dest]['fullname'] = row['Destination']
                        H.nodes[dest]['num_introductions'] = 1

                if not G.has_edge(org, dest):
                    G.add_edge(org, dest, year = row["Year"], num_introductions = 1)
                    if 'year_introduced' not in G.nodes[dest]:
                        G.nodes[dest]['year_introduced'] = row['Year']
                        G.nodes[dest]['introduced_from'] = row['Origin']

    for node in G.nodes():
        if H.nodes[node]['num_introductions'] >= 11:
            total_intros_dict[node] = total_intros_dict[node] + "<br> " + str(H.nodes[node]['num_introductions']) + " more"
    '''            
    if ClickData != None:
        country_selection = ClickData['points'][0]['text']
        year_selection_click = G.nodes[country_selection]['year_introduced']
    '''
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

    def Sort_Tuple(tup):  
      
    # getting length of list of tuples 
        lst = len(tup)  
        for i in range(0, lst):  
            
            for j in range(0, lst-i-1):  
                if (tup[j][1] > tup[j + 1][1]):  
                    temp = tup[j]  
                    tup[j]= tup[j + 1]  
                    tup[j + 1]= temp  
        return tup  

    sort_in_deg = Sort_Tuple(in_degs_tuples)
    sort_out_deg = Sort_Tuple(out_degs_tuples)




    def convertTuple(tup): 
        text =  ' - '.join(map(str, tup))
        return text
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

    def truncate(f, n):
        '''Truncates/pads a float f to n decimal places without rounding'''
        s = '%.12f' % f
        i, p, d = s.partition('.')
        return '.'.join([i, (d+'0'*n)[:n]])

    ####### LAYOUT AND DISPLAY ######
    
    
    if layout_opts == 'twopi':
        pos = hierarchy_pos(tree, country_selection, width = 7, leaf_vs_root_factor= 0.9)
        pos = {u:(r*math.cos(theta),r*math.sin(theta)) for u, (theta, r) in pos.items()}
    
        arrowangle = 9 #sets the arrow angle, used in the addEdge call. Edges look nice on radial graphs, but not on trees



    elif layout_opts == 'tree':
        pos = hierarchy_pos(tree, country_selection, leaf_vs_root_factor= 0.6)
        arrowangle = 0
    elif layout_opts == 'all':
        #for visualizing graph of number of total (not first) introductions through time. Foregoes much of the layout and visualization of nodes
        all_intros_dict = {}
        for index, row in input_data.iterrows():
            year = int(str(row['TS'])[:4])
            if year <= year_selection_slider:
                month = int(str(row['TS'])[4:6])
                date = datetime(year = year, month = month, day = 1)
                if date in all_intros_dict:
                    all_intros_dict[date] = all_intros_dict[date] + 1 
                else:
                    all_intros_dict[date] = 1
        years = []
        intros = []
        if all_intros_dict != {}:

            all_intros_dict = sorted(all_intros_dict.items())
            years, intros = zip(*all_intros_dict)

        fig = go.Figure(data=go.Scatter(x=years, y=intros, line_color='#d6861e'))



        fig.update_layout(
            height = 850, #sets fig size - could potentially be adaptive
            showlegend=False,
            plot_bgcolor='#19191a',
            paper_bgcolor = '#19191a',
            yaxis=dict(color="white"),
            xaxis=dict(color="white")
      )
            
           
    
        fig.update_yaxes(range=[0, 45])

            
        
        
        return fig , num_introduced_countries, introduction_tally, truncate(avg_reintros,2), truncate(median_reintros,2), truncate(avg_indegree,2), truncate(avg_outdegree,2), top_in_deg, top_out_deg


    H = tree
    nx.set_node_attributes(H, pos, "pos") 

    selected_edges = list(H.edges())
    
    
    master_years = nx.get_edge_attributes(G,'year')
    master_intros = nx.get_edge_attributes(G,'num_introductions')

    edge_label_text = []
    edge_color_list = []


    for edge in selected_edges: #writes and applies text for each edge
        edge_text = str(master_years[edge])
        total_intros = str(master_intros[edge])
        edge_text = "First introduction: " + edge_text +  " | Total introductions: " + total_intros
        for i in range(9):
            edge_label_text.append(edge_text)
            

    edge_x = []
    edge_y = []

    #if ViewOpts == 'all':
    #  selected_edges = complete_edgelist

# else:
    # selected_edges = tree_edgelist
    

    edge_trace_list = []
    path_edges = []

    #This allows tracing of the path from the root node to the US , if it appears in the graph. Throws an error if it does not appear
    if "United States" in tree and year_selection_slider  >= int(G.nodes['United States']['year_introduced']) :
        path = nx.shortest_path(tree, source = country_selection, target = "United States")
        
        for node in range(len(path)-1):
            path_edges.append((path[node], path[node + 1]))

    #the following for loop selects colors for each edge, including coloring the path to the US
    for edge in selected_edges:
        start = H.nodes[edge[0]]['pos']
        if edge[0] != 'Origin':
            end = H.nodes[edge[1]]['pos']
            edge_x_pos = []
            edge_y_pos = []
            #edge_x, edge_y = addEdge(start, end, edge_x, edge_y, 1, 'end', .02, 6, 40)
            edge_x_pos, edge_y_pos = addEdge(start, end, edge_x_pos, edge_y_pos, 1, 'end', .02, arrowangle, 30)
            edge_x.extend(edge_x_pos)
            edge_y.extend(edge_y_pos)
            edge_text = str(master_years[edge])
            total_intros = str(master_intros[edge])
            edge_text = "First introduction: " + edge_text +  " | Total introductions: " + total_intros
            edge_text_list = []
            for i in range(9): #each label needs to be duplicated 9 times for each of the 9 points of the drawn arrows from addEdge()
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
            else :
                edge_color = "#4C484E"
                edge_weight = 5

            trace = go.Scatter( #creates a trace for each edge, appends to list to be drawn later)
            x=edge_x_pos, y=edge_y_pos,
            line=dict(width=edge_weight, color= edge_color),
            hoverinfo='text',
            text = (edge_text_list),
            mode='lines')
            edge_trace_list.append(trace)
    
    node_annotations = []
    #for node in H.nodes():
    total_prob_dict = {}
    total_prob_list = []
    #selects probabilites based on the year from the year_selection_slider        
    column_to_select = 'Agg Prob Intro '
    column_to_select = column_to_select + str(year_selection_slider)
    presence_select = "Presence "
    presence_select = presence_select + str(year_selection_slider)
    selection = []
    for node in H.nodes():
        if node != 'Origin':
            namerow = probability_data.loc[probability_data['NAME'] == node]
            pres = list(namerow[presence_select])
            if pres[0] == True: #currently selecting probabililities the same if pest is introduced or not. Pest presence at the moment initialized from O/D pairs
                
                dat = list(namerow[column_to_select]) 
                total_prob_list.append(dat[0])
                H.nodes[node]['nod_col'] = dat[0]
                selection.append(node)

                prob_trimmed = str(dat[0])
                prob_trimmed = prob_trimmed[0:4]
                total_prob_dict[node] = prob_trimmed
            else:

                
                dat = list(namerow[column_to_select]) 
                total_prob_list.append(dat[0])
                H.nodes[node]['nod_col'] = dat[0]
                selection.append(node)

                prob_trimmed = str(dat[0])
                prob_trimmed = prob_trimmed[0:4]
                total_prob_dict[node] = prob_trimmed

    #sets the color for node borders
    def SetColorEdge(x):
        if(x == 0):
            return "#1E1820"
        elif(x == 9999 and layout_opts == "twopi"): #green border for US
            return "green"
        elif(x >= 10):
            return "#AAA8AA"
        elif(x >= 100):
            return "#7B787C"
        elif (x >= 150):
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
    
    for node in H.nodes(): # node annotations 
        if node != 'Origin':
            x, y = H.nodes[node]['pos']
            node_anno_text.append("<b>" + node + "</b><br> <br>P(intro):  " + str(total_prob_dict[node]) +  " <br>Introductions: <br>" + total_intros_dict[node] )
            node_anno_size.append(25)
            node_anno_col.append('blue')
            node_x.append(x)
            node_y.append(y)
            node_num_intros.append(master_node_intros[node])
    

    colorscale = [
        [0, 'rgba(77, 67, 58, .9)'],
        [0.25, 'rgba(132, 82, 33, .8)'],
        [0.5, 'rgba(194, 116, 37, .8)'], 
        [0.75, 'rgba(213, 77, 40, .8)'],
        [1.0, 'rgba(196, 37, 37, .8)']
    ]
    
    #node traces
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        
        hoverinfo= 'text',
        hovertext = node_anno_text,
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale=colorscale,
            reversescale=False,
            color=[],
            size=35,
            colorbar=dict(
                thickness=35,
                title='Probability of Introduction',
                titlefont = dict(color = 'white', size = 14),
                tickfont = dict(color ='white'),
                xanchor='left',
                titleside='right',
                bgcolor = '#19191a',
            
            ),
            
            line_width=3,
            line_color = list(map(SetColorEdge, node_num_intros))))

    
    node_annotations = []
    annotations = []
    for node in H.nodes(): #sets color for node ISO code annotations, which float over each node
        if node != 'Origin':     
            x, y = H.nodes[node]['pos']
            node_col = H.nodes[node]['nod_col']
            node_text = country_codes_dict[node] 
            if node_col > .65:
                text_col = 'white'
            else:
                text_col = 'white'

            annotations.append(
        dict(x=x,
            y=y,
            xref="x",
            yref="y",
            text= node_text, # node name that will be displayed
            xanchor='right',
            xshift=15,
            font=dict(color=text_col, size=12),
            showarrow=False, arrowhead=1, ax=-10, ay=-10),
            
        )


    node_text = []

    node_trace.marker.color = total_prob_list
    node_trace.text = selection



    fig = go.Figure(data=[node_trace],
                layout=go.Layout(
                    plot_bgcolor='#19191a',
                    
                    paper_bgcolor = '#19191a',
                    titlefont_size=16,
                    showlegend=False,
                    
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0, pad=0),
                    
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
                )
    for trace in edge_trace_list:
        fig.add_trace(trace)
    

    fig.update_layout(
        height = 950, #sets fig size - could potentially be adaptive
        showlegend=False,
        annotations= annotations #shows iSO annotations
    )
    
    return dcc.Graph(figure=fig) , num_introduced_countries, introduction_tally, truncate(avg_reintros,2), truncate(median_reintros,2), truncate(avg_indegree,2), truncate(avg_outdegree,2), top_in_deg, top_out_deg


        
        
'''
@app.callback(

        Output('graphic', 'clickData'),
        [Input('reset_btn', 'n_clicks')])
def update(reset):
    return None
'''

if __name__ == '__main__':
    app.run_server(debug=True)

