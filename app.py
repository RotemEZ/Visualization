import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as sch

# Load the dataset
spotify = pd.read_csv("spotify-2023.csv", encoding='ISO-8859-1')

# Remove strange row
# Convert streams, in_shazam_charts and in_deezer_playlists columns to numeric type
spotify["streams"] = pd.to_numeric(spotify['streams'], errors='coerce')
spotify["in_deezer_playlists"] = pd.to_numeric(spotify['in_deezer_playlists'].str.replace(",", ""), errors='coerce')
spotify["in_shazam_charts"] = pd.to_numeric(spotify['in_shazam_charts'], errors='coerce')


# Remove released_day column(not relevant) and key column(too many missing values)
# Remove instrumentalness_%, liveness_%, speechiness_% columns(not relevant a significant amount of low values)
spotify.drop(columns=['released_day', "key", 'instrumentalness_%', 'liveness_%', 'speechiness_%'], inplace=True)

# Convert each percentage column to a range between 0 and 1
percentage_columns = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%']
for column in percentage_columns:
    spotify[column] = spotify[column] / 100


energy_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
energy_labels = ['Very Low (0-0.2)', 'Low (0.2-0.4)', 'Medium (0.4-0.6)', 'High (0.6-0.8)', 'Very High (0.8-1.0)']
valence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
valence_labels = ['Very Low (0-0.2)', 'Low (0.2-0.4)', 'Medium (0.4-0.6)', 'High (0.6-0.8)', 'Very High (0.8-1.0)']

# Bin the energy and valence columns
spotify['energy level'] = pd.cut(spotify['energy_%'], bins=energy_bins, labels=energy_labels)
spotify['valence level'] = pd.cut(spotify['valence_%'], bins=valence_bins, labels=valence_labels)

spotify.drop(columns=['energy_%', 'valence_%' ], inplace=True)

# Fill all null values in the 'in_shazam_charts' column with 0
spotify['in_shazam_charts'].fillna(0, inplace=True)


# Convert the 'released_month' column from numbers to month names
spotify['released_month'] = pd.to_datetime(spotify['released_month'], format='%m').dt.strftime('%B')

# Set up the Streamlit app
st.set_page_config(layout="wide")  # Set the layout to wide mode
st.title('Spotify User Preferences Dashboard')


# Define columns layout
col1, empty, col2= st.columns([2, 0.1, 2])
with col1:
    st.markdown('### Top 10 Most Popular Artists by Streams')
    # Most Popular Artists
    min_year, max_year = st.slider('Select Year Range for Bar Chart', min_value=int(spotify['released_year'].min()),
                                   max_value=int(spotify['released_year'].max()),
                                   value=(int(spotify['released_year'].min()), int(spotify['released_year'].max())))
    filtered_data_bar = spotify[(spotify['released_year'] >= min_year) & (spotify['released_year'] <= max_year)]

    # Most Popular Artists
    artist_popularity = filtered_data_bar.groupby('artist(s)_name')['streams'].sum().reset_index()
    artist_popularity = artist_popularity.sort_values(by='streams', ascending=False).head(10)
    artist_popularity['streams_billions'] = artist_popularity['streams'] / 1e9
    bar_fig = px.bar(
        artist_popularity,
        x='streams_billions',
        y='artist(s)_name',
        orientation='h',
        labels={'streams_billions': 'Total Streams (in billions)', 'artist(s)_name': 'Artist'},
        color='streams_billions',
        color_continuous_scale='Blues'  # Using color saturation with 'Blues' color scale
    )
    bar_fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400  
    )
    bar_fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("### Distribution of Streams by Song's Release Month")
    # Define month names
    min_year_box, max_year_box = st.slider('Select Year Range for Box Plot',
                                           min_value=int(spotify['released_year'].min()),
                                           max_value=int(spotify['released_year'].max()), value=(
        int(spotify['released_year'].min()), int(spotify['released_year'].max())), key='box_slider')
    filtered_data_box = spotify[(spotify['released_year'] >= min_year_box) & (spotify['released_year'] <= max_year_box)]

    # Define month names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']
    filtered_data_box['released_month'] = pd.Categorical(filtered_data_box['released_month'], categories=month_names,
                                                         ordered=True)

    # Box Plot for Streams by Month
    box_fig_month = px.box(
        filtered_data_box,
        y='released_month',
        x='streams',
        category_orders={'released_month': month_names},  # Ensure months are ordered correctly
        color_discrete_sequence=['purple']  # Set color to purple
    )

    # Update layout for better readability and increase spacing between categories
    box_fig_month.update_layout(
        yaxis_tickangle=0,
        yaxis_categoryorder='array',
        yaxis_categoryarray=month_names[::-1],  # Reverse the order of the months
        yaxis_title=None,
        height=900,
        width=700  
    )

    st.plotly_chart(box_fig_month, use_container_width=True)

with col2:
    st.markdown('### Sum of Spotify Charts by Energy and Valence Combinations')

    # Group by combinations of energy and valence bins and calculate the sum of charts
    combination_charts_sum = spotify.groupby(['energy level', 'valence level'])['in_spotify_charts'].sum().reset_index()

    # Create the line plot with sum of charts
    line_combination_fig_sum = px.line(
        combination_charts_sum,
        x='energy level',
        y='in_spotify_charts',
        color='valence level',
        labels={'energy level': 'Energy Levels', 'in_spotify_charts': 'Sum of Spotify Charts',
                'valence level': 'Valence Levels'},
        markers=True,
        line_shape='linear'
    )

    # Update the layout to expand the figure width
    line_combination_fig_sum.update_layout(
        width=1000, height=600,  
        xaxis=dict(
            title='Energy Levels',
            tickmode='array',
            tickvals=['Very Low (0-0.2)', 'Low (0.2-0.4)', 'Medium (0.4-0.6)', 'High (0.6-0.8)', 'Very High (0.8-1.0)'],
            ticktext=['Very Low\n(0-0.2)', 'Low\n(0.2-0.4)', 'Medium\n(0.4-0.6)', 'High\n(0.6-0.8)',
                      'Very High\n(0.8-1.0)'],
            dtick=1,
            rangeslider=dict(
                visible=True
            )
        )
    )

    st.plotly_chart(line_combination_fig_sum, use_container_width=True)

    st.markdown('### Song Attribute vs. Number of Spotify Playlists')

    # Create Facet Scatter Plots
    spotify_melted = spotify.melt(id_vars=['in_spotify_playlists'], value_vars=['danceability_%', 'bpm', 'acousticness_%'], var_name='attribute', value_name='value')

    facet_fig = px.scatter(
        spotify_melted,
        x='value',
        y='in_spotify_playlists',
        facet_row='attribute',
        labels={
            'in_spotify_playlists': 'Number of Spotify Playlists',
            'value': ''
        },   category_orders={'attribute': ['danceability_%', 'bpm', 'acousticness_%']},
        facet_row_spacing=0.15,
)

    facet_fig.update_traces(marker=dict(size=5, color='black'))
    facet_fig.update_xaxes(matches=None, showticklabels=True)




    # Update x-axis titles to show the attribute name
    for i, axis in enumerate(facet_fig.layout.annotations):
        axis_title = facet_fig.layout.annotations[i]['text'].split('=')[1].capitalize().replace('_%', '')
        facet_fig.layout[f'xaxis{i + 1}']['title']['text'] = axis_title

    facet_fig.update_layout(
        width=600,  
        height=800  
    )
    st.plotly_chart(facet_fig)


st.markdown('#### Correlation Heatmap of Song Success Across Different Platforms')


columns = [
        'streams', 'in_spotify_playlists', 'in_spotify_charts',
        'in_apple_charts', 'in_apple_playlists', 'in_deezer_playlists',
        'in_deezer_charts', 'in_shazam_charts']

correlation_matrix = spotify[columns].corr().round(2)

    # Generate the linkage matrix for clustering
linkage = sch.linkage(correlation_matrix, method='ward')

    # Create a dendrogram to get the order of the features
dendrogram = sch.dendrogram(linkage, labels=correlation_matrix.columns, no_plot=True)
clustered_order = dendrogram['ivl']

    # Reorder the correlation matrix
correlation_matrix = correlation_matrix.loc[clustered_order, clustered_order]

correlation_matrix = correlation_matrix.iloc[:, ::-1]

    # Heatmap using Plotly Express with PuOr colorscale
heatmap_fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        colorscale='dense',
        showscale=True,
        hoverinfo="none",
        font_colors=["black", "white"]
    )

heatmap_fig.update_layout(
        xaxis_title='Features',
        yaxis_title='Features',
        xaxis=dict(side='bottom', tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=100, r=20, t=50, b=150)  
    )

st.plotly_chart(heatmap_fig, use_container_width=True)
