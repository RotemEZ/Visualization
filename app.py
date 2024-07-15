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


# Remove released_day, artist_count, mode columns (not relevant), key column(too many missing values)
# Remove instrumentalness_%, liveness_%, speechiness_% columns(not relevant a significant amount of low values)
spotify.drop(columns=['released_day', "key", 'instrumentalness_%', 'liveness_%', 'speechiness_%', 'artist_count',
                      'mode'], inplace=True)

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

spotify.drop(columns=['energy_%', 'valence_%'], inplace=True)

# Fill all null values in the 'in_shazam_charts' column with 0
spotify['in_shazam_charts'].fillna(0, inplace=True)


# Convert the 'released_month' column from numbers to month names
spotify['released_month'] = pd.to_datetime(spotify['released_month'], format='%m').dt.strftime('%B')

# Set up the Streamlit app
st.set_page_config(layout="wide")  # Set the layout to wide mode

# Title and description
st.markdown("<h1 style='text-align: left;'>Spotify User Preferences Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left;'>This visualization explores the most streamed songs on Spotify. Spotify is a popular music streaming app with over 615 million monthly users, including 239 million paid subscribers. Understanding user preferences can provide insights into global entertainment culture and current musical trends, aiding business decisions for app managers, record companies, and musicians.</p>", unsafe_allow_html=True)

# Define columns layout
col1, empty, col2 = st.columns([2, 0.1, 2])

with col1:
    st.markdown('### Top 10 Most Popular Artists by Streams')

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
        height=400,
        xaxis_title=dict(
            text='Total Streams (in billions)',
            font=dict(size=17)
        ),
        yaxis_title=dict(
            text='Artist',
            font=dict(size=17)
        )
    )
    bar_fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("### Distribution of Streams by Song's Release Month")
    # Define month names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']

    selected_months = st.multiselect('Select Months', month_names, default=month_names)

    filtered_data_box = spotify[spotify['released_month'].isin(selected_months)]
    filtered_data_box['released_month'] = pd.Categorical(filtered_data_box['released_month'],
                                                         categories=selected_months, ordered=True)

    # Box Plot for Streams by Month
    box_fig_month = px.box(
        filtered_data_box,
        y='released_month',
        x='streams',
        category_orders={'released_month': selected_months},
        color_discrete_sequence=['purple'],
        points=False
    )

    box_fig_month.update_layout(
        yaxis_tickangle=0,
        yaxis_categoryorder='array',
        yaxis_title=None,
        xaxis_title=dict(
            text='Streams',
            font=dict(size=17)
        ),
        height=850,
        width=700
    )
    box_fig_month.update_xaxes(tickfont=dict(size=14))
    box_fig_month.update_yaxes(tickfont=dict(size=14))

    st.plotly_chart(box_fig_month, use_container_width=True)

with col2:
    st.markdown('### Average of Spotify Charts by Energy and Valence Combinations')

    # Group by combinations of energy and valence bins and calculate the average of charts
    combination_charts_avg = spotify.groupby(['energy level', 'valence level'])[
        'in_spotify_charts'].mean().reset_index()

    # Create the line plot with average of charts
    line_combination_fig_avg = px.line(
        combination_charts_avg,
        x='energy level',
        y='in_spotify_charts',
        color='valence level',
        labels={'energy level': 'Energy Levels', 'in_spotify_charts': 'Average of Spotify Charts',
                'valence level': 'Valence Levels'},
        markers=True,
        line_shape='linear'
    )

    line_combination_fig_avg.update_layout(
        width=1000, height=800,
        xaxis=dict(
            title='Energy Levels',
            tickmode='array',
            tickvals=['Very Low (0-0.2)', 'Low (0.2-0.4)', 'Medium (0.4-0.6)', 'High (0.6-0.8)', 'Very High (0.8-1.0)'],
            ticktext=['Very Low\n(0-0.2)', 'Low\n(0.2-0.4)', 'Medium\n(0.4-0.6)', 'High\n(0.6-0.8)',
                      'Very High\n(0.8-1.0)'],
            dtick=1,
            tickangle=50,
            rangeslider=dict(
                visible=True
            ),
            titlefont=dict(size=17)
        ),
        yaxis=dict(
            title='Average of Spotify Charts',
            titlefont=dict(size=17),
            tickfont=dict(size=14)
        ),
        legend=dict(
            title=dict(
                text='Valence Levels',
                font=dict(size=17)
            ),
            font=dict(size=14)
        )
    )
    line_combination_fig_avg.update_xaxes(tickfont=dict(size=14))
    line_combination_fig_avg.update_yaxes(tickfont=dict(size=14))

    st.plotly_chart(line_combination_fig_avg, use_container_width=True)

    st.markdown('### Song Attribute vs. Number of Spotify Playlists')

    # Create Facet Scatter Plots
    spotify_melted = spotify.melt(id_vars=['in_spotify_playlists', 'track_name'],
                                  value_vars=['danceability_%', 'bpm', 'acousticness_%'], var_name='attribute',
                                  value_name='value')

    facet_fig = px.scatter(
        spotify_melted,
        x='value',
        y='in_spotify_playlists',
        facet_row='attribute',
        hover_data={'track_name': True, 'attribute': False},
        labels={
            'in_spotify_playlists': 'Number of Spotify Playlists',
            'value': "Attribute's value",
            'track_name': 'Song Name'
        },
        category_orders={'attribute': ['danceability_%', 'bpm', 'acousticness_%']},
        facet_row_spacing=0.15
    )

    facet_fig.update_traces(marker=dict(size=5, color='black'))
    facet_fig.update_xaxes(matches=None, showticklabels=True)

    # Update x-axis titles to show the attribute name
    for i, axis in enumerate(facet_fig.layout.annotations):
        axis_title = facet_fig.layout.annotations[i]['text'].split('=')[1].capitalize().replace('_%', '')
        facet_fig.layout[f'xaxis{i + 1}']['title']['text'] = axis_title

    facet_fig.for_each_annotation(lambda a: a.update(text=''))

    facet_fig.update_layout(
        width=600,
        height=700
    )

    st.plotly_chart(facet_fig, use_container_width=True)


st.markdown('#### Correlation Heatmap of Song Success Across Different Platforms')


columns = [
        'streams', 'in_spotify_playlists', 'in_spotify_charts',
        'in_apple_charts', 'in_apple_playlists', 'in_deezer_playlists',
        'in_deezer_charts', 'in_shazam_charts']

correlation_matrix = spotify[columns].corr().round(2)


linkage = sch.linkage(correlation_matrix, method='ward')


dendrogram = sch.dendrogram(linkage, labels=correlation_matrix.columns, no_plot=True)
clustered_order = dendrogram['ivl']


correlation_matrix = correlation_matrix.loc[clustered_order, clustered_order]

correlation_matrix = correlation_matrix.iloc[:, ::-1]


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
    xaxis_title=dict(
        text='Features',
        font=dict(size=17)
    ),
    yaxis_title=dict(
        text='Features',
        font=dict(size=17)
    ),
    xaxis=dict(
        side='bottom',
        tickangle=45,
        tickfont=dict(size=13)
    ),
    yaxis=dict(
        tickfont=dict(size=13)
    ),
    font=dict(
        size=14),
    margin=dict(l=100, r=20, t=20, b=70)
)

st.plotly_chart(heatmap_fig, use_container_width=True)
