import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------
# Data + features
mids_selected = pd.read_csv("mids_selected.csv", encoding="utf-8")
metadata_cols = ['Player', 'Nation', 'Squad', 'Comp', 'Age', '90s']
features = [
    'Gls', 'Ast', 'xG', 'npxG', 'xA', 'SCA', 'Sh', 'SoT', 'SoT%',
    'Cmp', 'Cmp%', 'PrgP', 'KP', 'PPA', 'CrsPA',
    'Carries', 'PrgC', 'PrgR', 'CPA', 'PrgDist',
    'Tkl', 'TklW', 'Int', 'Blocks', 'Mis', 'Dis', 'Fld', 'Err'
]

# -------------------------
# Compute PCA, distances, clustering, and MDS
@st.cache_data
def compute_embeddings(df, n_clusters=4):
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.9)
    X_pca = pca.fit_transform(X_scaled)

    dist_euc = euclidean_distances(X_pca)
    dist_cos = cosine_distances(X_pca)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    mds_euc = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords_euc = mds_euc.fit_transform(dist_euc)

    mds_cos = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords_cos = mds_cos.fit_transform(dist_cos)

    return X_pca, dist_euc, dist_cos, clusters, coords_euc, coords_cos

X_pca, dist_euclidean, dist_cosine, clusters, coords_euclidean, coords_cosine = compute_embeddings(mids_selected)
mids_selected["Cluster"] = clusters
mids_selected["MDS1_euclidean"], mids_selected["MDS2_euclidean"] = coords_euclidean[:,0], coords_euclidean[:,1]
mids_selected["MDS1_cosine"], mids_selected["MDS2_cosine"] = coords_cosine[:,0], coords_cosine[:,1]


@st.cache_data
def compute_percentiles(df):
    return df.set_index("Player").rank(pct=True)

percentile_ranks = compute_percentiles(mids_selected)

# -------------------------
# Streamlit
st.title("Midfielder Similarity Explorer")
tab1, tab2, tab3 = st.tabs(["Player Similarity Search", "Exploratory Plots", "Player Comparison"])

# =========================
# Similarity Search
with tab1:
    distance_metric = st.radio("Choose distance measure:", ["Euclidean", "Cosine"])
    x_col = "MDS1_euclidean" if distance_metric == "Euclidean" else "MDS1_cosine"
    y_col = "MDS2_euclidean" if distance_metric == "Euclidean" else "MDS2_cosine"

    selected_player = st.selectbox("Select a player:", mids_selected["Player"].unique())

    def closest_players(player_name, top_n=8, distance_matrix=dist_euclidean):
        idx = mids_selected.index[mids_selected["Player"] == player_name][0]
        distances = distance_matrix[idx]
        closest_idx = distances.argsort()[1:top_n+1]
        return mids_selected.iloc[closest_idx]

    distance_matrix = dist_euclidean if distance_metric == "Euclidean" else dist_cosine
    closest_df = closest_players(selected_player, distance_matrix=distance_matrix)

    closest_display = closest_df[["Player","Squad","Nation","Age","90s"]].reset_index(drop=True)
    closest_display["Age"] = closest_display["Age"].map("{:.0f}".format)
    closest_display["90s"] = closest_display["90s"].map("{:.1f}".format)
    st.write(f"### 8 players most similar to **{selected_player}**")
    st.dataframe(closest_display, use_container_width=True)

    highlight_idx = [mids_selected.index[mids_selected["Player"] == selected_player][0]] + list(closest_df.index)
    mids_selected["highlight"] = ["Selected/Similar" if i in highlight_idx else "Other" for i in mids_selected.index]
    mids_selected["text_label"] = [p if i in highlight_idx else "" for i, p in enumerate(mids_selected["Player"])]

    fig = px.scatter(
        mids_selected,
        x=x_col,
        y=y_col,
        color="highlight",
        color_discrete_map={"Selected/Similar": "red", "Other": "lightgrey"},
        hover_data={'Player': True, 'Squad': True, 'Nation': True, '90s': True, 'Comp': True, 'Age': True, x_col: False, y_col: False, 'text_label': False},
        text="text_label",
        labels={x_col: 'Dimension 1', y_col: 'Dimension 2', 'highlight':'Player'},
        title=f"MDS Plot of Midfielders ({distance_metric} Distance)"
    )
    fig.update_traces(textposition="top center", marker=dict(size=8))
    fig.update_layout(height=800, width=1300)
    st.plotly_chart(fig)

# =========================
# Exploratory Plots
with tab2:
    top_players = mids_selected.nlargest(25, '90s')['Player'].values
    mids_selected['text_label'] = [p if p in top_players else '' for p in mids_selected['Player']]

    st.subheader("Euclidean MDS with Clusters")
    fig_euclid = px.scatter(
        mids_selected, x='MDS1_euclidean', y='MDS2_euclidean',
        color=mids_selected['Cluster'].astype(str), text='text_label',
        hover_data={'Player': True,'Squad': True,'Nation': True,'90s': True,'Comp': True,'Age': True,'MDS1_euclidean': False,'MDS2_euclidean': False,'text_label': False},
        labels={'MDS1_euclidean':'Dimension 1','MDS2_euclidean':'Dimension 2','color':'Cluster'},
        title='MDS plot of Midfielders: Euclidean Distance'
    )
    fig_euclid.update_traces(marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),textposition='top center',textfont=dict(size=8.5))
    fig_euclid.update_layout(height=800,width=1300)
    st.plotly_chart(fig_euclid)

    st.subheader("Cosine MDS with Clusters")
    fig_cosine = px.scatter(
        mids_selected, x='MDS1_cosine', y='MDS2_cosine',
        color=mids_selected['Cluster'].astype(str), text='text_label',
        hover_data={'Player': True,'Squad': True,'Nation': True,'90s': True,'Comp': True,'Age': True,'MDS1_cosine': False,'MDS2_cosine': False,'text_label': False},
        labels={'MDS1_cosine':'Dimension 1','MDS2_cosine':'Dimension 2','color':'Cluster'},
        title='MDS plot of Midfielders: Cosine Distance'
    )
    fig_cosine.update_traces(marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),textposition='top center',textfont=dict(size=8.5))
    fig_cosine.update_layout(height=800,width=1300)
    st.plotly_chart(fig_cosine)

# =========================
# Player Comparison
with tab3:
    st.header("Midfielder Comparison")

    abstract_metrics = {
        "Shooting": ['Gls', 'xG', 'npxG', 'Sh', 'SoT', 'SoT%'],
        "Creativity": ['Ast', 'xA', 'SCA', 'CrsPA', 'PPA'],
        "Progression": ['PrgP', 'Carries', 'PrgC', 'CPA', 'PrgDist'],
        "Passing": ['Cmp', 'Cmp%', 'KP'],
        "Defence": ['Tkl', 'TklW', 'Int', 'Blocks'],
        "Errors": ['Mis', 'Dis', 'Err']
    }

    players_to_compare = st.multiselect(
        "Select players to compare:", mids_selected["Player"].unique(),
        default=mids_selected["Player"].iloc[:2]
    )

    radar_df = pd.DataFrame(index=players_to_compare, columns=abstract_metrics.keys())
    for metric_name, metric_features in abstract_metrics.items():
        metric_features = [f for f in metric_features if f in percentile_ranks.columns]
        radar_df.loc[players_to_compare, metric_name] = percentile_ranks.loc[players_to_compare, metric_features].mean(axis=1)
    radar_df["Errors"] = 1 - radar_df["Errors"]

    # Radar chart
    fig_radar = go.Figure()
    colors = ['red','blue','green','orange','purple']
    def rgba(color_name, alpha=0.2):
        import matplotlib.colors as mcolors
        r,g,b = [int(x*255) for x in mcolors.to_rgb(color_name)]
        return f'rgba({r},{g},{b},{alpha})'

    for i, player in enumerate(players_to_compare):
        values = radar_df.loc[player].values.astype(float)
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_df.columns,
            fill='toself',
            name=player,
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=rgba(colors[i % len(colors)], alpha=0.2)
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True,
        title="Midfielder Radar Comparison (Percentiles)",
        width=700,
        height=700,
        margin=dict(l=120,r=50,t=50,b=50)
    )
    st.plotly_chart(fig_radar)

    # Bar charts for individual features
    for metric_name, metric_features in abstract_metrics.items():
        metric_features = [f for f in metric_features if f in percentile_ranks.columns]
        if not metric_features:
            continue
        fig = go.Figure()
        for i, player in enumerate(players_to_compare):
            values = percentile_ranks.loc[player, metric_features].values
            fig.add_trace(go.Bar(
                x=metric_features,
                y=values,
                name=player,
                orientation='v',
                marker=dict(color=colors[i % len(colors)], opacity=0.75, line=dict(width=1,color=colors[i % len(colors)])),
                text=[f"{v:.2f}" for v in values],
                textposition='inside',
                textfont=dict(size=10)
            ))
        fig.update_layout(
            barmode='group',
            yaxis=dict(title='Percentile', range=[0,1]),
            xaxis=dict(title=f"{metric_name} Features", automargin=True),
            title=f"{metric_name} Percentile Comparison",
            width=900,
            height=300 + 20 * len(metric_features)
        )
        st.plotly_chart(fig, use_container_width=True)
