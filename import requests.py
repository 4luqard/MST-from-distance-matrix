import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy

def get_top_50_cryptos():
    """
    Fetches the top 50 cryptocurrencies by market capitalization using CoinGecko API.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&page=1"
    response = requests.get(url)
    data = response.json()

    # Extract the symbols of the top 50 cryptocurrencies
    top_50_symbols = [coin['symbol'].upper() for coin in data]  # Convert to uppercase to match Binance symbols
    return top_50_symbols

def get_unique_crypto_symbols_filtered(top_50_symbols):
    """
    Fetches all trading symbols from Binance, splits them into individual symbols,
    and returns a unique list of cryptocurrency symbols that are in the top 50 by market cap.
    """
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    
    all_symbols = [symbol['symbol'] for symbol in data['symbols']]
    return all_symbols
    unique_symbols = set()
    
    for symbol_pair in all_symbols:
        base_currencies = ['BTC', 'ETH', 'USDT', 'BUSD', 'XRP', 'LTC', 'ADA', 'BNB']
        for base_currency in base_currencies:
            if symbol_pair.endswith(base_currency):
                base = base_currency
                quote = symbol_pair.replace(base_currency, '')
                #if base in top_50_symbols:
                unique_symbols.add(base)
                #if quote in top_50_symbols:
                unique_symbols.add(quote)
                break
                
    # Filter to only include symbols that are in the top 50
    filtered_symbols = [symbol for symbol in unique_symbols if symbol in top_50_symbols]
    
    return unifiltered_symbols

def get_time_range():
    """
    Calculates the start and end timestamps for the last week.
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=28)
    return int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000)

def get_binance_candles(symbol, start_time, end_time, interval='5m'):
    url = "https://fapi.binance.com/fapi/v1/klines"
    limit = 1000  # Max limit allowed by Binance
    all_candles = []

    while start_time < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit,
        }
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_candles.extend(data)
        start_time = data[-1][6] + 1  # Prepare for the next loop iteration
        
    # Convert directly to DataFrame
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(all_candles, columns=columns)
    df['asset'] = symbol
    return df
"""
top_50_cryptos = get_top_50_cryptos()
symbols = get_unique_crypto_symbols_filtered(top_50_cryptos)
#symbols = [f"{symbol}USDT" for symbol in symbols if not 'USDT' == symbol]
start_time, end_time = get_time_range()

columns = [
        'asset', 'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]

assets = pd.DataFrame(columns=columns)

for symbol in symbols:  # Example: limit to first 1 symbols for demonstration
    print(f"Fetching data for {symbol}")
    asset = get_binance_candles(symbol, start_time, end_time)
    assets = pd.concat([assets, asset], ignore_index=True)
    print(f"Number of candles fetched for {symbol}: {len(asset)}")  # changed candles to asset
    

assets.to_csv('assets.csv', index=False)
"""
assets = pd.read_csv('assets.csv')

assets = assets[assets['asset'].str.endswith('USDT')]
assets = assets[~assets['asset'].isin({"BTCSTUSDT", "BTSUSDT", "COCOSUSDT",
                                       "CVCUSDT", "FTTUSDT", "HNTUSDT",
                                       "RAYUSDT", "SCUSDT", "SRMUSDT", "TOMOUSDT", "USDCUSDT"})]

assets['open_time'] = pd.to_datetime(assets['open_time'])

assets['close'] = assets['close'].astype(float)

assets['percentage_change'] = assets.groupby('asset')['close'].transform(lambda x: x.diff() / x.shift(1) * 100)

baseline = 100

assets['value'] = assets.groupby('asset')['percentage_change'].transform(lambda x: baseline * (1 + x / 100).cumprod())

pivot_table = assets.pivot(index='open_time', columns='asset', values='value')

display(assets[assets['asset'] == 'ETHUSDT'])

pivot_table.plot()

df = assets.copy()
# %%
import dcor

df = df[df['asset'].str.endswith("USDT")]
corr_df = pd.concat([asset.rename(asset_name) for asset_name, asset in df.set_index('open_time').groupby('asset', as_index=False)['close']], axis=1)
#corr_df = corr_df.dropna(axis=1).pct_change()
corr_df = corr_df.dropna(axis=1).rolling(24 * 3).rank(pct=True)
corr_df = corr_df.dropna()
#corr_df = corr_df.drop(columns=['BONKUSDT', 'XMRUSDT', 'WBTCUSDT'])
bad_symbols = [c for c in corr_df.columns if len(corr_df[c].value_counts()) == 1]

corr_df = corr_df.drop(columns=bad_symbols)

def distance_corr_matrix(df):
    n = df.shape[1]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                # The distance correlation of a variable with itself is 1
                corr_matrix[i, j] = 1.0
            elif i < j:
                # Calculate distance correlation and fill in the matrix symmetrically
                corr = dcor.distance_correlation(df.iloc[:, i], df.iloc[:, j])
                corr_matrix[i, j] = corr_matrix[j, i] = corr
    return corr_matrix

#corrmat = distance_corr_matrix(corr_df)
corrmat = corr_df.corr(method='spearman')
corrmat = pd.DataFrame(corrmat, index=corr_df.columns, columns=corr_df.columns)

plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

sns.clustermap(corrmat, method='ward', figsize=(12,9), cmap='coolwarm')
plt.show()

corr_df
# %%
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import networkx as nx

def plot_mst(corrmat, corr_df):
    num_assets = corrmat.shape[1]

    distance_matrix = np.sqrt(2 * (1 - corrmat))

    dist_df = pd.DataFrame(distance_matrix, index=corr_df.columns,
                           columns=corr_df.columns)

    # Generate the MST using the networkx library
    G = nx.from_pandas_adjacency(dist_df)

    # Use Kruskal's algorithm to find the MST (networkx uses Kruskal's by default)
    T = nx.minimum_spanning_tree(G)

    # Draw the MST
    pos = nx.spring_layout(T)  # positions for all nodes

    # Function to create the interactive MST plot
    def plot_interactive_mst(T, pos):
        edge_x = []
        edge_y = []
        edge_weight = []

        for edge in T.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weight.append(weight)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []

        for node in T.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                size=10,
                line_width=2))

        # Add edge weights to the plot
        middle_node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='text',
            textposition="bottom center",
            hoverinfo='none'
        )

        for edge in T.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            middle_node_trace['x'] += tuple([(x0 + x1) / 2])
            middle_node_trace['y'] += tuple([(y0 + y1) / 2])
            weight = edge[2]['weight']
            middle_node_trace['text'] += tuple([f'{weight:.2f}'])

        fig = go.Figure(data=[edge_trace, node_trace, middle_node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
        return fig

    # Using the positions from the previous matplotlib plot
    interactive_fig = plot_interactive_mst(T, pos)
    interactive_fig.show()
    
plot_mst(corrmat, corr_df)
# %%

fdf = assets.copy()

fdf = pd.concat([asset.rename(asset_name) for asset_name, asset in fdf.set_index('open_time').groupby('asset', as_index=False)['close']], axis=1)

cutoff_date = pd.Timestamp.now() - timedelta(days=14)

fdf = fdf[fdf.index > cutoff_date]

# %%
import hdbscan

cfdf = fdf.corr(method='spearman')

plot_mst(cfdf, fdf)

# %%
CM = np.sqrt(2 * (1 - corr_df.corr().values))

HDB = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')

clusters = pd.Series(HDB.fit_predict(CM.astype(np.float64)), index=corr_df.columns).sort_values()

display(clusters)

# %%
import statsmodels.api as sm

display(assets[assets['asset'].isin(["MANAUSDT", "SANDUSDT", "ENJUSDT", "AXSUSDT"])])


data = assets[assets['asset'].isin(["MANAUSDT", "SANDUSDT", "ENJUSDT", "AXSUSDT"])]
y_train = data['close']
data.drop(columns='close')
X_train = data

X_train = sm.add_constant(X_train)
model = sm.OLS(X_train, y_train).fit()

# %%
volume_hist = assets.query("asset != 'BTCUSDT' & asset != 'ETHUSDT'")
display(volume_hist[volume_hist['asset'] == 'ETHUSDT'])

volume_hist['hour'] = volume_hist['open_time'].dt.hour

grouped_volume = volume_hist.groupby('hour')['quote_asset_volume'].mean().reset_index()
# %%
grouped_volume
# %%
plt.figure(figsize=(10, 6))  # Specifies the figure size

# Since 'hour' is categorical, we use the number of unique 'hour' values to define the bins
# We add 1 to the number of bins because there are 24 hours in a day and we need 24 bins, not 23.
# Also, we set the range to (-0.5, 23.5) to center the bins around the integer hour values.
plt.hist(grouped_volume['hour'], bins=24, range=(-0.5, 23.5), weights=grouped_volume['quote_asset_volume'])

plt.title('Volume by Hour')  # Adds a title to the histogram
plt.xlabel('Hour of the Day')  # Adds a label to the x-axis
plt.ylabel('Volume')  # Adds a label to the y-axis

# Display the histogram
plt.show()
# %%
volume_net_hist = assets.query("asset != 'BTCUSDT' & asset != 'ETHUSDT'")
volume_net_hist['hour'] = volume_net_hist['open_time'].dt.hour
display

volume_net_hist['volume_sell'] = volume_net_hist['quote_asset_volume'] - volume_net_hist['taker_buy_quote_asset_volume']
volume_net_hist['volume_net'] = np.abs(volume_net_hist['volume_sell'] - volume_net_hist['taker_buy_quote_asset_volume'])

grouped_net_volume = volume_net_hist.groupby('hour')['volume_net'].mean().reset_index()
#grouped_net_volume = volume_net_hist.groupby('asset').median().reset_index()

# %%
grouped_net_volume

# %%
plt.figure(figsize=(10, 6))  # Specifies the figure size

# Since 'hour' is categorical, we use the number of unique 'hour' values to define the bins
# We add 1 to the number of bins because there are 24 hours in a day and we need 24 bins, not 23.
# Also, we set the range to (-0.5, 23.5) to center the bins around the integer hour values.
plt.hist(grouped_net_volume['hour'], bins=24, range=(-0.5, 23.5), weights=grouped_net_volume['volume_net'])

plt.title('Volume by Hour')  # Adds a title to the histogram
plt.xlabel('Hour of the Day')  # Adds a label to the x-axis
plt.ylabel('Volume')  # Adds a label to the y-axis
# %%
