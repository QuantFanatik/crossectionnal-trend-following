import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

yf.pdr_override()
np.random.seed(63)

# Function to get the current list of S&P 500 stocks
def get_sp500_stocks():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = requests.get(url).content
    df_list = pd.read_html(html)
    df = df_list[0]
    return df['Symbol'].tolist()


def download_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)['Close']
    return data


def fit_markov_model(data, n_states=3):
    returns = np.log(data / data.shift(1)).dropna()
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    model.fit(returns.values.reshape(-1, 1))
    hidden_states = model.predict(returns.values.reshape(-1, 1))

    # Determine the state means
    state_means = [np.mean(returns[hidden_states == i]) for i in range(n_states)]
    min_state = np.argmin(state_means)
    max_state = np.argmax(state_means)

    # Map states to buy, hold, and sell
    state_mapping = {min_state: 'sell', max_state: 'buy'}
    for i in range(n_states):
        if i not in state_mapping:
            state_mapping[i] = 'hold'

    sorted_hidden_states = pd.Series(hidden_states).map(state_mapping).values

    return model, sorted_hidden_states, returns.index


def predict_markov_model(model, data):
    returns = np.log(data / data.shift(1)).dropna()
    hidden_states = model.predict(returns.values.reshape(-1, 1))

    # Determine the state means
    state_means = [np.mean(returns[hidden_states == i]) for i in range(model.n_components)]
    min_state = np.argmin(state_means)
    max_state = np.argmax(state_means)
    print(state_means, "min:", min_state, "max:", max_state)

    # Map states to buy, hold, and sell
    state_mapping = {min_state: 'sell', max_state: 'buy'}
    for i in range(model.n_components):
        if i not in state_mapping:
            state_mapping[i] = 'hold'

    sorted_hidden_states = pd.Series(hidden_states).map(state_mapping).values

    return sorted_hidden_states, returns.index


def trend_following_strategy(data, hidden_states, hidden_states_index, spy_cumulative_returns):
    ema_5 = data.ewm(span=5, adjust=False).mean()
    ema_200 = data.ewm(span=200, adjust=False).mean()

    signals = pd.DataFrame(index=data.index, columns=data.columns)
    signals[data.columns] = np.where(ema_5 > ema_200, 1, 0)

    state_signals = pd.Series(hidden_states, index=hidden_states_index)
    state_signals = state_signals.reindex(data.index).ffill().bfill()  # Align indices

    # Modify signals based on state
    signals[state_signals == 'sell'] = 0  # Stop trading
    signals[state_signals == 'hold'] = 0  # Continue trading based on strategy
    signals[state_signals == 'buy'] = signals  # Long state

    returns = data.pct_change()
    strategy_returns = signals.shift(1) * returns
    cumulative_returns = (1 + strategy_returns).cumprod()

    equal_weighted_return_strategy = cumulative_returns.mean(axis=1)

    # Calculate equal-weighted portfolio without the strategy
    equal_weighted_returns = returns.mean(axis=1)
    cumulative_equal_weighted_returns = (1 + equal_weighted_returns).cumprod()

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(equal_weighted_return_strategy.index, equal_weighted_return_strategy, label='Equal Weighted Strategy + HMM')
    ax.plot(cumulative_equal_weighted_returns.index, cumulative_equal_weighted_returns, label='Equal Weighted Portfolio')
    ax.plot(spy_cumulative_returns.index, spy_cumulative_returns, label='SPY Cumulative Returns', color='gray', alpha=0.5)

    # Highlight the sell states
    for state, date in zip(state_signals, state_signals.index):
        if state == 'sell' or state=='hold':
            ax.axvspan(date, date, color='red', alpha=0.1)

    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

    return strategy_returns


# Define training and testing sets
train_start_date = '1950-01-01'
train_end_date = '2002-07-31'
test_start_date = '2002-08-01'
test_end_date = '2023-06-23'

# Fit Markov model to SPY data
spy_data = download_stock_data(['SPY'], train_start_date, train_end_date)
markov_model, hidden_states, hidden_states_index = fit_markov_model(spy_data)

# Predict states for test set
spy_test_data = download_stock_data(['SPY'], test_start_date, test_end_date)
test_hidden_states, test_hidden_states_index = predict_markov_model(markov_model, spy_test_data)

# Calculate SPY cumulative returns for the test period
spy_returns = spy_test_data.pct_change().dropna()
spy_cumulative_returns = (1 + spy_returns).cumprod()

# Download S&P 500 stock data
symbols = get_sp500_stocks()
test_data = download_stock_data(symbols, test_start_date, test_end_date)

# Run the trend following strategy
trend_following_strategy(test_data, test_hidden_states, test_hidden_states_index, spy_cumulative_returns)
