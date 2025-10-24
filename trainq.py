import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import random
from pyfinancialdata import *
from datetime import time, timedelta, datetime
from random import choice
import ast
import os



TICKER_SYMBOL_TRAINED = 'SPY'
PLOT_DAYS = 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999999

ACTION_MAP = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
NUM_ACTIONS = len(ACTION_MAP)
INITIAL_CAPITAL = 1000.0
ACCURACY_START_EPISODE = 100


SAVE_DIR = 'qlearning_results'
os.makedirs(SAVE_DIR, exist_ok=True)



class BaseStrategy:
    """Class to hold results for P&L plotting."""

    def __init__(self, name, initial_capital=INITIAL_CAPITAL, is_long_only=True):
        self.strategy_name = name
        self.initial_capital = initial_capital
        self.is_long_only = is_long_only
        self.buy_dates = []
        self.sell_dates_signal = []
        self.portfolio_dates = [pd.NaT]
        self.portfolio_value_history = [initial_capital]

    def record_trade(self, df_day: pd.DataFrame, trades: list, perfect_trade_mode=False):
        """
        Calculates portfolio value for a single day based on trades,
        tracking value minute-by-minute while holding a position.
        """

        self.buy_dates = []
        self.sell_dates_signal = []

        buys = [(p, t) for p, t, a in trades if a == 1]
        sells = [(p, t) for p, t, a in trades if a == 2]

        if not buys or not sells:
            self.portfolio_dates = df_day.index.tolist()
            self.portfolio_value_history = [self.initial_capital] * len(df_day.index)
            return


        buy_price, buy_time = buys[0]
        sell_price, sell_time = sells[-1]
        self.buy_dates = [buy_time]
        self.sell_dates_signal = [sell_time]

        shares = self.initial_capital / buy_price if buy_price > 0 else 0

        current_capital = self.initial_capital
        portfolio_history = {}

        portfolio_history[df_day.index.min()] = self.initial_capital

        holding = False

        for idx, row in df_day.iterrows():
            try:

                close_price = row[TICKER_SYMBOL_TRAINED, 'Close']
            except KeyError:
                close_price = row['Close']

            if not holding and idx >= buy_time:
                portfolio_history[buy_time] = self.initial_capital
                holding = True

            if holding and idx >= sell_time:
                final_value = shares * sell_price
                portfolio_history[sell_time] = final_value
                holding = False
                current_capital = final_value

            if holding:
                portfolio_history[idx] = shares * close_price

            elif idx > sell_time:
                portfolio_history[idx] = current_capital

            elif idx < buy_time:
                portfolio_history[idx] = self.initial_capital

        self.portfolio_dates = list(portfolio_history.keys())
        self.portfolio_value_history = list(portfolio_history.values())

        history_series = pd.Series(self.portfolio_value_history, index=self.portfolio_dates)
        history_series = history_series[~history_series.index.duplicated(keep='last')]
        history_series.sort_index(inplace=True)

        history_series = history_series.reindex(df_day.index, method='ffill').dropna()

        self.portfolio_dates = history_series.index.tolist()
        self.portfolio_value_history = history_series.values.tolist()



def get_close_col(df: pd.DataFrame):
    if ('Close' in df.columns and df.columns.nlevels == 1):
        return df['Close']
    elif (TICKER_SYMBOL_TRAINED, 'Close') in df.columns:
        return df[TICKER_SYMBOL_TRAINED, 'Close']
    return df['Close']


def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    return get_close_col(df).rolling(window=period, min_periods=1).mean()


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return get_close_col(df).ewm(span=period, adjust=False, min_periods=1).mean()


def calculate_ema_200(df: pd.DataFrame) -> pd.Series:
    return calculate_ema(df, period=200)


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    close_series = get_close_col(df)
    exp1 = close_series.ewm(span=12, adjust=False, min_periods=1).mean()
    exp2 = close_series.ewm(span=26, adjust=False, min_periods=1).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=1).mean()
    macd_histogram = macd_line - signal_line
    return pd.DataFrame({'MACD': macd_line, 'Signal_Line': signal_line, 'MACD_Histogram': macd_histogram})


def calculate_sma_slope(df: pd.DataFrame, sma_period: int) -> pd.Series:
    sma_col_name = f'SMA_{sma_period}'

    if sma_col_name not in df.columns:
        temp_df = pd.DataFrame(index=df.index)
        temp_df['Close'] = get_close_col(df)
        df[sma_col_name] = calculate_sma(temp_df, sma_period)


    if df.columns.nlevels == 1 and sma_col_name in df.columns:
        sma_series = df[sma_col_name]
    elif df.columns.nlevels > 1 and (TICKER_SYMBOL_TRAINED, sma_col_name) in df.columns:
        sma_series = df[TICKER_SYMBOL_TRAINED, sma_col_name]
    else:
        return pd.Series(0.0, index=df.index)

    return sma_series.diff() / sma_series.shift(1) * 100



def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index.tz_convert('America/New_York')
    df = df[df.index.dayofweek < 5]
    start_time = time(9, 30)
    end_time = time(16, 0)
    df = df[(df.index.time >= start_time) & (df.index.time < end_time)]
    return df


def reformat_local_data(df, ticker='SPY'):
    column_mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'price': 'Adj Close'}
    df.rename(columns=column_mapping, inplace=True)
    if 'Volume' not in df.columns: df['Volume'] = 0
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    if df.index.tz is None: df.index = df.index.tz_localize('UTC', nonexistent='shift_forward',
                                                            ambiguous='NaT').dropna()
    df.index = df.index.tz_convert('America/New_York')
    df = filter_market_hours(df)
    top_index = [ticker] * len(df.columns)
    df.columns = pd.MultiIndex.from_arrays([df.columns, top_index], names=['Metric', 'Ticker'])
    return df.swaplevel(0, 1, axis=1).sort_index(axis=1)


def add_indicators_to_local_data(df: pd.DataFrame, ticker: str):
    close_col_name = (ticker, 'Close')
    temp_df = pd.DataFrame(index=df.index)
    temp_df['Close'] = df[close_col_name]

    temp_df['SMA_13'] = calculate_sma(temp_df, period=13)
    temp_df['SMA_21'] = calculate_sma(temp_df, period=21)
    temp_df['SMA_55'] = calculate_sma(temp_df, period=55)
    temp_df['EMA_8'] = calculate_ema(temp_df, period=8)
    temp_df['SMA_50'] = calculate_sma(temp_df, period=50)
    temp_df['EMA_200'] = calculate_ema_200(temp_df)
    macd_results = calculate_macd(temp_df)
    temp_df = temp_df.join(macd_results)

    temp_df['SMA_Slope_1m'] = temp_df['SMA_50'].diff() / temp_df['SMA_50'].shift(1) * 100

    del temp_df['Close']


    final_df = df.loc[temp_df.index]
    temp_df.dropna(inplace=True)

    if temp_df.empty:
        return pd.DataFrame()

    final_df = final_df.loc[temp_df.index]
    indicator_metrics = temp_df.columns.tolist()
    indicator_tickers = [ticker] * len(indicator_metrics)
    temp_df.columns = pd.MultiIndex.from_arrays([indicator_tickers, indicator_metrics], names=['Ticker', 'Metric'])
    final_df = final_df.join(temp_df, how='inner')
    return final_df.sort_index(axis=1)



def get_state(df_row: pd.Series, ticker: str, multi_index: bool) -> tuple:
    if multi_index:
        get_col = lambda name: df_row[(ticker, name)]
    else:
        get_col = lambda name: df_row[name]
    ema8 = get_col('EMA_8')
    sma55 = get_col('SMA_55')
    ema_cross_state = 0
    if ema8 > sma55 * 1.0005:
        ema_cross_state = 1
    elif ema8 < sma55 * 0.9995:
        ema_cross_state = -1
    macd_histo = get_col('MACD_Histogram')
    macd_state = 0
    if macd_histo > 0.005:
        macd_state = 1
    elif macd_histo < -0.005:
        macd_state = -1
    sma_slope = get_col('SMA_Slope_1m')
    slope_state = 0
    if sma_slope > 0.005:
        slope_state = 1
    elif sma_slope < -0.005:
        slope_state = -1
    return (ema_cross_state, macd_state, slope_state)


def get_action(state: tuple, q_table: dict, epsilon: float, exploitation_only: bool = False) -> int:
    if state not in q_table: q_table[state] = np.zeros(NUM_ACTIONS)
    if not exploitation_only and random.random() < epsilon:
        return random.choice(list(ACTION_MAP.keys()))
    else:
        return np.argmax(q_table[state])


def get_reward(lowest_close, highest_close, action_sequence) -> float:
    perfect_profit = highest_close - lowest_close
    agent_profit = 0.0

    buy_actions = [p for p, t, a in action_sequence if a == 1]
    sell_actions = [p for p, t, a in action_sequence if a == 2]

    if buy_actions and sell_actions:
        agent_buy_price = buy_actions[0]
        agent_sell_price = sell_actions[-1]
        agent_profit = agent_sell_price - agent_buy_price

    if perfect_profit <= 0:
        normalized_reward = agent_profit
    else:
        normalized_reward = agent_profit / perfect_profit

    if agent_profit < 0:
        divisor = perfect_profit if perfect_profit > 0.01 else 1.0
        normalized_loss = abs(agent_profit / divisor)
        penalty = normalized_loss ** 3
        normalized_reward = normalized_reward - penalty

    return max(-5.0, min(1.0, normalized_reward))


def run_validation_episode(df_day: pd.DataFrame, q_table: dict, ticker: str) -> float:
    """Runs a single episode using the current Q-table in exploitation-only mode for validation."""

    day_prices = df_day[(ticker, 'Close')].copy()
    max_profit = 0.0
    min_price = float('inf')
    best_buy_time = pd.NaT
    best_sell_time = pd.NaT
    current_buy_time = pd.NaT

    for idx, price in day_prices.items():
        if price < min_price:
            min_price = price
            current_buy_time = idx

        current_profit = price - min_price

        if current_profit > max_profit:
            max_profit = current_profit
            best_buy_time = current_buy_time
            best_sell_time = idx

    if max_profit > 0 and pd.notna(best_buy_time) and pd.notna(best_sell_time):
        perfect_profit = day_prices.loc[best_sell_time] - day_prices.loc[best_buy_time]
    else:
        perfect_profit = 0.0

    agent_position = 0
    action_sequence = []

    for idx, row in df_day.iterrows():
        current_state = get_state(row, ticker, multi_index=True)
        action = get_action(current_state, q_table, epsilon=0.0, exploitation_only=True)
        current_price = row[ticker, 'Close']

        if action == 1 and agent_position == 0:
            agent_position = 1
            action_sequence.append((current_price, idx, 1))
        elif action == 2 and agent_position == 1:
            agent_position = 0
            action_sequence.append((current_price, idx, 2))

    last_price = df_day[ticker, 'Close'].iloc[-1]
    last_time = df_day.index[-1]
    if agent_position == 1: action_sequence.append((last_price, last_time, 2))

    agent_profit = 0.0
    buy_actions = [p for p, t, a in action_sequence if a == 1]
    sell_actions = [p for p, t, a in action_sequence if a == 2]

    if buy_actions and sell_actions:
        agent_buy_price = buy_actions[0]
        agent_sell_price = sell_actions[-1]
        agent_profit = agent_sell_price - agent_buy_price

    if perfect_profit > 0:
        accuracy = min(1.0, max(0.0, agent_profit / perfect_profit))
    else:
        accuracy = 1.0 if agent_profit >= 0 else 0.0

    return accuracy



def get_portfolio_series(strategy, plot_index):
    if not strategy.portfolio_dates or len(strategy.portfolio_dates) == 1:
        return pd.Series(strategy.initial_capital, index=plot_index)

    history_df = pd.DataFrame({'Value': strategy.portfolio_value_history}, index=strategy.portfolio_dates)
    history_df = history_df[~history_df.index.duplicated(keep='last')]
    history_df.sort_index(inplace=True)

    series = history_df['Value'].reindex(plot_index, method='ffill')
    return series.fillna(method='ffill')


def plot_comparison_results(analysis_df: pd.DataFrame, q_agent_strategy: BaseStrategy, perfect_strategy: BaseStrategy,
                            ticker_symbol: str, target_day: pd.Timestamp, file_path: str):
    plt.style.use('seaborn-v0_8-whitegrid')

    df_day_slice = analysis_df.loc[
                   f'{target_day.strftime("%Y-%m-%d")} 09:30:00-05:00':f'{target_day.strftime("%Y-%m-%d")} 15:59:00-05:00'].copy()

    if df_day_slice.columns.nlevels > 1:
        df_day_slice.columns = df_day_slice.columns.get_level_values(1)

    plot_index = df_day_slice.index

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1, 2], hspace=0.1, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[:, 1])

    all_strategies = [q_agent_strategy, perfect_strategy]
    display_date = target_day.strftime('%Y-%m-%d')


    ax1.plot(plot_index, df_day_slice['Close'], linewidth=2, color='#1f77b4', label='Close Price')
    ax1.plot(plot_index, df_day_slice['EMA_200'], linewidth=1.5, color='#ff7f0e', label='EMA 200')
    ax1.plot(plot_index, df_day_slice['SMA_55'], linewidth=1.5, color='purple', linestyle='-', alpha=0.7,
             label='SMA 55')
    ax1.plot(plot_index, df_day_slice['EMA_8'], linewidth=1.0, color='cyan', linestyle='-', alpha=0.7, label='EMA 8')

    for strategy in all_strategies:
        is_perfect = (strategy.strategy_name == "Perfect Benchmark")
        color = 'red' if is_perfect else 'green'
        marker = '*' if is_perfect else '^'
        s = 100 if is_perfect else 50

        buy_dates = [date for date in strategy.buy_dates if
                     pd.notna(date) and date.strftime('%Y-%m-%d') == display_date]
        sell_dates = [date for date in strategy.sell_dates_signal if
                      pd.notna(date) and date.strftime('%Y-%m-%d') == display_date]

        if buy_dates:
            ax1.scatter(buy_dates, df_day_slice.loc[buy_dates]['Close'].values, marker=marker, color=color, s=s,
                        zorder=5,
                        label=f'{strategy.strategy_name} Buy')

        if sell_dates:
            ax1.scatter(sell_dates, df_day_slice.loc[sell_dates]['Close'].values, marker='v', color=color, s=s,
                        zorder=5,
                        label=f'{strategy.strategy_name} Sell')

    ax1.set_title(f'{ticker_symbol} Price Action & Trade Signals on {display_date}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)


    df_day_slice['MACD'].plot(ax=ax2, linewidth=2, color='#0000FF', label='MACD Line')
    df_day_slice['Signal_Line'].plot(ax=ax2, linewidth=1.5, color='#FF00FF', label='Signal Line')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('MACD Value', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax2.get_xticklabels(), visible=False)


    bar_colors = ['green' if x >= 0 else 'red' for x in df_day_slice['MACD_Histogram']]
    ax3.bar(plot_index, df_day_slice['MACD_Histogram'],
            width=timedelta(minutes=1) * 0.9,
            color=bar_colors,
            label='MACD Histogram')

    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Histogram', fontsize=8)
    ax3.set_xlabel('Time (ET)', fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax3.get_xticklabels(), rotation=0, ha='center', fontsize=9)


    q_series = get_portfolio_series(q_agent_strategy, plot_index)
    q_series.plot(ax=ax4, linewidth=2, color='green', linestyle='-',
                  label=f'Q-Agent (Final Value: ${q_series.iloc[-1]:.2f})')

    perfect_series = get_portfolio_series(perfect_strategy, plot_index)
    perfect_series.plot(ax=ax4, linewidth=3, color='red', linestyle='--',
                        label=f'Perfect Benchmark (Max Value: ${perfect_series.iloc[-1]:.2f})')

    ax4.axhline(y=INITIAL_CAPITAL, color='black', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Initial Capital (${INITIAL_CAPITAL:.0f})')

    ax4.set_title(f'Portfolio Value Comparison Over Time (One Day)', fontsize=16, fontweight='bold')
    ax4.set_ylabel('Portfolio Value (USD)', fontsize=12)
    ax4.set_xlabel('Time (ET)', fontsize=12)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    ax4.legend(loc='upper left', fontsize=10)

    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax4.get_xticklabels(), rotation=0, ha='center', fontsize=9)
    ax4.set_xlim(plot_index.min(), plot_index.max())

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)


def plot_accuracy_trend(accuracy_history_df: pd.DataFrame, file_path: str):
    plt.figure(figsize=(12, 6))

    if accuracy_history_df.empty:
        print("Warning: Accuracy history is empty, skipping accuracy trend plot.")
        return


    plot_window = 50
    train_rolling_mean = accuracy_history_df['Training_Accuracy'].rolling(window=plot_window, min_periods=1).mean()
    val_rolling_mean = accuracy_history_df['Validation_Accuracy'].rolling(window=plot_window, min_periods=1).mean()

    plt.plot(accuracy_history_df.index, train_rolling_mean,
             linewidth=2, color='blue', label=f'{plot_window}-Episode Rolling Avg (Train)')

    plt.plot(accuracy_history_df.index, val_rolling_mean,
             linewidth=2, color='red', label=f'{plot_window}-Episode Rolling Avg (Validation)')

    plt.title(f'Q-Learning Agent Accuracy Trend', fontsize=14)
    plt.xlabel(f'Episode', fontsize=12)
    plt.ylabel('Accuracy (Normalized [0.0 - 1.0])', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()



def load_q_table_for_training(save_dir, ticker_symbol_trained, default_q_table):
    """Loads Q-table and accuracy history to get the next starting point."""
    model_filename = os.path.join(save_dir, f'q_table_{ticker_symbol_trained}_model.ai')
    acc_filename = os.path.join(save_dir, f'accuracy_history.csv')

    accuracy_history = pd.DataFrame(columns=['Episode', 'Training_Accuracy', 'Validation_Accuracy'])
    start_episode = 0
    if os.path.exists(acc_filename):
        try:

            accuracy_history = pd.read_csv(acc_filename, index_col='Episode')
            accuracy_history.index = accuracy_history.index.astype(int)
            if not accuracy_history.empty:

                start_episode = accuracy_history.index.max() + 1

        except Exception as e:
            print(f"❌ WARNING: Failed to load accuracy history CSV ({e}). Starting history from scratch.")

    if not os.path.exists(model_filename):

        return default_q_table, EPSILON_START, accuracy_history, start_episode

    try:
        q_table_df = pd.read_csv(model_filename, index_col=0)
        q_table = {ast.literal_eval(k): v.values for k, v in q_table_df.iterrows()}



        epsilon_continue = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** start_episode))


        return q_table, epsilon_continue, accuracy_history, start_episode
    except Exception as e:
        print(f"❌ ERROR: Failed to load Q-Table CSV ({e}). Starting training from scratch.")
        return default_q_table, EPSILON_START, pd.DataFrame(
            columns=['Episode', 'Training_Accuracy', 'Validation_Accuracy']), 0






def train_single_episode(
        q_table: dict,
        epsilon: float,
        accuracy_history_df: pd.DataFrame,
        current_episode: int,
        final_analysis_df: pd.DataFrame,
        all_days: list,
        ticker: str = TICKER_SYMBOL_TRAINED,
        save_dir: str = SAVE_DIR
) -> tuple[dict, float, pd.DataFrame]:
    """
    Runs a single Q-Learning training episode, updates the Q-table,
    logs accuracy, and saves assets. This is the new standalone function.

    Args:
        q_table: The current Q-table dictionary.
        epsilon: The current exploration rate.
        accuracy_history_df: The DataFrame of past accuracy results.
        current_episode: The number of the episode being run (0-indexed).
        final_analysis_df: The complete pre-processed data with indicators.
        all_days: A list of unique trading days in final_analysis_df.
        ticker: The stock ticker symbol being trained.
        save_dir: The directory to save the model and plots.

    Returns:
        A tuple containing (updated_q_table, next_epsilon, updated_accuracy_history_df).
    """

    episode = current_episode


    if not all_days:
        print("Error: No trading days available for training.")
        return q_table, epsilon, accuracy_history_df

    random_day_timestamp = random.choice(all_days)
    date_str = random_day_timestamp.strftime('%Y-%m-%d')
    start_time_str = f'{date_str} 09:30:00-05:00'
    end_time_str = f'{date_str} 15:59:00-05:00'

    df_day = final_analysis_df.loc[start_time_str:end_time_str].copy()

    if df_day.empty:
        print(f"Warning: Selected day {date_str} is empty. Skipping episode.")
        return q_table, epsilon, accuracy_history_df

    lowest_close = df_day[ticker, 'Close'].min()
    highest_close = df_day[ticker, 'Close'].max()

    day_prices = df_day[ticker, 'Close']
    max_profit = 0.0
    min_price = float('inf')
    best_buy_time = pd.NaT
    best_sell_time = pd.NaT
    current_buy_time = pd.NaT


    for idx, price in day_prices.items():
        if price < min_price:
            min_price = price
            current_buy_time = idx

        current_profit_val = price - min_price

        if current_profit_val > max_profit:
            max_profit = current_profit_val
            best_buy_time = current_buy_time
            best_sell_time = idx


    if max_profit > 0 and pd.notna(best_buy_time) and pd.notna(best_sell_time):
        buy_price_perfect = day_prices.loc[best_buy_time]
        sell_price_perfect = day_prices.loc[best_sell_time]
        perfect_profit_for_accuracy = sell_price_perfect - buy_price_perfect
    else:

        best_buy_time = day_prices.index[0]
        best_sell_time = day_prices.index[-1]
        buy_price_perfect = day_prices.iloc[0]
        sell_price_perfect = day_prices.iloc[-1]
        perfect_profit_for_accuracy = 0.0


    agent_position = 0
    action_sequence = []
    states_visited = []
    current_state_action = 0

    for idx, row in df_day.iterrows():
        current_state = get_state(row, ticker, multi_index=True)
        action = get_action(current_state, q_table, epsilon)
        current_price = row[ticker, 'Close']

        states_visited.append({'state': current_state, 'action': action})

        if len(action_sequence) == 0:
            current_state_action = action

        if action == 1 and agent_position == 0:
            agent_position = 1
            action_sequence.append((current_price, idx, 1))
        elif action == 2 and agent_position == 1:
            agent_position = 0
            action_sequence.append((current_price, idx, 2))

    last_price = df_day[ticker, 'Close'].iloc[-1]
    last_time = df_day.index[-1]

    if agent_position == 1: action_sequence.append((last_price, last_time, 2))


    agent_profit_for_accuracy = 0.0
    buy_actions = [p for p, t, a in action_sequence if a == 1]
    sell_actions = [p for p, t, a in action_sequence if a == 2]
    trade_summary = "NO TRADE"
    if buy_actions and sell_actions:
        agent_buy_price = buy_actions[0]
        agent_sell_price = sell_actions[-1]
        agent_profit_for_accuracy = agent_sell_price - agent_buy_price
        trade_summary = f"Profit: ${agent_profit_for_accuracy:.2f}"

    if perfect_profit_for_accuracy > 0:
        train_accuracy = min(1.0, max(0.0, agent_profit_for_accuracy / perfect_profit_for_accuracy))
    else:

        train_accuracy = 1.0 if agent_profit_for_accuracy >= 0 else 0.0


    episode_reward = get_reward(lowest_close, highest_close, action_sequence)

    if not states_visited:
        return q_table, epsilon, accuracy_history_df


    first_state = states_visited[0]['state']
    first_state_q_values = q_table.get(first_state, np.zeros(NUM_ACTIONS))
    best_q_value = np.max(first_state_q_values)
    best_q_action = ACTION_MAP[np.argmax(first_state_q_values)] if first_state_q_values.size > 0 else 'N/A'

    print(
        f"| EP {episode + 1:<5} | \u03B5 {epsilon:.8f} | Rwd {episode_reward:+.4f} | State {first_state} | Action {ACTION_MAP.get(current_state_action, 'N/A'):<4} | Best Q {best_q_value:+.4f} ({best_q_action:<4}) | {trade_summary}")



    for step in states_visited:
        state = step['state']
        action = step['action']
        if state not in q_table: q_table[state] = np.zeros(NUM_ACTIONS)

        q_table[state][action] = q_table[state][action] + ALPHA * (episode_reward - q_table[state][action])


    validation_accuracy = 0.0
    if episode + 1 >= ACCURACY_START_EPISODE:

        available_val_days = [d for d in all_days if d != random_day_timestamp]

        validation_day = random.choice(available_val_days) if available_val_days else random_day_timestamp

        val_date_str = validation_day.strftime('%Y-%m-%d')
        val_start_time_str = f'{val_date_str} 09:30:00-05:00'
        val_end_time_str = f'{val_date_str} 15:59:00-05:00'
        df_val_day = final_analysis_df.loc[val_start_time_str:val_end_time_str].copy()

        validation_accuracy = run_validation_episode(df_val_day, q_table,
                                                     ticker) if not df_val_day.empty else 0.0

        save_episode = episode + 1
        new_row = pd.DataFrame({
            'Training_Accuracy': [train_accuracy],
            'Validation_Accuracy': [validation_accuracy]
        }, index=[save_episode])
        new_row.index.name = 'Episode'


        accuracy_history_df = pd.concat([accuracy_history_df, new_row])


    MODEL_FILENAME = os.path.join(save_dir, f'q_table_{ticker}_model.ai')
    ACC_FILENAME = os.path.join(save_dir, f'accuracy_history.csv')
    ACC_PLOT_FILENAME = os.path.join(save_dir, 'Accuracy_Trend.png')

    q_table_df = pd.DataFrame.from_dict(q_table, orient='index',
                                        columns=[ACTION_MAP[i] for i in range(NUM_ACTIONS)])
    q_table_df.index = q_table_df.index.map(str)
    q_table_df.to_csv(MODEL_FILENAME)
    accuracy_history_df.to_csv(ACC_FILENAME)


    current_agent_actions = action_sequence
    current_perfect_actions = [
        (buy_price_perfect, best_buy_time, 1),
        (sell_price_perfect, best_sell_time, 2)
    ]
    current_day_df = df_day.copy()
    target_day = current_day_df.index.normalize().min()

    os.makedirs(os.path.join(save_dir, 'screenshots'), exist_ok=True)
    PL_FILENAME = os.path.join(save_dir,
                               f'screenshots\\{target_day.strftime("%Y%m%d")}_PNL_Eps{episode + 1:05d}.png')

    q_agent_strategy = BaseStrategy("Q-Agent")
    perfect_strategy = BaseStrategy("Perfect Benchmark")

    q_agent_strategy.record_trade(current_day_df, current_agent_actions)
    perfect_strategy.record_trade(current_day_df, current_perfect_actions, perfect_trade_mode=True)
    if not episode % 100:
        plot_comparison_results(
            analysis_df=final_analysis_df.copy(),
            q_agent_strategy=q_agent_strategy,
            perfect_strategy=perfect_strategy,
            ticker_symbol=ticker,
            target_day=target_day,
            file_path=PL_FILENAME
        )

    plot_accuracy_trend(accuracy_history_df, ACC_PLOT_FILENAME)


    next_epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    print("-" * 100)
    print(
        f"| QLEARNING EPISODE {episode + 1} COMPLETE | \u03B5 (next): {next_epsilon:.8f} | Train Acc: {train_accuracy:.4f} | Val Acc: {validation_accuracy:.4f}")
    print("-" * 100)


    return q_table, next_epsilon, accuracy_history_df






if __name__ == '__main__':




    print("--- PHASE 1: Data Preparation for Training ---")





    try:
        print(f"Attempting to load SPY data as a placeholder for {TICKER_SYMBOL_TRAINED}...")
        data = get_multi_year(list(range(2010, 2018)), TICKER_SYMBOL_TRAINED, provider='histdata')
        data.columns = [col.lower() for col in data.columns]

        if data.empty:
            print("Warning: Data load returned an empty DataFrame.")
            exit()
        else:
            print(f"✅ Data loaded. Initial rows: {len(data)}")

    except Exception as e:
        print(f"FATAL: Error loading data (Placeholder): {e}")
        exit()

    data_reformatted = reformat_local_data(data.copy(), ticker=TICKER_SYMBOL_TRAINED)
    final_analysis_df = add_indicators_to_local_data(data_reformatted.copy(), ticker=TICKER_SYMBOL_TRAINED)

    if final_analysis_df.empty:
        print("\nFATAL: Final processed DataFrame is empty.")
        exit()

    all_days = final_analysis_df.index.normalize().unique().tolist()

    if len(all_days) < 2:
        print(f"FATAL: Not enough data for training (only {len(all_days)} days).")
        exit()
    print(f"Processed training days: {len(all_days)}")


    q_table, epsilon, accuracy_history_df, start_episode = load_q_table_for_training(SAVE_DIR, TICKER_SYMBOL_TRAINED,
                                                                                     {})

    print("\n" + "=" * 60)
    print(f"--- Q-Learning Agent Initialized ---")
    print(f"   Next Episode to Run: {start_episode + 1}")
    print(f"   Starting Epsilon: {epsilon:.8f}")
    print(f"   Q-Table States Loaded: {len(q_table)}")
    print("=" * 60)








    print("\n--- Example: Running a single training episode ---")

    q_table, next_epsilon, accuracy_history_df = train_single_episode(
        q_table,
        epsilon,
        accuracy_history_df,
        start_episode,
        final_analysis_df,
        all_days,
        TICKER_SYMBOL_TRAINED,
        SAVE_DIR
    )

    print("--------------------------------------------------")
    print(f"Single Episode Complete.")
    print(f"Model state saved to disk for Episode {start_episode + 1}.")
    print(f"New Epsilon for next run: {next_epsilon:.8f}")
    print("--------------------------------------------------")










