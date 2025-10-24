import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import random
from pyfinancialdata import *
from datetime import time, timedelta, datetime
from random import choice
import os
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque





TICKER_SYMBOL_TRAINED = 'SPY'
PLOT_DAYS = 100

LEARNING_RATE = 0.0001
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000


EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99995

ACTION_MAP = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
NUM_ACTIONS = len(ACTION_MAP)
INITIAL_CAPITAL = 1000.0


ACCURACY_START_EPISODE = 100
STATE_SIZE = 12



SAVE_DIR = 'dqn_results'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'weights'), exist_ok=True)



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
        self.buy_dates = []
        self.sell_dates_signal = []

        buys = [(p, t, a) for p, t, a in trades if a == 1]
        sells = [(p, t, a) for p, t, a in trades if a == 2]

        if not buys or not sells:
            self.portfolio_dates = df_day.index.tolist()
            self.portfolio_value_history = [self.initial_capital] * len(df_day.index)
            return

        buy_price, buy_time, _ = buys[0]
        sell_price, sell_time, _ = sells[-1]
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


def reformat_local_data(df, ticker='SPXUSD'):
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
    temp_df['Price_Change'] = temp_df['Close'].pct_change() * 100
    temp_df['Close_vs_EMA200'] = (temp_df['Close'] - temp_df['EMA_200']) / temp_df['EMA_200'] * 100
    temp_df['MACD_vs_Signal'] = (temp_df['MACD'] - temp_df['Signal_Line']) / temp_df['Signal_Line'].abs().replace(0,
                                                                                                                  1e-6) * 100
    temp_df['EMA8_vs_SMA55'] = (temp_df['EMA_8'] - temp_df['SMA_55']) / temp_df['SMA_55'] * 100

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




class DQN(nn.Module):
    """Deep Q-Network Model"""

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    """Experience Replay Buffer"""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Save a transition (s, a, r, s')"""
        self.memory.append(transition)

    def sample(self, batch_size):
        """Retrieve a random batch of transitions"""
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_Agent:
    """DQN Agent with Experience Replay and Target Network."""

    def __init__(self, state_size, action_size, learning_rate, gamma, memory_size, target_update_freq):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        self.criterion = nn.SmoothL1Loss()

    def get_action(self, state_vector, epsilon, exploitation_only=False) -> int:
        """Epsilon-greedy action selection."""
        if not exploitation_only and random.random() < epsilon:
            return random.choice(list(ACTION_MAP.keys()))
        else:
            state = torch.tensor(state_vector, dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update_model(self, batch_size):
        """Perform one step of optimization on the policy network."""
        transitions = self.memory.sample(batch_size)
        if transitions is None:
            return 0.0

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.int64).to(self.device).unsqueeze(-1)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(np.array(done_batch, dtype=int), dtype=torch.float32).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(-1)

        next_state_values = self.target_net(next_state_batch).max(1)[0]
        next_state_values = next_state_values * (1 - done_batch)

        expected_state_action_values = reward_batch + self.gamma * next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()


def get_state_vector(df_row: pd.Series, ticker: str, multi_index: bool) -> np.ndarray:
    """Transforms the OHLCV data and indicators into a normalized feature vector."""
    if multi_index:
        get_col = lambda name: df_row[(ticker, name)]
    else:
        get_col = lambda name: df_row[name]

    features = [
        get_col('Price_Change'),
        get_col('Close_vs_EMA200'),
        get_col('EMA8_vs_SMA55'),
        get_col('MACD'),
        get_col('Signal_Line'),
        get_col('MACD_Histogram'),
        get_col('MACD_vs_Signal'),
        get_col('SMA_Slope_1m'),
        get_col('SMA_13') / get_col('Close') - 1.0,
        get_col('SMA_21') / get_col('Close') - 1.0,
        get_col('SMA_50') / get_col('Close') - 1.0,
        (get_col('High') - get_col('Low')) / get_col('Close'),
    ]
    return np.array(features, dtype=np.float32)


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


def run_validation_episode(df_day: pd.DataFrame, agent: DQN_Agent, ticker: str) -> float:
    """Runs a single episode using the current Policy Network in exploitation-only mode."""

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
        current_state_vector = get_state_vector(row, ticker, multi_index=True)
        action = agent.get_action(current_state_vector, epsilon=0.0, exploitation_only=True)
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
                  label=f'DQN-Agent (Final Value: ${q_series.iloc[-1]:.2f})')

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

    plt.title(f'DQN Agent Accuracy Trend', fontsize=14)
    plt.xlabel(f'Episode', fontsize=12)
    plt.ylabel('Accuracy (Normalized [0.0 - 1.0])', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()



def load_model_for_training(save_dir, agent: DQN_Agent):
    model_filename = os.path.join(save_dir, 'weights', f'dqn_policy_net_{TICKER_SYMBOL_TRAINED}.pth')
    acc_filename = os.path.join(save_dir, f'accuracy_history.csv')

    accuracy_history = pd.DataFrame(columns=['Episode', 'Training_Accuracy', 'Validation_Accuracy'])
    start_episode = 0
    if os.path.exists(acc_filename):
        try:
            accuracy_history = pd.read_csv(acc_filename, index_col='Episode')
            accuracy_history.index = accuracy_history.index.astype(int)
            if not accuracy_history.empty:
                start_episode = accuracy_history.index.max() + 1
        except Exception:
            pass

    if os.path.exists(model_filename):
        try:
            agent.load_model(model_filename)
            epsilon_continue = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** start_episode))
            return epsilon_continue, accuracy_history, start_episode
        except Exception:
            pass

    return EPSILON_START, accuracy_history, start_episode






def train_single_episode_dqn(
        agent: DQN_Agent,
        epsilon: float,
        accuracy_history_df: pd.DataFrame,
        current_episode: int,
        final_analysis_df: pd.DataFrame,
        all_days: list,
        ticker: str = TICKER_SYMBOL_TRAINED,
        save_dir: str = SAVE_DIR
) -> tuple[DQN_Agent, float, pd.DataFrame]:
    """
    Runs a single DQN training episode, updates the Agent,
    logs accuracy, and saves assets. This is the new standalone function.

    Args:
        agent: The current DQN_Agent instance.
        epsilon: The current exploration rate.
        accuracy_history_df: The DataFrame of past accuracy results.
        current_episode: The number of the episode being run (0-indexed).
        final_analysis_df: The complete pre-processed data with indicators.
        all_days: A list of unique trading days in final_analysis_df.
        ticker: The stock ticker symbol being trained.
        save_dir: The directory to save the model and plots.

    Returns:
        A tuple containing (updated_DQN_Agent, next_epsilon, updated_accuracy_history_df).
    """

    episode = current_episode


    if not all_days:
        print("Error: No trading days available for training. Returning unchanged state.")
        return agent, epsilon, accuracy_history_df

    random_day_timestamp = random.choice(all_days)
    date_str = random_day_timestamp.strftime('%Y-%m-%d')
    start_time_str = f'{date_str} 09:30:00-05:00'
    end_time_str = f'{date_str} 15:59:00-05:00'

    df_day = final_analysis_df.loc[start_time_str:end_time_str].copy()

    if df_day.empty:
        print(f"Warning: Selected day {date_str} is empty. Skipping episode.")
        return agent, epsilon, accuracy_history_df

    day_prices = df_day[ticker, 'Close']
    lowest_close = day_prices.min()
    highest_close = day_prices.max()

    max_profit, min_price, best_buy_time, best_sell_time, buy_price_perfect, sell_price_perfect, perfect_profit_for_accuracy = (
        0.0, float('inf'), pd.NaT, pd.NaT, day_prices.iloc[0], day_prices.iloc[-1], 0.0
    )
    current_buy_time = pd.NaT
    for idx, price in day_prices.items():
        if price < min_price:
            min_price = price
            current_buy_time = idx
        if price - min_price > max_profit:
            max_profit = price - min_price
            best_buy_time = current_buy_time
            best_sell_time = idx

    if max_profit > 0 and pd.notna(best_buy_time) and pd.notna(best_sell_time):
        buy_price_perfect = day_prices.loc[best_buy_time]
        sell_price_perfect = day_prices.loc[best_sell_time]
        perfect_profit_for_accuracy = sell_price_perfect - buy_price_perfect


    agent_position = 0
    action_sequence = []
    current_state_vector = None
    first_state_action = 0

    for i, (idx, row) in enumerate(df_day.iterrows()):
        next_state_vector = get_state_vector(row, ticker, multi_index=True)

        action = agent.get_action(next_state_vector, epsilon)
        current_price = row[ticker, 'Close']

        if current_state_vector is None:
            current_state_vector = next_state_vector
            first_state_action = action


        if i > 0:

            agent.memory.push((current_state_vector, action, 0.0, next_state_vector, False))


        if action == 1 and agent_position == 0:
            agent_position = 1
            action_sequence.append((current_price, idx, 1))
        elif action == 2 and agent_position == 1:
            agent_position = 0
            action_sequence.append((current_price, idx, 2))

        current_state_vector = next_state_vector


    last_price = df_day[ticker, 'Close'].iloc[-1]
    if agent_position == 1: action_sequence.append((last_price, df_day.index[-1], 2))


    agent_profit_for_accuracy = 0.0
    buy_actions = [p for p, t, a in action_sequence if a == 1]
    sell_actions = [p for p, t, a in action_sequence if a == 2]
    trade_summary = "NO TRADE"
    if buy_actions and sell_actions:
        agent_buy_price = buy_actions[0]
        agent_sell_price = sell_actions[-1]
        agent_profit_for_accuracy = agent_sell_price - agent_buy_price
        trade_summary = f"Profit: ${agent_profit_for_accuracy:.2f}"

    train_accuracy = min(1.0, max(0.0,
                                  agent_profit_for_accuracy / perfect_profit_for_accuracy)) if perfect_profit_for_accuracy > 0 else (
        1.0 if agent_profit_for_accuracy >= 0 else 0.0)


    episode_reward = get_reward(lowest_close, highest_close, action_sequence)
    current_loss = 0.0




    if len(agent.memory) > 0:

        for i in range(len(df_day) - 1):
            if len(agent.memory) - 1 - i >= 0:
                idx_in_memory = len(agent.memory) - 1 - i


                s, a, r_old, s_prime, done_old = agent.memory[idx_in_memory]

                new_done = True if i == 0 else False
                new_reward = episode_reward

                agent.memory[idx_in_memory] = (s, a, new_reward, s_prime, new_done)


    if len(agent.memory) >= BATCH_SIZE:
        current_loss = agent.update_model(BATCH_SIZE)


    first_q_values = torch.zeros(NUM_ACTIONS)
    best_q_value = 0.0
    best_q_action = 'N/A'
    if current_state_vector is not None:
        state_tensor = torch.tensor(current_state_vector, dtype=torch.float32).to(agent.device).unsqueeze(0)
        with torch.no_grad():
            first_q_values = agent.policy_net(state_tensor).squeeze(0)
        best_q_value = first_q_values.max().item()
        best_q_action = ACTION_MAP[first_q_values.argmax().item()]

    print(
        f"| EP {episode + 1:<5} | \u03B5 {epsilon:.8f} | Rwd {episode_reward:+.4f} | Loss {current_loss:.6f} | Action {ACTION_MAP[first_state_action]:<4} | Best Q {best_q_value:+.4f} ({best_q_action:<4}) | {trade_summary}")


    validation_accuracy = 0.0
    if episode + 1 >= ACCURACY_START_EPISODE:

        available_val_days = [d for d in all_days if d != random_day_timestamp]
        validation_day = random.choice(available_val_days) if available_val_days else random_day_timestamp

        val_date_str = validation_day.strftime('%Y-%m-%d')
        df_val_day = final_analysis_df.loc[f'{val_date_str} 09:30:00-05:00':f'{val_date_str} 15:59:00-05:00'].copy()

        validation_accuracy = run_validation_episode(df_val_day, agent, ticker) if not df_val_day.empty else 0.0

        save_episode = episode + 1
        new_row = pd.DataFrame({
            'Training_Accuracy': [train_accuracy],
            'Validation_Accuracy': [validation_accuracy]
        }, index=[save_episode])
        new_row.index.name = 'Episode'

        accuracy_history_df = pd.concat([accuracy_history_df, new_row])


    MODEL_FILENAME = os.path.join(save_dir, 'weights', f'dqn_policy_net_{ticker}.pth')
    ACC_FILENAME = os.path.join(save_dir, f'accuracy_history.csv')
    ACC_PLOT_FILENAME = os.path.join(save_dir, 'Accuracy_Trend.png')

    agent.save_model(MODEL_FILENAME)
    accuracy_history_df.to_csv(ACC_FILENAME)


    current_day_df = df_day.copy()
    target_day = current_day_df.index.normalize().min()

    os.makedirs(os.path.join(save_dir, 'screenshots'), exist_ok=True)
    PL_FILENAME = os.path.join(save_dir,
                               f'screenshots\\{target_day.strftime("%Y%m%d")}_PNL_Eps{episode + 1:05d}.png')

    q_agent_strategy = BaseStrategy("DQN-Agent")
    perfect_strategy = BaseStrategy("Perfect Benchmark")
    current_perfect_actions = [(buy_price_perfect, best_buy_time, 1), (sell_price_perfect, best_sell_time, 2)]

    q_agent_strategy.record_trade(current_day_df, action_sequence)
    perfect_strategy.record_trade(current_day_df, current_perfect_actions, perfect_trade_mode=True)

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
        f"| DEEP LEARNING EPISODE {episode + 1} COMPLETE | \u03B5 (next): {next_epsilon:.8f} | Train Acc: {train_accuracy:.4f} | Val Acc: {validation_accuracy:.4f}")
    print("-" * 100)


    return agent, next_epsilon, accuracy_history_df






if __name__ == '__main__':








    print("\n[DEMO MODE] Initializing DQN agent with dummy data...")
    try:
        data = get_multi_year(list(range(2010, 2018)), TICKER_SYMBOL_TRAINED, provider='histdata')
        data.columns = [col.lower() for col in data.columns]
        data_reformatted = reformat_local_data(data.copy(), ticker=TICKER_SYMBOL_TRAINED)
        final_analysis_df = add_indicators_to_local_data(data_reformatted.copy(), ticker=TICKER_SYMBOL_TRAINED)
        all_days = final_analysis_df.index.normalize().unique().tolist()
    except:
        print("FATAL: Cannot run demo without internet/yfinance data.")
        exit()


    dqn_agent = DQN_Agent(
        state_size=STATE_SIZE,
        action_size=NUM_ACTIONS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        memory_size=REPLAY_MEMORY_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )


    epsilon, accuracy_history_df, start_episode = load_model_for_training(SAVE_DIR, dqn_agent)


    new_agent, new_epsilon, new_history = train_single_episode_dqn(
        dqn_agent,
        epsilon,
        accuracy_history_df,
        start_episode,
        final_analysis_df,
        all_days,
        TICKER_SYMBOL_TRAINED,
        SAVE_DIR
    )

    print("\n--- DEMO END ---")
    print("To continue training from an external file, pass the three returned variables back into the function.")