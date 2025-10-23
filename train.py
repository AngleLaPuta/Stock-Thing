import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import random
import yfinance as yf
from datetime import time, timedelta, datetime
from random import choice
import ast
import os
from main import amazingprogram


# --- CONFIGURATION & HYPERPARAMETERS ---
TICKER_SYMBOL_TRAINED = 'SPXUSD'
EPISODES = 500000
PLOT_DAYS = 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

ACTION_MAP = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
NUM_ACTIONS = len(ACTION_MAP)
INITIAL_CAPITAL = 1000.0
# ---------------------------------------

# --- Setup Directory for Saving Assets ---
SAVE_DIR = 'qlearning_results'
os.makedirs(SAVE_DIR, exist_ok=True)


# --- Base Strategy Class for Plotting (MODIFIED) ---
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

        # Reset history for the day
        self.buy_dates = []
        self.sell_dates_signal = []

        # Trades contain (price, time, action)
        buys = [(p, t) for p, t, a in trades if a == 1]
        sells = [(p, t) for p, t, a in trades if a == 2]

        if not buys or not sells:
            # If no complete trade, portfolio value remains initial capital
            self.portfolio_dates = df_day.index.tolist()
            self.portfolio_value_history = [self.initial_capital] * len(df_day.index)
            return

        # --- Trade Execution Details ---
        buy_price, buy_time = buys[0]
        sell_price, sell_time = sells[-1]
        self.buy_dates = [buy_time]
        self.sell_dates_signal = [sell_time]

        # Calculate shares purchased
        shares = self.initial_capital / buy_price

        current_capital = self.initial_capital
        portfolio_history = {}  # Use dict to store timestamp: value

        # --- Continuous Portfolio Value Calculation ---

        # 1. Before Buy: Capital is flat
        portfolio_history[df_day.index.min()] = self.initial_capital

        holding = False

        for idx, row in df_day.iterrows():
            close_price = row[TICKER_SYMBOL_TRAINED, 'Close']

            # Action event: BUY
            if not holding and idx >= buy_time:
                # Value at the moment of buy remains initial capital
                portfolio_history[buy_time] = self.initial_capital
                holding = True

            # Action event: SELL
            if holding and idx >= sell_time:
                # Value at the moment of sell (shares * sell price)
                final_value = shares * sell_price
                portfolio_history[sell_time] = final_value
                holding = False
                current_capital = final_value  # Update capital for post-trade history

            # Continuous Holding: Calculate value while position is open
            if holding:
                # Value = shares * current price
                portfolio_history[idx] = shares * close_price

            # After Sell: Capital is flat at the final value
            elif idx > sell_time:
                portfolio_history[idx] = current_capital

            # Before Buy: Capital is flat at initial value
            elif idx < buy_time:
                portfolio_history[idx] = self.initial_capital

        # --- Finalize History ---
        self.portfolio_dates = list(portfolio_history.keys())
        self.portfolio_value_history = list(portfolio_history.values())

        # Reindex to the full minute-by-minute index of the day
        history_series = pd.Series(self.portfolio_value_history, index=self.portfolio_dates)

        # Fill in missing timestamps using forward fill to get the minute-by-minute line
        history_series = history_series.reindex(df_day.index, method='ffill').dropna()

        self.portfolio_dates = history_series.index.tolist()
        self.portfolio_value_history = history_series.values.tolist()


# --- 1. INDICATOR FUNCTIONS ---
def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    return df['Close'].rolling(window=period, min_periods=1).mean()


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df['Close'].ewm(span=period, adjust=False, min_periods=1).mean()


def calculate_ema_200(df: pd.DataFrame) -> pd.Series:
    return calculate_ema(df, period=200)


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    exp1 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=1).mean()
    macd_histogram = macd_line - signal_line
    return pd.DataFrame({'MACD': macd_line, 'Signal_Line': signal_line, 'MACD_Histogram': macd_histogram})


def calculate_sma_slope(df: pd.DataFrame, sma_period: int) -> pd.Series:
    if f'SMA_{sma_period}' not in df.columns: df[f'SMA_{sma_period}'] = calculate_sma(df, period=sma_period)
    return df[f'SMA_{sma_period}'].diff() / df[f'SMA_{sma_period}'].shift(1) * 100


# --- 2. DATA PREPARATION FUNCTIONS ---
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
    temp_df['SMA_Slope_1m'] = calculate_sma_slope(temp_df, sma_period=50)
    del temp_df['Close']
    initial_rows = len(temp_df)
    temp_df.dropna(inplace=True)
    rows_retained = len(temp_df)
    if rows_retained == 0:
        print(
            f"FATAL WARNING: All {initial_rows} data rows were dropped after indicator calculation (likely due to EMA_200 lookback).")
        return pd.DataFrame()
    final_df = df.loc[temp_df.index]
    indicator_metrics = temp_df.columns.tolist()
    indicator_tickers = [ticker] * len(indicator_metrics)
    temp_df.columns = pd.MultiIndex.from_arrays([indicator_tickers, indicator_metrics], names=['Ticker', 'Metric'])
    final_df = final_df.join(temp_df, how='inner')
    return final_df.sort_index(axis=1)


# --- 3. Q-LEARNING CORE FUNCTIONS ---
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
    # Correctly unpacks (price, time, action)
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
    return max(-1.0, min(1.0, normalized_reward))


# --- PLOTTING UTILITIES ---

def get_portfolio_series(strategy, plot_index):
    """
    Creates a time series of portfolio value for plotting, ensuring monotonic index.
    """
    if not strategy.portfolio_dates or len(strategy.portfolio_dates) == 1:
        return pd.Series(strategy.initial_capital, index=plot_index)

    history_df = pd.DataFrame({'Value': strategy.portfolio_value_history}, index=strategy.portfolio_dates)
    history_df = history_df[~history_df.index.duplicated(keep='last')]

    # FIX: Ensure index is sorted before reindexing/filling (prevents ValueError)
    history_df.sort_index(inplace=True)

    series = history_df['Value'].reindex(plot_index, method='ffill')
    return series.fillna(method='ffill')


def plot_comparison_results(analysis_df: pd.DataFrame, q_agent_strategy: BaseStrategy, perfect_strategy: BaseStrategy,
                            ticker_symbol: str, target_day: pd.Timestamp, file_path: str):
    """
    Generates the full multi-panel plot comparing price, indicators, and P&L.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Slice the analysis_df to the target day and flatten columns for plotting indicators
    df_day_slice = analysis_df.loc[
                   f'{target_day.strftime("%Y-%m-%d")} 09:30:00-05:00':f'{target_day.strftime("%Y-%m-%d")} 15:59:00-05:00'].copy()
    df_day_slice.columns = df_day_slice.columns.get_level_values(1)
    plot_index = df_day_slice.index

    # Grid layout: 3 rows on the left (Price/MACD) and 1 tall column on the right (P&L)
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1, 2], hspace=0.1, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])  # Top-left (Price)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Mid-left (MACD Lines)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)  # Bottom-left (MACD Histo)
    ax4 = fig.add_subplot(gs[:, 1])  # Right (Portfolio Value Comparison)

    all_strategies = [q_agent_strategy, perfect_strategy]
    display_date = target_day.strftime('%Y-%m-%d')

    # --- Panel 1: Price Action and Signals ---
    ax1.plot(plot_index, df_day_slice['Close'], linewidth=2, color='#1f77b4', label='Close Price')
    ax1.plot(plot_index, df_day_slice['EMA_200'], linewidth=1.5, color='#ff7f0e', label='EMA 200')
    ax1.plot(plot_index, df_day_slice['SMA_55'], linewidth=1.5, color='purple', linestyle='-', alpha=0.7,
             label='SMA 55')
    ax1.plot(plot_index, df_day_slice['EMA_8'], linewidth=1.0, color='cyan', linestyle='-', alpha=0.7, label='EMA 8')

    # Plot trade signals for the Q-Agent and Perfect Trade
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

    # --- Panel 2: MACD Lines ---
    df_day_slice['MACD'].plot(ax=ax2, linewidth=2, color='#0000FF', label='MACD Line')
    df_day_slice['Signal_Line'].plot(ax=ax2, linewidth=1.5, color='#FF00FF', label='Signal Line')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('MACD Value', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # --- Panel 3: MACD Histogram ---
    bar_colors = ['green' if x >= 0 else 'red' for x in df_day_slice['MACD_Histogram']]
    # Use the index directly for plotting time series data
    ax3.bar(plot_index, df_day_slice['MACD_Histogram'],
            width=timedelta(minutes=1) * 0.9,
            color=bar_colors,
            label='MACD Histogram')

    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Histogram', fontsize=8)
    ax3.set_xlabel('Time (ET)', fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    # Format time axis for Panel 3
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax3.get_xticklabels(), rotation=0, ha='center', fontsize=9)

    # --- Panel 4: Portfolio Value Comparison (P&L Chart) ---
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

    # Format time axis for Panel 4
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax4.get_xticklabels(), rotation=0, ha='center', fontsize=9)
    ax4.set_xlim(plot_index.min(), plot_index.max())

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)


# --- Q-Table Loading Function (for continued training) ---
def load_q_table_for_training(save_dir, ticker_symbol_trained, default_q_table):
    """Loads the Q-Table from disk or returns the default if not found."""
    model_filename = os.path.join(save_dir, f'q_table_{ticker_symbol_trained}_model.ai')

    if not os.path.exists(model_filename):
        print("⚠️ No existing model found. Starting training from scratch.")
        return default_q_table, EPSILON_START

    try:
        q_table_df = pd.read_csv(model_filename, index_col=0)
        # Use ast.literal_eval to convert string index (e.g., '(1, 0, -1)') back to tuple
        q_table = {ast.literal_eval(k): v.values for k, v in q_table_df.iterrows()}
        print(f"✅ Q-Table loaded successfully for continued training (Size: {len(q_table)} states).")
        epsilon_continue = max(EPSILON_END, 0.1)
        print(f"   Continuing Epsilon at: {epsilon_continue:.4f}")
        return q_table, epsilon_continue
    except Exception as e:
        print(f"❌ ERROR: Failed to load Q-Table CSV ({e}). Starting training from scratch.")
        return default_q_table, EPSILON_START


# --- 4. DATA LOADING AND TRAINING EXECUTION ---

print("--- PHASE 1: Data Preparation for Training ---")

data = pd.DataFrame()
TICKER_SYMBOL_TRAINED = 'SPXUSD'

try:
    from pyfinancialdata import get_multi_year

    print(f"Attempting to load {TICKER_SYMBOL_TRAINED} data using pyfinancialdata (2010 - 2018)...")
    data = get_multi_year(list(range(2010,2018)), TICKER_SYMBOL_TRAINED, provider='histdata')

    if data.empty:
        print("Warning: pyfinancialdata returned an empty DataFrame for the specified range.")
    else:
        print(f"✅ Data loaded from pyfinancialdata. Initial rows: {len(data)}")

except ImportError:
    print("FATAL: 'pyfinancialdata' not found. Cannot proceed with training data loading.")
    exit()
except Exception as e:
    print(f"FATAL: Error loading data from pyfinancialdata: {e}")
    exit()


data_reformatted = reformat_local_data(data.copy(), ticker=TICKER_SYMBOL_TRAINED)
final_analysis_df = add_indicators_to_local_data(data_reformatted.copy(), ticker=TICKER_SYMBOL_TRAINED)

if final_analysis_df.empty:
    print("\nFATAL: Final processed DataFrame is empty. Review data source and indicator periods.")
    exit()

all_days = final_analysis_df.index.normalize().unique().tolist()

if len(all_days) < 2:
    print(f"FATAL: Not enough data for training (only {len(all_days)} days).")
    exit()
print(f"Processed training days: {len(all_days)}")

# Training
q_table, epsilon = load_q_table_for_training(SAVE_DIR, TICKER_SYMBOL_TRAINED, {})

reward_history = []
plotting_trades = []
print("\n" + "=" * 60)
print(f"--- PHASE 2: Starting Q-Learning Training ({EPISODES} Episodes) ---")
print("=" * 60)


for episode in range(EPISODES):
    random_day_timestamp = random.choice(all_days)
    date_str = random_day_timestamp.strftime('%Y-%m-%d')
    start_time_str = f'{date_str} 09:30:00-05:00'
    end_time_str = f'{date_str} 15:59:00-05:00'

    df_day = final_analysis_df.loc[start_time_str:end_time_str].copy()

    if df_day.empty:
        continue


    lowest_close = df_day[TICKER_SYMBOL_TRAINED, 'Close'].min()
    highest_close = df_day[TICKER_SYMBOL_TRAINED, 'Close'].max()

    # --- NEW OPTIMAL SINGLE TRANSACTION BENCHMARK LOGIC ---
    day_prices = df_day[TICKER_SYMBOL_TRAINED, 'Close']
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

    # Determine the buy/sell prices for the record
    if max_profit > 0 and pd.notna(best_buy_time) and pd.notna(best_sell_time):
        buy_price_perfect = day_prices.loc[best_buy_time]
        sell_price_perfect = day_prices.loc[best_sell_time]
    else:
        # No profitable single-transaction trade on this day.
        # We record a zero-profit trade at the start/end for plotting a flat line.
        best_buy_time = day_prices.index[0]
        best_sell_time = day_prices.index[-1]
        buy_price_perfect = day_prices.iloc[0]
        sell_price_perfect = day_prices.iloc[-1]
    # ----------------------------------------------------

    # 1. Agent's Trade Simulation
    agent_position = 0
    action_sequence = []
    states_visited = []

    for idx, row in df_day.iterrows():
        current_state = get_state(row, TICKER_SYMBOL_TRAINED, multi_index=True)
        # Q-table is updated by get_action if a new state is encountered
        action = get_action(current_state, q_table, epsilon)
        current_price = row[TICKER_SYMBOL_TRAINED, 'Close']

        states_visited.append({'state': current_state, 'action': action})

        if action == 1 and agent_position == 0:
            agent_position = 1
            action_sequence.append((current_price, idx, 1))
        elif action == 2 and agent_position == 1:
            agent_position = 0
            action_sequence.append((current_price, idx, 2))

    last_price = df_day[TICKER_SYMBOL_TRAINED, 'Close'].iloc[-1]
    last_time = df_day.index[-1]
    if agent_position == 1: action_sequence.append((last_price, last_time, 2))

    episode_reward = get_reward(lowest_close, highest_close, action_sequence)
    reward_history.append(episode_reward)

    if not states_visited:
        continue

    # Q-Table Update
    for step in states_visited:
        state = step['state']
        action = step['action']
        # The Q-Learning update rule
        q_table[state][action] = q_table[state][action] + ALPHA * (episode_reward - q_table[state][action])

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # --- SAVE MODEL AFTER EVERY EPISODE ---
    MODEL_FILENAME = os.path.join(SAVE_DIR, f'q_table_{TICKER_SYMBOL_TRAINED}_model.ai')
    q_table_df = pd.DataFrame.from_dict(q_table, orient='index', columns=[ACTION_MAP[i] for i in range(NUM_ACTIONS)])
    # Important: convert tuple index to string for saving, will be converted back on load
    q_table_df.index = q_table_df.index.map(str)
    q_table_df.to_csv(MODEL_FILENAME)

    # --- PLOT AFTER EVERY 500 EPISODES (OR EVERY EPISODE IF YOU SET % 1 TO 0) ---
    if (episode + 1) % 500 == 0:

        print(
            f"Episode {episode + 1}/{EPISODES} | Avg Reward (last 500): {np.mean(reward_history[-500:]):.4f} | Epsilon: {epsilon:.4f}")
        print(f"  --> Q-Table Model Saved to: {MODEL_FILENAME}")

        # --- GENERATE PLOT FOR THE CURRENT EPISODE'S TRADING DAY ---

        current_agent_actions = action_sequence

        # USE THE CORRECTED PERFECT ACTIONS
        current_perfect_actions = [
            (buy_price_perfect, best_buy_time, 1),
            (sell_price_perfect, best_sell_time, 2)
        ]

        current_day_df = df_day.copy()
        target_day = current_day_df.index.normalize().min()

        # Create unique filename including episode and date
        os.makedirs(os.path.join(SAVE_DIR, 'screenshots'), exist_ok=True)
        PL_FILENAME = os.path.join(SAVE_DIR, f'screenshots\\{target_day.strftime("%Y%m%d")}_PNL_Eps{episode + 1:05d}.png')

        q_agent_strategy = BaseStrategy("Q-Agent")
        perfect_strategy = BaseStrategy("Perfect Benchmark")

        q_agent_strategy.record_trade(current_day_df, current_agent_actions)
        perfect_strategy.record_trade(current_day_df, current_perfect_actions, perfect_trade_mode=True)

        plot_comparison_results(
            analysis_df=final_analysis_df.copy(),
            q_agent_strategy=q_agent_strategy,
            perfect_strategy=perfect_strategy,
            ticker_symbol=TICKER_SYMBOL_TRAINED,
            target_day=target_day,
            file_path=PL_FILENAME
        )
        print(f"✅ P&L Comparison Plot Saved to: {PL_FILENAME}")
        stock_tickers = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "CSCO",
            "JPM", "V", "MA", "XOM", "CVX", "LLY", "JNJ", "WMT", "COST", "HD",
            "NFLX", "T", "SPY", "QQQ", "DIA"
        ]
        amazingprogram(choice(stock_tickers),plot=False)

print("\nTraining Complete.")

# -----------------------------------------------------------------------------
# --- PHASE 3: SAVE MODEL & VISUALIZATIONS ---
# -----------------------------------------------------------------------------

print(f"✅ Final Q-Table Model Saved to: {MODEL_FILENAME}")
print(f"💡 Total states learned: {len(q_table)}")

# 3b. SAVE REWARD HISTORY PLOT
plt.figure(figsize=(12, 4))
if len(reward_history) > 100:
    # Use cumulative reward history (includes previous runs if model was loaded)
    plt.plot(pd.Series(reward_history).rolling(window=100).mean(), label='100-Episode Rolling Avg Reward')
else:
    plt.plot(pd.Series(reward_history), label='Episode Reward')

plt.title('Q-Learning Agent Reward History')
plt.xlabel('Episode')
plt.ylabel('Normalized Reward')
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'Reward_History.png'))
plt.close()
print(f"✅ Reward History Plot Saved to: {os.path.join(SAVE_DIR, 'Reward_History.png')}")


# -----------------------------------------------------------------------------
# --- PHASE 4: MODEL TESTING (Validation on Unseen Data) ---
# -----------------------------------------------------------------------------
stock_tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "CSCO",
    "JPM", "V", "MA", "XOM", "CVX", "LLY", "JNJ", "WMT", "COST", "HD",
    "NFLX", "T", "SPY", "QQQ", "DIA"
]
TEST_TICKER = choice(stock_tickers)

# --- Load the Q-Table for Testing ---
try:
    # Re-load the final Q-table (saved from the last episode) for testing
    q_table_df = pd.read_csv(MODEL_FILENAME, index_col=0)
    q_table_test = {ast.literal_eval(k): v.values for k, v in q_table_df.iterrows()}
    print(f"\n--- PHASE 4: Running Test on Unseen Asset ({TEST_TICKER}) ---")
    print(f"✅ Model successfully loaded for testing.")
    # Epsilon is set to 0.0 for exploitation-only in the testing phase
except FileNotFoundError:
    print(f"❌ Error: Q-Table file not found. Cannot proceed with testing.")
    exit()
except Exception as e:
    print(f"❌ Error: Failed to load Q-Table for testing: {e}")
    exit()

# 1. Setup Test Data Parameters
ticker_symbol = TEST_TICKER
start_date = (pd.Timestamp.today() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
end_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

try:
    df = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1m", auto_adjust=False)
    if df.columns.nlevels > 1: df.columns = df.columns.get_level_values(0)

    if df.empty:
        print(f"Error: Could not retrieve data for {ticker_symbol}.")
    else:
        # 2. Data Cleaning and Indicator Calculation
        df.index = df.index.tz_convert('America/New_York')
        df = df[df.index.dayofweek < 5]
        df = df[df.index.time >= time(9, 30)]
        df = df[df.index.hour < 16]
        df = df[~((df.index.hour == 16) & (df.index.minute == 0))]

        df['SMA_13'] = calculate_sma(df, period=13)
        df['SMA_21'] = calculate_sma(df, period=21)
        df['SMA_55'] = calculate_sma(df, period=55)
        df['EMA_8'] = calculate_ema(df, period=8)
        df['SMA_50'] = calculate_sma(df, period=50)
        df['EMA_200'] = calculate_ema_200(df)
        macd_results = calculate_macd(df)
        df = df.join(macd_results)
        df['SMA_Slope_1m'] = calculate_sma_slope(df, sma_period=50)

        analysis_df = df.dropna().copy()
        trading_dates = analysis_df.index.normalize().unique().sort_values()

        if len(trading_dates) < 2:
            print(f"Not enough full trading days available ({len(trading_dates)}). Cannot test.")
        else:
            test_date = trading_dates[1]
            single_day_df = analysis_df[analysis_df.index.normalize() == test_date].copy()

            if single_day_df.empty:
                print(f"Error: No data found for the test date {test_date.strftime('%Y-%m-%d')}.")
            else:
                print(f"Trading Day Selected for Testing: {test_date.strftime('%Y-%m-%d')}")

                # 3. Q-AGENT TESTING LOGIC
                agent_position = 0
                action_sequence = []
                for idx, row in single_day_df.iterrows():
                    current_state = get_state(row, ticker_symbol, multi_index=False)
                    # Use the loaded Q-table and set exploitation_only=True
                    action = get_action(current_state, q_table_test, epsilon=0.0, exploitation_only=True)
                    current_price = row['Close']

                    if action == 1 and agent_position == 0:
                        agent_position = 1
                        action_sequence.append((current_price, idx, 1))
                    elif action == 2 and agent_position == 1:
                        agent_position = 0
                        action_sequence.append((current_price, idx, 2))

                last_price = single_day_df['Close'].iloc[-1]
                if agent_position == 1: action_sequence.append((last_price, single_day_df.index[-1], 2))

                # 4. Evaluate Performance
                lowest_close = single_day_df['Close'].min()
                highest_close = single_day_df['Close'].max()
                test_reward = get_reward(lowest_close, highest_close, action_sequence)

                # --- RESULTS ---
                print("-" * 60)
                if [p for p, t, a in action_sequence if a == 1] and [p for p, t, a in action_sequence if a == 2]:
                    agent_buy = [p for p, t, a in action_sequence if a == 1][0]
                    agent_sell = [p for p, t, a in action_sequence if a == 2][-1]
                    agent_profit = agent_sell - agent_buy
                    print(f"🤖 Agent Trade: Profit ${agent_profit:.2f}")
                    print(f"⭐ **Normalized Test Reward (Against Perfect): {test_reward:.4f}**")
                else:
                    print("🤖 Agent Trade: No successful buy/sell cycle was completed.")
                    print(f"⭐ **Normalized Test Reward: {test_reward:.4f}**")

except Exception as e:
    print(f"An error occurred during testing: {e}")