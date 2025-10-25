import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from random import choice
from main import amazingprogram
from trainLSTM import *
from traindeep import *
from trainq import train_single_episode, load_q_table_for_training

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
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_LEARNING_RATE = 0.001
LSTM_INPUT_FEATURES = 12
LSTM_INPUT_FEATURES = 7
SUPERVISED_CLASSIFICATION_PERIOD = 5
MODEL_CONFIGS = [
    {'name': 'QL_SPY', 'type': 'QL', 'ticker': 'SPY', 'save_dir': 'qlearning_results'},
    {'name': 'DQN_SPY', 'type': 'DQN', 'ticker': 'SPY', 'save_dir': 'dqn_results'},
    {'name': 'LSTM_SPY', 'type': 'LSTM', 'ticker': 'SPY', 'save_dir': 'lstm_results_v2'},
    {'name': 'GRU_SPY', 'type': 'GRU', 'ticker': 'SPY', 'save_dir': 'gru_results'},
    {'name': 'PPO_SPY', 'type': 'PPO', 'ticker': 'SPY', 'save_dir': 'ppo_results'},
    {'name': 'XGB_SPY', 'type': 'SUPERVISED', 'model_type': 'XGB', 'ticker': 'SPY', 'save_dir': 'supervised_results'},
    {'name': 'RF_SPY', 'type': 'SUPERVISED', 'model_type': 'RF', 'ticker': 'SPY', 'save_dir': 'supervised_results'},
]
SESSION_EPISODES = 9999999999999


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


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index.tz_convert('America/New_York')
    df = df[df.index.dayofweek < 5]
    start_time = time(9, 30)
    end_time = time(16, 0)
    df = df[(df.index.time >= start_time) & (df.index.time < end_time)]
    return df


def reformat_local_data(df, ticker=TICKER_SYMBOL_TRAINED):
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


def plot_price_path_comparison(
        analysis_df: pd.DataFrame,
        plot_data: dict,
        ticker_symbol: str,
        file_path: str,
        predicted_path_prices: np.ndarray = None,
        predicted_path_times: pd.DatetimeIndex = None
):
    """
    Plots the actual price action and a path derived from model prediction
    (either a full path array or a polynomial approximation of key points).
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    day = analysis_df.index.normalize().min()
    date_str = day.strftime('%Y-%m-%d')
    df_day_slice = analysis_df.loc[date_str].copy()
    if isinstance(df_day_slice.columns, pd.MultiIndex):
        close_series = df_day_slice.swaplevel(0, 1, axis=1)[ticker_symbol]['Close']
    else:
        close_series = df_day_slice['Close']
    start_time = plot_data['start_time']
    end_time = plot_data['end_time']
    if predicted_path_prices is not None and predicted_path_times is not None:
        predicted_curve_times = predicted_path_times
        predicted_prices = predicted_path_prices
        path_label = 'Model Predicted Path'
    else:
        path_label = 'Predicted Path (Polynomial Approximation)'
        points = [
            (start_time, plot_data['start_price']),
            (plot_data['high_time'], plot_data['high_price']),
            (plot_data['low_time'], plot_data['low_price']),
            (end_time, plot_data['end_price'])
        ]
        points.sort(key=lambda x: x[0])
        times_numeric = np.array([mdates.date2num(t) for t, p in points])
        prices = np.array([p for t, p in points])
        predicted_curve_times = pd.date_range(start=start_time, end=end_time, freq='min', tz=day.tz)
        predicted_curve_times_numeric = np.array([mdates.date2num(t) for t in predicted_curve_times])
        if len(times_numeric) >= 4:
            p_fit = np.polyfit(times_numeric, prices, deg=3)
            p_poly = np.poly1d(p_fit)
            predicted_prices = p_poly(predicted_curve_times_numeric)
        else:
            predicted_prices = np.interp(predicted_curve_times_numeric, times_numeric, prices)
    fig, ax = plt.subplots(figsize=(14, 7))
    actual_path_series = close_series.loc[start_time:end_time]
    ax.plot(actual_path_series.index, actual_path_series.values,
            linewidth=2, color='#1f77b4', label='Actual Price Action (10:30-16:00)')
    ax.plot(predicted_curve_times, predicted_prices,
            linewidth=2, linestyle='--', color='red', label=path_label)
    ax.scatter(plot_data['high_time'], plot_data['high_price'], marker='^', color='green', s=100, zorder=5,
               label=f'Predicted High: ${plot_data["high_price"]:.2f}')
    ax.scatter(plot_data['low_time'], plot_data['low_price'], marker='v', color='orange', s=100, zorder=5,
               label=f'Predicted Low: ${plot_data["low_price"]:.2f}')
    ax.scatter(end_time, plot_data['end_price'], marker='*', color='purple', s=150, zorder=5,
               label=f'Predicted Close: ${plot_data["end_price"]:.2f}')
    ax.scatter(start_time, plot_data['start_price'], marker='o', color='black', s=80, zorder=5,
               label=f'10:30 AM Price: ${plot_data["start_price"]:.2f}')
    ax.set_title(f'{ticker_symbol} Daily Price Path Prediction on {date_str}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.set_xlabel('Time (ET)', fontsize=12)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(start_time, end_time)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)


def get_state_vector(df_row: pd.Series, ticker: str, multi_index: bool) -> np.ndarray:
    """Transforms the OHLCV data and indicators into a normalized feature vector."""
    get_col = lambda name: df_row[(ticker, name)] if multi_index else df_row[name]
    features = [
        get_col('Price_Change'), get_col('Close_vs_EMA200'), get_col('EMA8_vs_SMA55'),
        get_col('MACD'), get_col('Signal_Line'), get_col('MACD_Histogram'), get_col('MACD_vs_Signal'),
        get_col('SMA_Slope_1m'), get_col('SMA_13') / get_col('Close') - 1.0,
                                 get_col('SMA_21') / get_col('Close') - 1.0, get_col('SMA_50') / get_col('Close') - 1.0,
                                 (get_col('High') - get_col('Low')) / get_col('Close'),
    ]
    return np.array(features, dtype=np.float32)


def get_reward(lowest_close, highest_close, action_sequence) -> float:
    perfect_profit = highest_close - lowest_close
    agent_profit = 0.0
    buys = [p for p, t, a in action_sequence if a == 1]
    sells = [p for p, t, a in action_sequence if a == 2]
    if buys and sells: agent_profit = sells[-1] - buys[0]
    normalized_reward = agent_profit / perfect_profit if perfect_profit > 0 else agent_profit
    if agent_profit < 0: normalized_reward -= abs(
        agent_profit / (perfect_profit if perfect_profit > 0.01 else 1.0)) ** 3
    return max(-5.0, min(1.0, normalized_reward))


def calculate_perfect_trade(df_day: pd.DataFrame, ticker: str):
    """Calculates the single best possible buy/sell trade for the day."""
    day_prices = df_day[(ticker, 'Close')]
    max_profit, min_price, best_buy_time, best_sell_time = 0.0, float('inf'), pd.NaT, pd.NaT
    current_buy_time = pd.NaT
    for idx, price in day_prices.items():
        if price < min_price:
            min_price = price
            current_buy_time = idx
        if price - min_price > max_profit:
            max_profit = price - min_price
            best_buy_time = current_buy_time
            best_sell_time = idx
    buy_price_perfect = day_prices.loc[best_buy_time] if pd.notna(best_buy_time) else day_prices.iloc[0]
    sell_price_perfect = day_prices.loc[best_sell_time] if pd.notna(best_sell_time) else day_prices.iloc[-1]
    perfect_profit = sell_price_perfect - buy_price_perfect
    current_perfect_actions = []
    if pd.notna(best_buy_time) and pd.notna(best_sell_time) and best_buy_time != best_sell_time:
        current_perfect_actions = [(buy_price_perfect, best_buy_time, 1), (sell_price_perfect, best_sell_time, 2)]
    elif perfect_profit != 0.0:
        current_perfect_actions = [(buy_price_perfect, df_day.index[0], 1), (sell_price_perfect, df_day.index[-1], 2)]
    return current_perfect_actions, buy_price_perfect, sell_price_perfect


def run_simulation_and_plot(
        action_sequence: list, model_type: str, df_day: pd.DataFrame, final_analysis_df: pd.DataFrame,
        ticker: str, save_dir: str, current_episode: int, perfect_actions: list
):
    """
    Runs a trading simulation based on an action sequence and generates a plot.
    """
    date_str = df_day.index.normalize().min().strftime('%Y-%m-%d')
    os.makedirs(os.path.join(save_dir, 'screenshots'), exist_ok=True)
    PL_FILENAME = os.path.join(save_dir, f'screenshots\\Episode {current_episode + 1:05d}.png')
    agent_strategy = BaseStrategy(f"{model_type}-Agent")
    perfect_strategy = BaseStrategy("Perfect Benchmark")
    agent_strategy.record_trade(df_day, action_sequence)
    perfect_strategy.record_trade(df_day, perfect_actions, perfect_trade_mode=True)
    plot_comparison_results(
        analysis_df=final_analysis_df.copy(),
        q_agent_strategy=agent_strategy,
        perfect_strategy=perfect_strategy,
        ticker_symbol=ticker,
        target_day=df_day.index.normalize().min(),
        file_path=PL_FILENAME
    )
    print(f"📸 Screenshot saved for {model_type} Episode {current_episode + 1}.")


class DQN(nn.Module):
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
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        if len(self.memory) < batch_size: return None
        return random.sample(self.memory, batch_size)

    def __len__(self): return len(self.memory)

    def __getitem__(self, idx): return self.memory[idx]

    def __setitem__(self, idx, value): self.memory[idx] = value


class DQN_Agent:
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
        if not exploitation_only and random.random() < epsilon:
            return random.choice(list(ACTION_MAP.keys()))
        else:
            state = torch.tensor(state_vector, dtype=torch.float32).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update_model(self, batch_size):
        transitions = self.memory.sample(batch_size)
        if transitions is None: return 0.0
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.int64).to(self.device).unsqueeze(-1)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(np.array(done_batch, dtype=int), dtype=torch.float32).to(self.device)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(-1)
        next_state_values = self.target_net(next_state_batch).max(1)[0] * (1 - done_batch)
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters(): param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0: self.target_net.load_state_dict(
            self.policy_net.state_dict())
        return loss.item()

    def save_model(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval();
        self.target_net.eval()


def get_state_vector(df_row: pd.Series, ticker: str, multi_index: bool) -> np.ndarray:
    """Transforms the OHLCV data and indicators into a normalized feature vector."""
    get_col = lambda name: df_row[(ticker, name)] if multi_index else df_row[name]
    features = [
        get_col('Price_Change'), get_col('Close_vs_EMA200'), get_col('EMA8_vs_SMA55'),
        get_col('MACD'), get_col('Signal_Line'), get_col('MACD_Histogram'), get_col('MACD_vs_Signal'),
        get_col('SMA_Slope_1m'), get_col('SMA_13') / get_col('Close') - 1.0,
                                 get_col('SMA_21') / get_col('Close') - 1.0, get_col('SMA_50') / get_col('Close') - 1.0,
                                 (get_col('High') - get_col('Low')) / get_col('Close'),
    ]
    return np.array(features, dtype=np.float32)


def get_reward(lowest_close, highest_close, action_sequence) -> float:
    perfect_profit = highest_close - lowest_close
    agent_profit = 0.0
    buys = [p for p, t, a in action_sequence if a == 1]
    sells = [p for p, t, a in action_sequence if a == 2]
    if buys and sells: agent_profit = sells[-1] - buys[0]
    normalized_reward = agent_profit / perfect_profit if perfect_profit > 0 else agent_profit
    if agent_profit < 0: normalized_reward -= abs(
        agent_profit / (perfect_profit if perfect_profit > 0.01 else 1.0)) ** 3
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


def load_model_for_training(save_dir, agent: DQN_Agent):
    model_filename = os.path.join(save_dir, 'weights', f'dqn_policy_net_{TICKER_SYMBOL_TRAINED}.pth')
    acc_filename = os.path.join(save_dir, f'accuracy_history.csv')
    accuracy_history = pd.DataFrame(columns=['Episode', 'Training_Accuracy', 'Validation_Accuracy'])
    start_episode = 0
    epsilon_continue = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** start_episode))
    return epsilon_continue, accuracy_history, start_episode


def train_single_episode_dqn(
        agent: DQN_Agent,
        epsilon: float,
        accuracy_history_df: pd.DataFrame,
        current_episode: int,
        final_analysis_df: pd.DataFrame,
        all_days: list,
        ticker: str = TICKER_SYMBOL_TRAINED,
        save_dir: str = 'dqn_results'
) -> tuple[DQN_Agent, float, pd.DataFrame]:
    """
    Runs a single DQN training episode, updates the Agent,
    logs accuracy, and saves assets. (Modified to use helper functions)
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
    perfect_actions, buy_price_perfect, sell_price_perfect = calculate_perfect_trade(df_day, ticker)
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
    for i in range(min(len(df_day) - 1, len(agent.memory))):
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
    if not episode % 100:
        run_simulation_and_plot(
            action_sequence=action_sequence, model_type='DQN', df_day=df_day,
            final_analysis_df=final_analysis_df, ticker=ticker, save_dir=save_dir,
            current_episode=episode, perfect_actions=perfect_actions
        )
    plot_accuracy_trend(accuracy_history_df, ACC_PLOT_FILENAME)
    next_epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    return agent, next_epsilon, accuracy_history_df


class StockLSTM_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_classes=2, output_regs=4):
        super(StockLSTM_V2, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_cls = nn.Linear(hidden_dim, output_classes)
        self.fc_reg = nn.Linear(hidden_dim, output_regs)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        final_state = hn[-1]
        return self.fc_cls(final_state), self.fc_reg(final_state)


class StockGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_classes=2, output_regs=4):
        super(StockGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_cls = nn.Linear(hidden_dim, output_classes)
        self.fc_reg = nn.Linear(hidden_dim, output_regs)

    def forward(self, x):
        _, hn = self.gru(x)
        final_state = hn[-1]
        return self.fc_cls(final_state), self.fc_reg(final_state)


def load_lstm_model_state(save_dir, ticker):
    model_filename = os.path.join(save_dir, 'weights', f'lstm_predictor_{ticker}.pth')
    start_episode = 0
    return {}, start_episode


def save_lstm_model_state(model, save_dir, ticker, episode):
    model_filename = os.path.join(save_dir, 'weights', f'lstm_predictor_{ticker}.pth')
    torch.save(model.state_dict(), model_filename)


def load_gru_model_state(save_dir, ticker):
    model_filename = os.path.join(save_dir, 'weights', f'gru_predictor_{ticker}.pth')
    start_episode = 0
    return {}, start_episode


def save_gru_model_state(model, save_dir, ticker, episode):
    model_filename = os.path.join(save_dir, 'weights', f'gru_predictor_{ticker}.pth')
    torch.save(model.state_dict(), model_filename)


def train_lstm_model(model, X_train, Y_cls_train, Y_reg_train, epochs=1, learning_rate=LSTM_LEARNING_RATE):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X_train, Y_cls_train, Y_reg_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    total_loss = 0.0
    for batch_x, batch_y_cls, batch_y_reg in dataloader:
        optimizer.zero_grad()
        cls_pred, reg_pred = model(batch_x)
        loss = cls_criterion(cls_pred, batch_y_cls) + reg_criterion(reg_pred, batch_y_reg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return model, total_loss / len(dataloader)


def train_single_episode_lstm(model, X_data, Y_cls_data, Y_reg_data, current_episode, config, final_analysis_df,
                              all_days):
    updated_model, avg_loss = train_lstm_model(
        model, X_data, Y_cls_data, Y_reg_data, epochs=1, learning_rate=LSTM_LEARNING_RATE
    )
    save_lstm_model_state(updated_model, config['save_dir'], config['ticker'], current_episode + 1)
    print(f"| EP {current_episode + 1:<5} | Loss {avg_loss:.6f} | Type LSTM | Model {config['name']} | Epoch Complete.")
    if not (current_episode + 1) % 100:
        random_day_timestamp = random.choice(all_days)
        date_str = random_day_timestamp.strftime('%Y-%m-%d')
        df_day = final_analysis_df.loc[date_str].copy()
        day_prices = df_day[(config['ticker'], 'Close')]
        action_sequence = []
        if not day_prices.empty:
            action_sequence.append((day_prices.iloc[0], day_prices.index[0], 1))
            action_sequence.append((day_prices.iloc[-1], day_prices.index[-1], 2))
        perfect_actions, _, _ = calculate_perfect_trade(df_day, config['ticker'])
        run_simulation_and_plot(
            action_sequence=action_sequence, model_type='LSTM', df_day=df_day,
            final_analysis_df=final_analysis_df, ticker=config['ticker'], save_dir=config['save_dir'],
            current_episode=current_episode, perfect_actions=perfect_actions
        )
    return updated_model, avg_loss


def train_single_episode_gru(model, X_data, Y_cls_data, Y_reg_data, current_episode, config, final_analysis_df,
                             all_days):
    updated_model, avg_loss = train_lstm_model(
        model, X_data, Y_cls_data, Y_reg_data, epochs=1, learning_rate=LSTM_LEARNING_RATE
    )
    save_gru_model_state(updated_model, config['save_dir'], config['ticker'], current_episode + 1)
    print(f"| EP {current_episode + 1:<5} | Loss {avg_loss:.6f} | Type GRU | Model {config['name']} | Epoch Complete.")
    if not (current_episode + 1) % 100:
        random_day_timestamp = random.choice(all_days)
        date_str = random_day_timestamp.strftime('%Y-%m-%d')
        df_day = final_analysis_df.loc[date_str].copy()
        day_prices = df_day[(config['ticker'], 'Close')]
        action_sequence = []
        if not day_prices.empty:
            action_sequence.append((day_prices.iloc[0], day_prices.index[0], 1))
            action_sequence.append((day_prices.iloc[-1], day_prices.index[-1], 2))
        perfect_actions, _, _ = calculate_perfect_trade(df_day, config['ticker'])
        run_simulation_and_plot(
            action_sequence=action_sequence, model_type='GRU', df_day=df_day,
            final_analysis_df=final_analysis_df, ticker=config['ticker'], save_dir=config['save_dir'],
            current_episode=current_episode, perfect_actions=perfect_actions
        )
    return updated_model, avg_loss


def create_supervised_datasets(df: pd.DataFrame, ticker: str):
    """Creates features (X) and a classification target (Y_cls) for supervised models."""
    close_col_name = (ticker, 'Close')
    df_features = df.copy()
    future_price = df_features[close_col_name].shift(-SUPERVISED_CLASSIFICATION_PERIOD)
    current_price = df_features[close_col_name]
    df_features['Target_Direction'] = (future_price > current_price).astype(int)
    indicator_cols = [col for col in df_features.columns if col[0] == ticker and col[1] != 'Close']
    X = df_features[indicator_cols].dropna().values
    cleaned_indices = df_features[indicator_cols].dropna().index
    Y = df_features.loc[cleaned_indices]['Target_Direction'].values
    X = X[:-SUPERVISED_CLASSIFICATION_PERIOD]
    Y = Y[:-SUPERVISED_CLASSIFICATION_PERIOD]
    Y_cls = Y.astype(np.int64)
    return X, Y_cls


def run_supervised_training(config: dict, training_data_df: pd.DataFrame, X_data, Y_cls_data, run_once: bool):
    """
    Trains and evaluates a supervised model (XGBoost or Random Forest).
    This only needs to run once, not episode by episode.
    """
    if run_once:
        print(f"Skipping SUPERVISED model {config['name']} (already trained/run).")
        return None
    model_type = config['model_type']
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    model_filename = os.path.join(save_dir, f'{model_type}_{config["ticker"]}.joblib')
    if X_data.size == 0:
        print(f"Error: Supervised training data is empty for {model_type}.")
        return None
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_cls_data, test_size=0.2, shuffle=False)
    print(f"--- Training {model_type} on {len(X_train)} samples ---")
    if model_type == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'XGB':
        model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss',
                              n_estimators=100, random_state=42, n_jobs=-1)
    else:
        print(f"Unknown supervised model type: {model_type}")
        return None
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"✅ {model_type} Training Complete. Validation Accuracy: {accuracy:.4f}")
    joblib.dump(model, model_filename)
    return model


class PPO_ActorCritic(nn.Module):
    def __init__(self, input_size, action_size):
        super(PPO_ActorCritic, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.actor(features), self.critic(features)


class PPO_Agent:
    """Simplified PPO Agent for comparison with DQN/QL."""

    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPO_ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_action(self, state_vector) -> int:
        state = torch.tensor(state_vector, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()


def train_single_episode_ppo(
        agent: PPO_Agent,
        current_episode: int,
        final_analysis_df: pd.DataFrame,
        all_days: list,
        ticker: str,
        save_dir: str
) -> PPO_Agent:
    """
    Runs a single PPO training episode and performs a simplified single-step update.
    (Modified to include screenshot logic)
    """
    random_day_timestamp = random.choice(all_days)
    date_str = random_day_timestamp.strftime('%Y-%m-%d')
    start_time_str = f'{date_str} 09:30:00-05:00'
    end_time_str = f'{date_str} 15:59:00-05:00'
    df_day = final_analysis_df.loc[start_time_str:end_time_str].copy()
    if df_day.empty: return agent
    states, actions = [], []
    agent_position = 0
    action_sequence = []
    for idx, row in df_day.iterrows():
        state_vector = get_state_vector(row, ticker, multi_index=True)
        action_int = agent.get_action(state_vector)
        states.append(state_vector)
        actions.append(action_int)
        current_price = row[ticker, 'Close']
        if action_int == 1 and agent_position == 0:
            agent_position = 1
            action_sequence.append((current_price, idx, 1))
        elif action_int == 2 and agent_position == 1:
            agent_position = 0
            action_sequence.append((current_price, idx, 2))
    day_prices = df_day[ticker, 'Close']
    lowest_close = day_prices.min()
    highest_close = day_prices.max()
    if agent_position == 1: action_sequence.append((day_prices.iloc[-1], df_day.index[-1], 2))
    episode_reward = get_reward(lowest_close, highest_close, action_sequence)
    if not states: return agent
    agent.optimizer.zero_grad()
    first_state = torch.tensor(states[0], dtype=torch.float32).to(agent.device).unsqueeze(0)
    first_action = torch.tensor([actions[0]]).to(agent.device)
    logits, value_pred = agent.model(first_state)
    dist = torch.distributions.Categorical(logits=logits)
    log_prob = dist.log_prob(first_action)
    advantage = episode_reward - value_pred.squeeze().detach()
    policy_loss = -log_prob * advantage
    reward_tensor = torch.tensor(episode_reward, dtype=torch.float32).to(agent.device)
    value_loss = nn.MSELoss()(value_pred.squeeze(), reward_tensor)
    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    agent.optimizer.step()
    print(
        f"| EP {current_episode + 1:<5} | Rwd {episode_reward:+.4f} | Loss {loss.item():.6f} | Type PPO | Model {ticker} |")
    if not (current_episode + 1) % 100:
        perfect_actions, _, _ = calculate_perfect_trade(df_day, ticker)
        run_simulation_and_plot(
            action_sequence=action_sequence, model_type='PPO', df_day=df_day,
            final_analysis_df=final_analysis_df, ticker=ticker, save_dir=save_dir,
            current_episode=current_episode, perfect_actions=perfect_actions
        )
    ppo_model_filename = os.path.join(save_dir, 'weights', f'ppo_actor_critic_{ticker}.pth')
    torch.save(agent.model.state_dict(), ppo_model_filename)
    return agent


def filter_easy_days(
        all_days: list,
        df: pd.DataFrame,
        ticker: str,
        skip_percentage: float = 0.75
) -> list:
    """
    Analyzes each day's price path to determine the overall trend (upward/downward).
    Downward trends (Highest point earlier than Lowest point) are kept.
    Upward/Sideways trends are skipped with a probability of `skip_percentage`.
    """
    print(f"Applying downward trend preference filter (Skip Easy Days: {skip_percentage * 100:.0f}%)")
    close_col_name = (ticker, 'Close')
    if close_col_name not in df.columns:
        print(f"Warning: Close column not found for ticker {ticker}. Skipping trend analysis.")
        return all_days
    filtered_days = []
    for day_timestamp in all_days:
        date_str = day_timestamp.strftime('%Y-%m-%d')
        df_day = df.loc[date_str].copy()
        if df_day.empty:
            continue
        day_prices = df_day[close_col_name]
        high_price = day_prices.max()
        low_price = day_prices.min()
        high_time = day_prices[day_prices == high_price].index.min()
        low_time = day_prices[day_prices == low_price].index.min()
        is_downward_trend = (high_time < low_time)
        if is_downward_trend:
            filtered_days.append(day_timestamp)
        else:
            if random.random() >= skip_percentage:
                filtered_days.append(day_timestamp)
    print(f"Original days: {len(all_days)}, Filtered days (available for selection): {len(filtered_days)}")
    return filtered_days


def run_multi_model_training_session(
        model_configs: list,
        training_data_df: pd.DataFrame,
        all_training_days: list,
):
    """
    Runs SESSION_EPISODES training steps for all configured models.
    """
    X_seq, Y_cls_seq, Y_reg_seq, num_features, _, _, _, _ = prepare_lstm_data_v2(
        training_data_df, TICKER_SYMBOL_TRAINED
    )
    global LSTM_INPUT_FEATURES
    LSTM_INPUT_FEATURES = num_features
    print(f"Sequence Model Data Prepared. Input Features: {LSTM_INPUT_FEATURES}")
    X_sup, Y_sup_cls = create_supervised_datasets(training_data_df, TICKER_SYMBOL_TRAINED)
    print(f"Supervised Model Data Prepared. Samples: {len(X_sup)}, Features: {X_sup.shape[1]}")
    filtered_training_days = filter_easy_days(
        all_training_days,
        training_data_df,
        TICKER_SYMBOL_TRAINED,
        skip_percentage=0.75
    )
    if not filtered_training_days:
        print("ERROR: Filtered list of days is empty. Training cannot continue.")
        return
    model_states = {}
    print(f"--- Initializing and Loading {len(model_configs)} Agents/Models ---")
    supervised_trained = False
    for config in model_configs:
        model_name = config['name']
        save_dir = config['save_dir']
        os.makedirs(os.path.join(save_dir, 'weights'), exist_ok=True)
        config['current_episode'] = 0
        if config['type'] == 'SUPERVISED':
            model_type = config['model_type']
            model_filename = os.path.join(save_dir, f'{model_type}_{config["ticker"]}.joblib')
            if os.path.exists(model_filename):
                print(f"✅ Found existing SUPERVISED model file for {model_name}. Skipping training.")
                continue
            else:
                print(f"⚠️ No existing file found for {model_name}. Starting one-time training...")
                run_supervised_training(config, training_data_df, X_sup, Y_sup_cls, run_once=False)
                continue
        if config['type'] == 'QL':
            q_table, epsilon, history_df, start_episode = load_q_table_for_training(save_dir, config['ticker'], {})
            model_states[model_name] = {'type': 'QL', 'q_table': q_table, 'epsilon': epsilon, 'history_df': history_df,
                                        'current_episode': start_episode, 'config': config}
            print(f"✅ Loaded QL Model ({model_name}): Start Ep {start_episode + 1}, Epsilon {epsilon:.6f}")
        elif config['type'] == 'DQN':
            agent = DQN_Agent(state_size=STATE_SIZE, action_size=NUM_ACTIONS, learning_rate=LEARNING_RATE, gamma=GAMMA,
                              memory_size=REPLAY_MEMORY_SIZE, target_update_freq=TARGET_UPDATE_FREQ)
            epsilon, history_df, start_episode = load_model_for_training(save_dir, agent)
            model_states[model_name] = {'type': 'DQN', 'agent': agent, 'epsilon': epsilon, 'history_df': history_df,
                                        'current_episode': start_episode, 'config': config}
            print(f"✅ Loaded DQN Model ({model_name}): Start Ep {start_episode + 1}, Epsilon {epsilon:.6f}")
        elif config['type'] == 'LSTM':
            lstm_model = StockLSTM_V2(input_dim=LSTM_INPUT_FEATURES, hidden_dim=LSTM_HIDDEN_DIM,
                                      num_layers=LSTM_NUM_LAYERS)
            loaded_data, start_episode = load_lstm_model_state(save_dir, config['ticker'])
            model_states[model_name] = {
                'type': 'LSTM', 'model': lstm_model, 'X_data': X_seq, 'Y_cls_data': Y_cls_seq,
                'Y_reg_data': Y_reg_seq, 'current_episode': start_episode, 'config': config
            }
            print(f"✅ Initialized LSTM Model ({model_name}): Start Ep {start_episode + 1}, Epochs {start_episode}")
        elif config['type'] == 'GRU':
            gru_model = StockGRU(input_dim=LSTM_INPUT_FEATURES, hidden_dim=LSTM_HIDDEN_DIM,
                                 num_layers=LSTM_NUM_LAYERS)
            loaded_data, start_episode = load_gru_model_state(save_dir, config['ticker'])
            model_states[model_name] = {
                'type': 'GRU', 'model': gru_model, 'X_data': X_seq, 'Y_cls_data': Y_cls_seq,
                'Y_reg_data': Y_reg_seq, 'current_episode': start_episode, 'config': config
            }
            print(f"✅ Initialized GRU Model ({model_name}): Start Ep {start_episode + 1}, Epochs {start_episode}")
        elif config['type'] == 'PPO':
            agent = PPO_Agent(state_size=STATE_SIZE, action_size=NUM_ACTIONS, learning_rate=LEARNING_RATE)
            model_states[model_name] = {'type': 'PPO', 'agent': agent, 'current_episode': 0, 'config': config}
            print(f"✅ Initialized PPO Model ({model_name}): Start Ep 1")
        elif config['type'] == 'SUPERVISED' and not supervised_trained:
            run_supervised_training(config, training_data_df, X_sup, Y_sup_cls, run_once=False)
            supervised_trained = True
    print("-" * 50)
    loop_models = {name: state for name, state in model_states.items() if state['type'] != 'SUPERVISED'}
    for session_episode in range(SESSION_EPISODES):
        if not (session_episode + 1) % 100:
            print(f"\n--- SESSION EPISODE {session_episode + 1}/{SESSION_EPISODES} ---")
        for name, state in loop_models.items():
            config = state['config']
            days_to_use = all_training_days if config['type'] in ('LSTM', 'GRU') else filtered_training_days
            if config['type'] == 'QL':
                updated_q_table, next_epsilon, updated_history_df = train_single_episode(
                    q_table=state['q_table'], epsilon=state['epsilon'], accuracy_history_df=state['history_df'],
                    current_episode=state['current_episode'], final_analysis_df=training_data_df,
                    all_days=days_to_use,
                    ticker=config['ticker'], save_dir=config['save_dir']
                )
                state.update({'q_table': updated_q_table, 'epsilon': next_epsilon, 'history_df': updated_history_df})
            elif config['type'] == 'DQN':
                updated_agent, next_epsilon, updated_history_df = train_single_episode_dqn(
                    agent=state['agent'], epsilon=state['epsilon'], accuracy_history_df=state['history_df'],
                    current_episode=state['current_episode'], final_analysis_df=training_data_df,
                    all_days=days_to_use,
                    ticker=config['ticker'], save_dir=config['save_dir']
                )
                state.update({'agent': updated_agent, 'epsilon': next_epsilon, 'history_df': updated_history_df})
            elif config['type'] == 'LSTM':
                updated_model, loss = train_single_episode_lstm(
                    model=state['model'], X_data=state['X_data'], Y_cls_data=state['Y_cls_data'],
                    Y_reg_data=state['Y_reg_data'], current_episode=state['current_episode'], config=config,
                    final_analysis_df=training_data_df, all_days=days_to_use
                )
                state['model'] = updated_model
            elif config['type'] == 'GRU':
                updated_model, loss = train_single_episode_gru(
                    model=state['model'], X_data=state['X_data'], Y_cls_data=state['Y_cls_data'],
                    Y_reg_data=state['Y_reg_data'], current_episode=state['current_episode'], config=config,
                    final_analysis_df=training_data_df, all_days=days_to_use
                )
                state['model'] = updated_model
            elif config['type'] == 'PPO':
                updated_agent = train_single_episode_ppo(
                    agent=state['agent'], current_episode=state['current_episode'],
                    final_analysis_df=training_data_df, all_days=days_to_use,
                    ticker=config['ticker'], save_dir=config['save_dir']
                )
                state['agent'] = updated_agent
            state['current_episode'] += 1
        if not (session_episode + 1) % 100:
            stock_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "V", "MA", "LLY"]
            amazingprogram(choice(stock_tickers), plot=False)
    print(f"\n=================================================")
    print(f"Training Session Complete: Ran {SESSION_EPISODES} steps per model.")
    print(f"=================================================")


if __name__ == '__main__':
    for config in MODEL_CONFIGS:
        os.makedirs(os.path.join(config['save_dir'], 'weights'), exist_ok=True)
    print("\n--- PHASE 0: Data Setup ---")
    try:
        data = get_multi_year(list(range(2010, 2018)), 'SPXUSD', provider='histdata')
        data.columns = [col.lower() for col in data.columns]
        data_reformatted = reformat_local_data(data.copy(), ticker='SPY')
        final_analysis_df = add_indicators_to_local_data(data_reformatted.copy(), ticker='SPY')
        all_days = final_analysis_df.index.normalize().unique().tolist()
        if final_analysis_df.empty or len(all_days) < 2:
            raise ValueError("Data frame is empty or has insufficient days.")
    except Exception as e:
        print(f"FATAL: Data loading error (Check 'pyfinancialdata' and internet connection): {e}")
        exit()
    print(f"✅ Data prepared for {TICKER_SYMBOL_TRAINED}: {len(all_days)} trading days available.")
    print("\n--- PHASE 1: Running Continuous Multi-Model Training Session ---")
    run_multi_model_training_session(
        model_configs=MODEL_CONFIGS,
        training_data_df=final_analysis_df,
        all_training_days=all_days,
    )
