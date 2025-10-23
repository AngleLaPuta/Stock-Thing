import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import time, timedelta
from random import choice, random, randint


def calculate_sma(data: pd.DataFrame, period: int = 25) -> pd.Series:
    return data['Close'].squeeze().rolling(window=period).mean()


def calculate_sma_slope(data: pd.DataFrame, sma_period: int = 25) -> pd.Series:
    sma_series = data['Close'].squeeze().rolling(window=sma_period).mean()
    return (sma_series - sma_series.shift(1))


def calculate_ema_200(data: pd.DataFrame) -> pd.Series:
    return data['Close'].squeeze().ewm(span=200, adjust=False).mean()


def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    close_series = data['Close'].squeeze()
    fast_ema = close_series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    return pd.DataFrame({
        'MACD': macd_line,
        'Signal_Line': signal_line,
        'MACD_Histogram': macd_histogram
    }, index=data.index)


def calculate_lookback_high(data: pd.DataFrame, period: int = 120) -> pd.Series:
    return data['Close'].squeeze().rolling(window=period).max()


def calculate_lookback_low(data: pd.DataFrame, period: int = 120) -> pd.Series:
    return data['Close'].squeeze().rolling(window=period).min()


def calculate_roc(data: pd.DataFrame, period: int = 5) -> pd.Series:
    close_series = data['Close'].squeeze()
    return (close_series - close_series.shift(period)) / close_series.shift(period)


class BaseStrategy:

    def __init__(self, initial_capital=1000.0, stop_loss_percent=0.01):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.stop_loss_percent = stop_loss_percent

        self.invested_value = 0.0
        self.entry_value = 0.0
        self.shares = 0.0

        self.peak_invested_value = 0.0

        self.portfolio_value_history = []
        self.portfolio_dates = []
        self.buy_dates = []
        self.sell_dates_signal = []
        self.sell_dates_stoploss = []
        self.sell_dates_eod = []

        self.trade_log = []

        self.last_close_price = None
        self.strategy_name = "Base Strategy"

        self.trade_today = False
        self.current_day = None

    def run_strategy(self, analysis_df: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement run_strategy()")

    def _update_portfolio(self, index, row):
        current_date = index.date()

        if self.current_day is None:
            self.current_day = current_date
            if not self.portfolio_dates:
                self.portfolio_value_history.append(self.initial_capital)
                self.portfolio_dates.append(index)
                self.last_close_price = row['Close']
                return self.initial_capital

        if self.last_close_price is not None and self.shares > 0:
            self.invested_value = self.shares * row['Close']

            self.peak_invested_value = max(self.peak_invested_value, self.invested_value)

        current_portfolio_value = self.cash + self.invested_value
        self.portfolio_value_history.append(current_portfolio_value)
        self.portfolio_dates.append(index)
        self.last_close_price = row['Close']
        return current_portfolio_value

    def _execute_buy(self, index, investment_amount):
        if self.trade_today:
            return

        if self.cash < 0.01 or investment_amount < 0.01:
            return

        buy_price = index['Close']
        shares_bought = investment_amount / buy_price

        self.shares += shares_bought
        self.cash -= investment_amount

        if self.entry_value > 0:
            self.entry_value += investment_amount
        else:
            self.entry_value = investment_amount

        self.invested_value = self.shares * buy_price

        self.peak_invested_value = self.invested_value

        self.buy_dates.append(index.name)
        self.trade_today = True

        self.trade_log.append({
            'Time': index.name,
            'Type': 'BUY',
            'Price': buy_price,
            'Shares': shares_bought,
            'Value': investment_amount,
            'Strategy': self.strategy_name
        })

    def _execute_sell(self, index, sell_type):
        if self.shares < 0.01:
            return

        sell_price = index['Close']

        cash_received = self.shares * sell_price

        self.trade_log.append({
            'Time': index.name,
            'Type': f'SELL ({sell_type})',
            'Price': sell_price,
            'Shares': self.shares,
            'Value': cash_received,
            'Strategy': self.strategy_name
        })

        self.cash += cash_received
        self.invested_value = 0.0
        self.entry_value = 0.0
        self.shares = 0.0

        self.peak_invested_value = 0.0

        if sell_type == "STOP-LOSS":
            self.sell_dates_stoploss.append(index.name)
        elif sell_type == "SIGNAL":
            self.sell_dates_signal.append(index.name)
        elif sell_type == "EOD":
            self.sell_dates_eod.append(index.name)

    def _check_stop_loss(self, index):
        if self.shares > 0.01:
            stop_loss_threshold = self.peak_invested_value * (1 - self.stop_loss_percent)
            if self.invested_value < stop_loss_threshold:
                self._execute_sell(index, "STOP-LOSS")
                return True
        return False


class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_name = "EMA Crossover (100% Cash)"
        self.last_above_ema = False

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            self._update_portfolio(index, row)

            if index.time() < time(10, 0):
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                    self.last_above_ema = False
                continue

            current_above_ema = row['Close'] > row['EMA_200']

            if self._check_stop_loss(row):
                self.last_above_ema = current_above_ema
                continue

            if self.shares > 0.01:
                if self.last_above_ema and not current_above_ema:
                    self._execute_sell(row, "SIGNAL")

            else:
                if (not self.trade_today and
                        not self.last_above_ema and
                        row['SMA_Slope_1m'] > 0 and
                        current_above_ema):

                    if self.cash > 0.01:
                        self._execute_buy(row, self.cash)

            self.last_above_ema = current_above_ema

        final_value = self.cash + self.invested_value
        return final_value


class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_name = "SMA Crossover (100% Cash)"
        self.last_above_sma = False

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            self._update_portfolio(index, row)

            if index.time() < time(10, 0):
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                    self.last_above_sma = False
                continue

            current_above_sma = row['Close'] > row['SMA_50']

            if self._check_stop_loss(row):
                self.last_above_sma = current_above_sma
                continue

            if self.shares > 0.01:
                if self.last_above_sma and not current_above_sma:
                    self._execute_sell(row, "SIGNAL")

            else:
                if (not self.trade_today and
                        not self.last_above_sma and
                        row['SMA_Slope_1m'] > 0 and
                        current_above_sma):

                    if self.cash > 0.01:
                        self._execute_buy(row, self.cash)

            self.last_above_sma = current_above_sma

        final_value = self.cash + self.invested_value
        return final_value


class MACDStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_name = "MACD Crossover (100% Cash)"
        self.last_macd = None
        self.last_signal = None

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            self._update_portfolio(index, row)

            if index.time() < time(10, 0):
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                    self.last_macd, self.last_signal = None, None
                continue

            if self._check_stop_loss(row):
                self.last_macd, self.last_signal = None, None
                continue

            if self.last_macd is not None:
                if self.shares > 0.01:
                    is_signal_sell = self.last_macd > self.last_signal and row['MACD'] < row['Signal_Line']
                    if is_signal_sell:
                        self._execute_sell(row, "SIGNAL")

                else:
                    is_signal_buy = (not self.trade_today and
                                     self.last_macd < self.last_signal and
                                     row['MACD'] > row['Signal_Line'] and
                                     row['SMA_Slope_1m'] > 0 and
                                     row['Close'] > row['EMA_200'])

                    if is_signal_buy and self.cash > 0.01:
                        self._execute_buy(row, self.cash)

            self.last_macd = row['MACD']
            self.last_signal = row['Signal_Line']

        final_value = self.cash + self.invested_value
        return final_value


class MACDHistoDerivativeStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_name = "MACD Histo Derivative (100% Cash)"
        self.last_histo_slope = None
        self.current_histo_slope = None

    def run_strategy(self, analysis_df: pd.DataFrame):
        histo_derivative = (analysis_df['MACD_Histogram'] - analysis_df['MACD_Histogram'].shift(1)).rename(
            'Histo_Slope')

        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']) or pd.isna(histo_derivative.loc[index]):
                continue

            self._update_portfolio(index, row)
            self.current_histo_slope = histo_derivative.loc[index]

            if index.time() < time(10, 0):
                self.last_histo_slope = self.current_histo_slope
                continue

            if self.last_histo_slope is None:
                self.last_histo_slope = self.current_histo_slope
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                self.last_histo_slope = self.current_histo_slope
                continue

            if self._check_stop_loss(row):
                self.last_histo_slope = self.current_histo_slope
                continue

            is_buy_signal = (self.shares < 0.01 and
                             not self.trade_today and
                             self.current_histo_slope > 0 and
                             row['Close'] > row['EMA_200'])
            is_sell_signal = (self.shares > 0.01 and
                              self.current_histo_slope < 0)

            if is_sell_signal:
                self._execute_sell(row, "HISTO DERIV SIGNAL")

            elif is_buy_signal:
                if self.cash > 0.01:
                    self._execute_buy(row, self.cash)

            self.last_histo_slope = self.current_histo_slope

        final_value = self.cash + self.invested_value
        return final_value


class SMAPolynomialStrategy(BaseStrategy):
    def __init__(self, initial_capital=1000.0, stop_loss_percent=0.01, buy_threshold=0.01, sell_threshold=0.01):
        super().__init__(initial_capital=initial_capital, stop_loss_percent=stop_loss_percent)
        self.strategy_name = "SMA Derivative Trade"
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.last_slope = None

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']) or pd.isna(row['SMA_Slope_1m']):
                self._update_portfolio(index, row)
                self.last_slope = row['SMA_Slope_1m']
                continue

            self._update_portfolio(index, row)
            current_slope = row['SMA_Slope_1m']

            if index.time() < time(10, 0):
                self.last_slope = current_slope
                continue

            if self.last_slope is None:
                self.last_slope = current_slope
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                self.last_slope = current_slope
                continue

            if self._check_stop_loss(row):
                self.last_slope = current_slope
                continue

            is_reversal_down = (self.shares > 0.01 and
                                self.last_slope > self.sell_threshold and
                                current_slope <= self.sell_threshold)

            if is_reversal_down:
                self._execute_sell(row, "DERIVATIVE SIGNAL")

            is_reversal_up = (self.shares < 0.01 and
                              not self.trade_today and
                              self.last_slope <= self.buy_threshold and
                              row['SMA_Slope_1m'] > 0 and
                              current_slope > self.buy_threshold)

            if is_reversal_up:
                if self.cash > 0.01:
                    self._execute_buy(row, self.cash)

            self.last_slope = current_slope

        final_value = self.cash + self.invested_value
        return final_value


class AllInOneHybridStrategy(BaseStrategy):
    def __init__(self, initial_capital, stop_loss_percent):
        self.initial_capital = initial_capital
        self.stop_loss_percent = stop_loss_percent
        self.strategy_name = "All-In-One Hybrid (1/5th Split)"

        fifth_capital = initial_capital / 5.0

        self.ema_sub = EMACrossoverStrategy(initial_capital=fifth_capital, stop_loss_percent=stop_loss_percent)
        self.ema_sub.strategy_name = "Hybrid EMA Sub"
        self.sma_sub = SMACrossoverStrategy(initial_capital=fifth_capital, stop_loss_percent=stop_loss_percent)
        self.sma_sub.strategy_name = "Hybrid SMA Sub"
        self.macd_sub = MACDStrategy(initial_capital=fifth_capital, stop_loss_percent=stop_loss_percent)
        self.macd_sub.strategy_name = "Hybrid MACD Sub"
        self.macd_deriv_sub = MACDHistoDerivativeStrategy(initial_capital=fifth_capital,
                                                          stop_loss_percent=stop_loss_percent)
        self.macd_deriv_sub.strategy_name = "Hybrid MACD Deriv Sub"
        self.poly_sub = SMAPolynomialStrategy(initial_capital=fifth_capital, stop_loss_percent=stop_loss_percent)
        self.poly_sub.strategy_name = "Hybrid SMA Deriv Sub"

        self.cash = initial_capital
        self.invested_value = 0.0
        self.portfolio_dates = []
        self.portfolio_value_history = []
        self.trade_log = []
        self.buy_dates = []
        self.sell_dates_signal = []
        self.sell_dates_stoploss = []
        self.sell_dates_eod = []

        self.sub_strategies = [self.ema_sub, self.sma_sub, self.macd_sub, self.macd_deriv_sub, self.poly_sub]

    def run_strategy(self, analysis_df: pd.DataFrame):
        for sub in self.sub_strategies:
            sub.run_strategy(analysis_df.copy())

        self.trade_log = []
        self.buy_dates = []
        self.sell_dates_signal = []
        self.sell_dates_stoploss = []
        self.sell_dates_eod = []
        history = {}

        for sub in self.sub_strategies:
            self.trade_log.extend(sub.trade_log)
            self.buy_dates.extend(sub.buy_dates)
            self.sell_dates_signal.extend(sub.sell_dates_signal)
            self.sell_dates_stoploss.extend(sub.sell_dates_stoploss)
            self.sell_dates_eod.extend(sub.sell_dates_eod)
            history[sub.strategy_name] = pd.Series(sub.portfolio_value_history, index=sub.portfolio_dates)

        combined_history = sum(history.values()).sort_index()

        self.portfolio_dates = combined_history.index.tolist()
        self.portfolio_value_history = combined_history.tolist()

        final_value = sum((sub.cash + sub.invested_value) for sub in self.sub_strategies)
        self.cash = final_value

        return final_value


class TimedRandomTraderStrategy(BaseStrategy):

    def __init__(self, initial_capital=1000.0, stop_loss_percent=0.01, allocation_fraction=1 / 8):
        super().__init__(initial_capital=initial_capital, stop_loss_percent=stop_loss_percent)
        self.strategy_name = f"Random Trader ({allocation_fraction * 100:.0f}% Hourly)"
        self.allocation_fraction = allocation_fraction

        self.target_minute = None
        self._set_next_target_minute(time(10, 0))

    def _set_next_target_minute(self, current_time):
        current_hour = current_time.hour
        current_minute = current_time.minute

        if current_hour == 10 and current_minute == 0:
            self.target_minute = randint(0, 59)
            return

        if current_hour >= 10 and current_hour < 15:
            self.target_minute = randint(0, 59)
            return

        if current_hour == 15:
            self.target_minute = randint(0, 59)
            return

        self.target_minute = -1

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            current_time = index.time()
            self._update_portfolio(index, row)

            if current_time < time(10, 0):
                continue

            if current_time == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                continue

            if self._check_stop_loss(row):
                continue

            is_target_minute = (current_time.minute == self.target_minute)

            if is_target_minute:

                current_hour_block = index.hour
                if current_hour_block >= 10 and current_hour_block < 15:
                    self._set_next_target_minute(time(current_hour_block + 1, 0))
                elif current_hour_block == 15:
                    self.target_minute = -1

                if random() < 0.5:
                    if self.shares > 0.01:
                        self._execute_sell(row, "HOURLY RANDOM SELL")
                    elif not self.trade_today:
                        if self.cash > 0.01:
                            investment_amount = self.cash * self.allocation_fraction
                            self.trade_today = False
                            self._execute_buy(row, investment_amount)

        final_value = self.cash + self.invested_value
        return final_value


class RandomTraderStrategy(BaseStrategy):

    def __init__(self, initial_capital=1000.0, stop_loss_percent=0.01, trade_chance=0.01):
        super().__init__(initial_capital=initial_capital, stop_loss_percent=stop_loss_percent)
        self.strategy_name = "Random Trader (1% Chance)"
        self.trade_chance = trade_chance

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            self._update_portfolio(index, row)

            if index.time() < time(10, 0):
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                continue

            if self._check_stop_loss(row):
                continue

            if self.current_day is not None and index.date() > self.current_day:
                self.trade_today = False
                self.current_day = index.date()

            if self.shares > 0.01:
                if random() < self.trade_chance:
                    self._execute_sell(row, "RANDOM SIGNAL")

            else:
                if not self.trade_today and random() < self.trade_chance:
                    if self.cash > 0.01:
                        self._execute_buy(row, self.cash)

        final_value = self.cash + self.invested_value
        return final_value


class SwingLowSupportStrategy(BaseStrategy):

    def __init__(self, initial_capital=1000.0, stop_loss_percent=0.01, lookback_period=15):
        super().__init__(initial_capital=initial_capital, stop_loss_percent=stop_loss_percent)
        self.strategy_name = f"Swing Low Support ({lookback_period}m)"
        self.lookback_period = lookback_period
        self.price_history = []

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            self._update_portfolio(index, row)

            if index.time() < time(10, 0):
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                self.price_history = []
                continue

            if self._check_stop_loss(row):
                self.price_history = []
                continue

            if self.shares > 0.01:
                self.price_history.append(row['Low'])
                if len(self.price_history) > self.lookback_period:
                    self.price_history.pop(0)

                dynamic_support = min(self.price_history) if self.price_history else row['Low']
                print(min(self.price_history), row['Close'])
                if row['Close'] < dynamic_support:
                    self._execute_sell(row, "SWING-LOW BREAK")
                    self.price_history = []

            else:
                if (not self.trade_today and
                        row['SMA_Slope_1m'] > 0 and
                        row['Close'] > row['EMA_200']):

                    if self.cash > 0.01:
                        self._execute_buy(row, self.cash)
                        self.price_history = [row['Low']]

        final_value = self.cash + self.invested_value
        return final_value


def print_trade_summary(strategies):
    all_trades = []
    for strategy in strategies:
        all_trades.extend(strategy.trade_log)

    if not all_trades:
        print("\n--- No Trades Executed ---")
        return

    trades_df = pd.DataFrame(all_trades).sort_values(by='Time').reset_index(drop=True)

    if 'Magnitude' not in trades_df.columns:
        trades_df['Magnitude'] = None

    def clean_strategy_name(name):
        if 'Hybrid' in name:
            return name.replace(' Hybrid Sub', '').replace('Hybrid ', 'Hybrid (') + ')'
        return name

    trades_df['Strategy'] = trades_df['Strategy'].apply(clean_strategy_name)
    trades_df['Time'] = trades_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
    trades_df['Price'] = trades_df['Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
    trades_df['Value'] = trades_df['Value'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
    trades_df['Shares'] = trades_df['Shares'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    trades_df = trades_df[['Time', 'Strategy', 'Type', 'Magnitude', 'Price', 'Shares', 'Value']]
    trades_df.columns = ['Time (ET)', 'Strategy', 'Event/Trade', 'Magnitude', 'Price', 'Shares', 'Value Traded']
    trades_df = trades_df.fillna("")

    print("\n" + "=" * 115)
    print("                      📊  CHRONOLOGICAL TRADE/EVENT SUMMARY (One Entry Per Day)  📊")
    print("=" * 115)
    print(trades_df.to_string())
    print("=" * 115)


def plot_comparison_results(analysis_df: pd.DataFrame, macd_strategy: MACDStrategy, ema_strategy: EMACrossoverStrategy,
                            sma_strategy: SMACrossoverStrategy,
                            macd_deriv_strategy: MACDHistoDerivativeStrategy,
                            hybrid_strategy: AllInOneHybridStrategy, random_strategy: TimedRandomTraderStrategy,
                            poly_strategy: SMAPolynomialStrategy, one_time_random_strategy: RandomTraderStrategy,
                            swing_low_strategy: SwingLowSupportStrategy,
                            ticker_symbol: str):
    plt.style.use('seaborn-v0_8-whitegrid')

    if analysis_df.index.empty:
        print("No data to plot.")
        return

    display_date = analysis_df.index[0].strftime('%Y-%m-%d')

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 12), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 2, 1]})

    analysis_df['Close'].plot(ax=ax1, linewidth=2, color='#1f77b4', label='Close Price')
    analysis_df['EMA_200'].plot(ax=ax1, linewidth=1.5, color='#ff7f0e', label='EMA 200')
    analysis_df['SMA_50'].plot(ax=ax1, linewidth=1.5, color='#d62728', linestyle=':', alpha=0.7,
                               label='SMA 50')

    def filter_trades_by_date(trade_dates, display_date):
        return [date for date in trade_dates if pd.notna(date) and date.strftime('%Y-%m-%d') == display_date]

    hybrid_eod_dates = filter_trades_by_date(hybrid_strategy.sell_dates_eod, display_date)
    if hybrid_eod_dates:
        ax1.axvline(x=hybrid_eod_dates[0], color='darkgreen', linestyle=':', alpha=0.9, linewidth=1.5,
                    label='Hybrid Sell (EOD)')
    for date in hybrid_eod_dates[1:]:
        ax1.axvline(x=date, color='darkgreen', linestyle=':', alpha=0.9, linewidth=1.5)

    hybrid_sl_dates = filter_trades_by_date(hybrid_strategy.sell_dates_stoploss, display_date)
    if hybrid_sl_dates:
        ax1.axvline(x=hybrid_sl_dates[0], color='gold', linestyle=':', alpha=0.9, linewidth=1.5,
                    label='Hybrid Sell (SL)')
    for date in hybrid_sl_dates[1:]:
        ax1.axvline(x=date, color='gold', linestyle=':', alpha=0.9, linewidth=1.5)

    swing_buy_dates = filter_trades_by_date(swing_low_strategy.buy_dates, display_date)
    if swing_buy_dates:
        ax1.axvline(x=swing_buy_dates[0], color='green', linestyle='-', alpha=1.0, linewidth=1.0, label='Swing Buy')
    for date in swing_buy_dates[1:]:
        ax1.axvline(x=date, color='green', linestyle='-', alpha=1.0, linewidth=1.0)

    swing_sell_dates = (
            swing_low_strategy.sell_dates_signal + swing_low_strategy.sell_dates_stoploss + swing_low_strategy.sell_dates_eod)
    swing_sell_dates = filter_trades_by_date(swing_sell_dates, display_date)

    if swing_sell_dates:
        ax1.axvline(x=swing_sell_dates[0], color='#FF6347', linestyle='--', alpha=0.9, linewidth=1.0,
                    label='Swing Sell (Exit)')
    for date in swing_sell_dates[1:]:
        ax1.axvline(x=date, color='#FF6347', linestyle='--', alpha=0.9, linewidth=1.0)

    ax1.set_title(f'{ticker_symbol} Price Action and Trade Signals on {display_date} (9:30 AM - 4:00 PM ET)',
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left', ncol=5, fontsize=9)
    ax1.tick_params(axis='x', which='both', bottom=False)

    def get_portfolio_series(strategy, plot_index):
        if not strategy.portfolio_dates:
            if not plot_index.empty:
                return pd.Series(strategy.initial_capital, index=[plot_index.min()]).reindex(plot_index, method='ffill')
            return pd.Series(dtype=float)

        history_df = pd.DataFrame({'Value': strategy.portfolio_value_history}, index=strategy.portfolio_dates)
        history_df = history_df[~history_df.index.duplicated(keep='last')]
        return history_df['Value'].reindex(plot_index, method='ffill')

    macd_series = get_portfolio_series(macd_strategy, analysis_df.index)
    ema_series = get_portfolio_series(ema_strategy, analysis_df.index)
    sma_series = get_portfolio_series(sma_strategy, analysis_df.index)
    macd_deriv_series = get_portfolio_series(macd_deriv_strategy, analysis_df.index)
    hybrid_series = get_portfolio_series(hybrid_strategy, analysis_df.index)
    poly_series = get_portfolio_series(poly_strategy, analysis_df.index)
    one_time_random_series = get_portfolio_series(one_time_random_strategy, analysis_df.index)
    timed_random_series = get_portfolio_series(random_strategy, analysis_df.index)
    swing_low_series = get_portfolio_series(swing_low_strategy, analysis_df.index)

    if not macd_series.empty:
        macd_series.plot(ax=ax2, linewidth=1.5, color='#4c72b0', label=macd_strategy.strategy_name)
    if not ema_series.empty:
        ema_series.plot(ax=ax2, linewidth=1.5, color='#9370DB', linestyle='-', label=ema_strategy.strategy_name)
    if not sma_series.empty:
        sma_series.plot(ax=ax2, linewidth=1.5, color='#008080', linestyle='-', label=sma_strategy.strategy_name)
    if not macd_deriv_series.empty:
        macd_deriv_series.plot(ax=ax2, linewidth=1.5, color='#FF4500', linestyle='-',
                               label=macd_deriv_strategy.strategy_name)
    if not poly_series.empty:
        poly_series.plot(ax=ax2, linewidth=1.5, color='#FFA500', linestyle='-', label=poly_strategy.strategy_name)
    if not hybrid_series.empty:
        hybrid_series.plot(ax=ax2, linewidth=2.5, color='#008000', linestyle='-', label=hybrid_strategy.strategy_name)
    if not one_time_random_series.empty:
        one_time_random_series.plot(ax=ax2, linewidth=1, color='#800000', linestyle='-',
                                    label=one_time_random_strategy.strategy_name)
    if not timed_random_series.empty:
        timed_random_series.plot(ax=ax2, linewidth=1.5, color='#8B4513', linestyle='-',
                                 label=random_strategy.strategy_name)
    if not swing_low_series.empty:
        swing_low_series.plot(ax=ax2, linewidth=1.5, color='#00FFFF', linestyle='-',
                              label=swing_low_strategy.strategy_name)

    ax2.axhline(y=macd_strategy.initial_capital, color='black', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Initial Capital (${macd_strategy.initial_capital:.0f})')

    ax2.set_title('Portfolio Value Comparison Over Time (One Day)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Value (USD)', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='best', fontsize=9)
    ax2.tick_params(axis='x', which='both', bottom=False)

    bar_colors = ['green' if x >= 0 else 'red' for x in analysis_df['MACD_Histogram']]

    ax3.bar(analysis_df.index, analysis_df['MACD_Histogram'],
            width=timedelta(minutes=1) * 0.9,
            color=bar_colors,
            label='MACD Histogram')

    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    ax3.set_title('MACD Indicator (12, 26, 9)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MACD Value', fontsize=10)
    ax3.set_xlabel('Time (ET)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.set_xlim(analysis_df.index.min(), analysis_df.index.max())

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def amazingprogram(ticker):
    ticker_symbol = ticker

    start_date = (pd.Timestamp.today() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"--- Running Strategies for {ticker_symbol} with data from {start_date} to {end_date} ---")

    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1m", auto_adjust=False)

        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"Error: Could not retrieve data for {ticker_symbol}.")
            return

        df.index = df.index.tz_convert('America/New_York')
        df = df[df.index.dayofweek < 5]
        df = df[df.index.time >= time(9, 30)]
        df = df[df.index.hour < 16]
        df = df[~((df.index.hour == 16) & (df.index.minute == 0))]

        df['SMA_50'] = calculate_sma(df, period=50)
        df['EMA_200'] = calculate_ema_200(df)
        macd_results = calculate_macd(df)
        df = df.join(macd_results)
        df['SMA_Slope_1m'] = calculate_sma_slope(df, sma_period=50)

        analysis_df = df.dropna()

        if analysis_df.empty:
            print(
                "Error: DataFrame is empty after calculating indicators and dropping NaNs. Need more historical data.")
            return

        trading_dates = analysis_df.index.normalize().unique().sort_values()

        if len(trading_dates) < 2:
            print(
                f"Not enough trading days available ({len(trading_dates)}) to skip the first and trade/plot the last. Need at least 2.")
            return

        test_date = trading_dates[1]
        single_day_df = analysis_df[analysis_df.index.normalize() == test_date].copy()

        if single_day_df.empty:
            print(f"Error: No data found for the test date {test_date.strftime('%Y-%m-%d')}.")
            return

        initial_capital = 1000.0
        stop_loss = 0.01

        print(f"\nTrading and plotting results for: {test_date.strftime('%Y-%m-%d')} (Trading starts at 10:00 AM ET)")

        macd_strategy = MACDStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        final_macd_value = macd_strategy.run_strategy(single_day_df.copy())

        ema_strategy = EMACrossoverStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        final_ema_value = ema_strategy.run_strategy(single_day_df.copy())

        sma_strategy = SMACrossoverStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        final_sma_value = sma_strategy.run_strategy(single_day_df.copy())

        macd_deriv_strategy = MACDHistoDerivativeStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        final_macd_deriv_value = macd_deriv_strategy.run_strategy(single_day_df.copy())

        poly_strategy = SMAPolynomialStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        final_poly_value = poly_strategy.run_strategy(single_day_df.copy())

        hybrid_strategy = AllInOneHybridStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        final_hybrid_value = hybrid_strategy.run_strategy(single_day_df.copy())

        one_time_random_strategy = RandomTraderStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss,
                                                        trade_chance=0.01)
        final_one_time_random_value = one_time_random_strategy.run_strategy(single_day_df.copy())

        timed_random_strategy = TimedRandomTraderStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        final_timed_random_value = timed_random_strategy.run_strategy(single_day_df.copy())

        swing_low_strategy = SwingLowSupportStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        final_swing_low_value = swing_low_strategy.run_strategy(single_day_df.copy())

        print(f"\n--- Backtesting Summary ---")
        print(f"Initial Capital: ${initial_capital:.2f}")

        print(f"\nResults for {macd_strategy.strategy_name}:")
        macd_return = ((final_macd_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_macd_value:.2f} | Return: {macd_return:.2f}%")

        print(f"\nResults for {ema_strategy.strategy_name}:")
        ema_return = ((final_ema_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_ema_value:.2f} | Return: {ema_return:.2f}%")

        print(f"\nResults for {sma_strategy.strategy_name}:")
        sma_return = ((final_sma_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_sma_value:.2f} | Return: {sma_return:.2f}%")

        print(f"\nResults for {macd_deriv_strategy.strategy_name}:")
        macd_deriv_return = ((final_macd_deriv_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_macd_deriv_value:.2f} | Return: {macd_deriv_return:.2f}%")

        print(f"\nResults for {poly_strategy.strategy_name}:")
        poly_return = ((final_poly_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_poly_value:.2f} | Return: {poly_return:.2f}%")

        print(f"\nResults for {hybrid_strategy.strategy_name}:")
        hybrid_return = ((final_hybrid_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_hybrid_value:.2f} | Return: {hybrid_return:.2f}%")

        print(f"\nResults for {one_time_random_strategy.strategy_name}:")
        one_time_random_return = ((final_one_time_random_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_one_time_random_value:.2f} | Return: {one_time_random_return:.2f}%")

        print(f"\nResults for {timed_random_strategy.strategy_name}:")
        timed_random_return = ((final_timed_random_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_timed_random_value:.2f} | Return: {timed_random_return:.2f}%")

        print(f"\nResults for {swing_low_strategy.strategy_name}:")
        swing_return = ((final_swing_low_value - initial_capital) / initial_capital) * 100
        print(f"  Final Value: ${final_swing_low_value:.2f} | Return: {swing_return:.2f}%")

        print_trade_summary(
            [macd_strategy, ema_strategy, sma_strategy, macd_deriv_strategy, hybrid_strategy, poly_strategy,
             one_time_random_strategy,
             timed_random_strategy, swing_low_strategy])

        plot_comparison_results(single_day_df, macd_strategy, ema_strategy, sma_strategy, macd_deriv_strategy,
                                hybrid_strategy, timed_random_strategy,
                                poly_strategy, one_time_random_strategy, swing_low_strategy, ticker_symbol)

    except Exception as e:
        print(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    stock_tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "ADBE",
        "AMD", "INTC", "CSCO", "CRM", "QCOM", "TXN", "SNPS", "ADSK", "MRVL", "LRCX",
        "MU", "FTNT", "DDOG", "PYPL", "SQ", "ROKU", "UBER", "ABNB",
        "JPM", "V", "MA", "GS", "SCHW", "C", "BAC", "WFC", "XOM", "CVX", "TMO",
        "LLY", "JNJ", "ABBV", "UNH", "PFE", "AMGN", "GILD", "VRTX", "SYK", "REGN",
        "WMT", "COST", "HD", "DIS", "MCD", "SBUX", "NKE", "DPZ", "F", "GM",
        "BA", "CAT", "GE", "MMM", "UPS", "HON", "EMR", "DUK", "PBR", "CLF",
        "NFLX", "T", "CMCSA", "VZ", "ATVI", "TMUS", "PLTR",
        "KO", "PEP", "PG", "CL", "K", "MDLZ", "MNST", "CVS",
        "BRK.B", "ORCL", "PLTR", "ASML", "BABA", "TSM", "SAP", "MELI", "WEIR", "TRI",
        "CRWD", "WDAY", "AXON", "ORLY", "HOG", "FITB", "CADE", "PETS", "BBAI", "NIO",
        "SPY", "QQQ", "DIA", "IWM",
        "ATGE", "RBLX", "AZN", "BIIB", "BNTX", "CDNS", "CEG", "CTAS", "GME", "AMC",
        "BYND", "RIG", "HPE", "INTU", "LMT", "MAR", "NDAQ", "NXPI", "PENN", "RKT"
    ]
    amazingprogram(choice(stock_tickers))