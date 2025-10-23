import pandas as pd
from datetime import time
import numpy as np
import os
import ast

# --- Utility/Indicator Functions ---

def calculate_sma(data: pd.DataFrame, period: int = 25) -> pd.Series:
    """Calculates the Simple Moving Average (SMA)."""
    return data['Close'].squeeze().rolling(window=period).mean()


def calculate_sma_slope(data: pd.DataFrame, sma_period: int = 25) -> pd.Series:
    """Calculates the one-period change (slope) of the SMA."""
    sma_series = data['Close'].squeeze().rolling(window=sma_period).mean()
    return (sma_series - sma_series.shift(1))


def calculate_ema_200(data: pd.DataFrame) -> pd.Series:
    """Calculates the 200-period Exponential Moving Average (EMA)."""
    return data['Close'].squeeze().ewm(span=200, adjust=False).mean()


def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculates the Exponential Moving Average (EMA)."""
    return data['Close'].squeeze().ewm(span=period, adjust=False).mean()


def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """Calculates the Moving Average Convergence Divergence (MACD) indicator."""
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
    """Calculates the highest close price over a lookback period."""
    return data['Close'].squeeze().rolling(window=period).max()


def calculate_lookback_low(data: pd.DataFrame, period: int = 120) -> pd.Series:
    """Calculates the lowest close price over a lookback period."""
    return data['Close'].squeeze().rolling(window=period).min()


def calculate_roc(data: pd.DataFrame, period: int = 5) -> pd.Series:
    """Calculates the Rate of Change (ROC)."""
    close_series = data['Close'].squeeze()
    return (close_series - close_series.shift(period)) / close_series.shift(period)


# --- Base Strategy Classes ---

class BaseStrategy:
    """
    Base class for all trading strategies, handling portfolio management and trade execution logic.
    """

    def __init__(self, initial_capital=1000.0, stop_loss_percent=0.01, is_long_only=True):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.stop_loss_percent = stop_loss_percent
        self.is_long_only = is_long_only

        self.invested_value = 0.0
        self.shares = 0.0
        self.entry_value = 0.0
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

    def _calculate_current_portfolio_value(self, current_price):
        """Calculates the current portfolio value considering long or short position."""
        if self.shares > 0:
            # Long position: cash + (shares * current_price)
            return self.cash + (self.shares * current_price)
        elif self.shares < 0:
            # Short position: cash + (entry_value - (abs(shares) * current_price))
            return self.cash + (self.entry_value - (abs(self.shares) * current_price))
        else:
            return self.cash

    def _update_portfolio(self, index, row):
        """Updates portfolio tracking metrics for the current bar."""
        current_date = index.date()
        current_price = row['Close']

        if self.current_day is None:
            self.current_day = current_date
            if not self.portfolio_dates:
                self.portfolio_value_history.append(self.initial_capital)
                self.portfolio_dates.append(index)
                self.last_close_price = current_price
                return self.initial_capital

        if current_date != self.current_day:
            self.trade_today = False
            self.current_day = current_date

        if self.last_close_price is not None:
            if self.shares > 0:
                self.invested_value = self.shares * current_price
                self.peak_invested_value = max(self.peak_invested_value, self.invested_value)
            elif self.shares < 0:
                current_cost_to_cover = abs(self.shares) * current_price
                if self.peak_invested_value == 0:
                    self.peak_invested_value = current_cost_to_cover
                # For shorts, peak is the lowest invested value (highest unrealized profit)
                self.peak_invested_value = min(self.peak_invested_value, current_cost_to_cover)
                self.invested_value = current_cost_to_cover

        current_portfolio_value = self._calculate_current_portfolio_value(current_price)

        last_valid_value = self.portfolio_value_history[-1] if self.portfolio_value_history else self.initial_capital

        # Basic check to filter out extreme data jumps (e.g., if a bar is missing or corrupted)
        MAX_JUMP_PERCENT = 0.50
        change_from_last = abs(current_portfolio_value - last_valid_value)

        if change_from_last > (last_valid_value * MAX_JUMP_PERCENT) and last_valid_value > 0:
            self.portfolio_value_history.append(last_valid_value)
        else:
            self.portfolio_value_history.append(current_portfolio_value)

        self.portfolio_dates.append(index)
        self.last_close_price = current_price
        return current_portfolio_value

    def _execute_buy(self, index, investment_amount):
        """Opens a LONG position."""
        if self.trade_today or self.shares > 0.01 or self.cash < 0.01 or investment_amount < 0.01:
            return

        buy_price = index['Close']
        shares_bought = investment_amount / buy_price

        self.shares += shares_bought
        self.cash -= investment_amount

        self.entry_value = investment_amount
        self.invested_value = self.shares * buy_price
        self.peak_invested_value = self.invested_value

        self.buy_dates.append(index.name)
        self.trade_today = True

        self.trade_log.append({
            'Time': index.name, 'Type': 'BUY (LONG)', 'Price': buy_price,
            'Shares': shares_bought, 'Value': investment_amount, 'Strategy': self.strategy_name
        })

    def _execute_sell(self, index, sell_type):
        """Closes a LONG position."""
        if self.shares < 0.01:
            return

        sell_price = index['Close']
        cash_received = self.shares * sell_price

        self.trade_log.append({
            'Time': index.name, 'Type': f'SELL ({sell_type})', 'Price': sell_price,
            'Shares': self.shares, 'Value': cash_received, 'Strategy': self.strategy_name
        })

        self.cash += cash_received
        self.invested_value = 0.0
        self.entry_value = 0.0
        self.shares = 0.0
        self.peak_invested_value = 0.0
        self.trade_today = True # Prevent immediate re-entry

        if sell_type == "STOP-LOSS":
            self.sell_dates_stoploss.append(index.name)
        elif sell_type == "SIGNAL":
            self.sell_dates_signal.append(index.name)
        elif sell_type == "EOD":
            self.sell_dates_eod.append(index.name)

    def _execute_short(self, index, cash_for_collateral):
        """Opens a SHORT position."""
        if self.trade_today or self.shares < -0.01 or self.is_long_only or self.cash < 0.01 or cash_for_collateral < 0.01:
            return

        short_price = index['Close']
        shares_shorted = cash_for_collateral / short_price

        # Cash increases by the short proceeds, shares are negative
        self.shares = -shares_shorted
        self.cash += cash_for_collateral

        self.entry_value = cash_for_collateral
        self.invested_value = abs(self.shares) * short_price # The value of the shorted stock
        self.peak_invested_value = self.invested_value

        self.sell_dates_signal.append(index.name) # Using signal dates for short open
        self.trade_today = True

        self.trade_log.append({
            'Time': index.name, 'Type': 'SELL (SHORT)', 'Price': short_price,
            'Shares': abs(self.shares), 'Value': cash_for_collateral, 'Strategy': self.strategy_name
        })

    def _execute_cover(self, index, shares_to_cover, stop_loss=False):
        """Closes a SHORT position."""
        if self.shares > -0.01:
            return

        cover_price = index['Close']
        cash_used_to_cover = abs(self.shares) * cover_price

        self.trade_log.append({
            'Time': index.name, 'Type': f'BUY (COVER: {"SL" if stop_loss else "SIGNAL"})',
            'Price': cover_price, 'Shares': abs(self.shares), 'Value': cash_used_to_cover,
            'Strategy': self.strategy_name
        })

        self.cash -= cash_used_to_cover

        if stop_loss:
            self.sell_dates_stoploss.append(index.name)
        else:
            self.buy_dates.append(index.name) # Using buy dates for short cover

        self.invested_value = 0.0
        self.entry_value = 0.0
        self.shares = 0.0
        self.peak_invested_value = 0.0
        self.trade_today = True

    def _check_stop_loss(self, index):
        """Checks and executes stop-loss for both long and short positions."""
        if self.shares > 0.01:  # Long Stop-Loss
            stop_loss_threshold = self.peak_invested_value * (1 - self.stop_loss_percent)
            if self.invested_value < stop_loss_threshold:
                self._execute_sell(index, "STOP-LOSS")
                return True
        elif self.shares < -0.01:  # Short Stop-Loss
            # Short loss occurs when current cost to cover is high (peak_invested_value is low)
            # The invested value (cost to cover) crosses the *higher* threshold
            stop_loss_threshold = self.peak_invested_value / (1 - self.stop_loss_percent)
            if self.invested_value > stop_loss_threshold:
                self._execute_cover(index, abs(self.shares), stop_loss=True)
                return True
        return False

    def run_strategy(self, analysis_df: pd.DataFrame):
        """Placeholder for the strategy execution logic."""
        raise NotImplementedError("Subclasses must implement run_strategy()")


class InverseBaseStrategy(BaseStrategy):
    """A base class for short-only strategies."""

    def __init__(self, *args, **kwargs):
        kwargs['is_long_only'] = False
        super().__init__(*args, **kwargs)
        self.strategy_name = "Inverse " + self.strategy_name


# --- Specific Strategy Implementations ---

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

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


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

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


class TripleSMACrossoverStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_name = "Triple SMA (13/21/55) + 8 EMA Crossover"
        self.last_8_ema = None
        self.last_55_sma = None

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            self._update_portfolio(index, row)

            if index.time() < time(10, 0):
                self.last_8_ema = row['EMA_8']
                self.last_55_sma = row['SMA_55']
                continue

            if self.last_8_ema is None or self.last_55_sma is None:
                self.last_8_ema = row['EMA_8']
                self.last_55_sma = row['SMA_55']
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                self.last_8_ema = row['EMA_8']
                self.last_55_sma = row['SMA_55']
                continue

            if self._check_stop_loss(row):
                self.last_8_ema = row['EMA_8']
                self.last_55_sma = row['SMA_55']
                continue

            is_buy_signal = (self.shares < 0.01 and
                             not self.trade_today and
                             self.last_8_ema < self.last_55_sma and
                             row['EMA_8'] > row['SMA_55'])

            is_sell_signal = (self.shares > 0.01 and
                              self.last_8_ema > self.last_55_sma and
                              row['EMA_8'] < row['SMA_55'])

            if is_sell_signal:
                self._execute_sell(row, "TRIPLE SMA CROSS SELL")
            elif is_buy_signal:
                if self.cash > 0.01:
                    self._execute_buy(row, self.cash)

            self.last_8_ema = row['EMA_8']
            self.last_55_sma = row['SMA_55']

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


class MACDStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(is_long_only=True, *args, **kwargs)
        self.strategy_name = "MACD Crossover (100% Cash)"
        self.last_macd = None
        self.last_signal = None

    def run_strategy(self, analysis_df: pd.DataFrame):
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            self._update_portfolio(index, row)

            if index.time() < time(10, 0):
                self.last_macd = row['MACD']
                self.last_signal = row['Signal_Line']
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

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])

class InverseMACDStrategy(InverseBaseStrategy):
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
                self.last_macd = row['MACD']
                self.last_signal = row['Signal_Line']
                continue

            if index.time() == time(15, 59):
                if self.shares < -0.01:
                    self._execute_cover(row, abs(self.shares), stop_loss=False)
                    self.last_macd, self.last_signal = None, None
                continue

            if self._check_stop_loss(row):
                self.last_macd, self.last_signal = None, None
                continue

            if self.last_macd is not None:
                if self.shares < -0.01:
                    is_signal_cover = self.last_macd < self.last_signal and row['MACD'] > row['Signal_Line']
                    if is_signal_cover:
                        self._execute_cover(row, abs(self.shares))
                else:
                    is_signal_short = (not self.trade_today and
                                       self.last_macd > self.last_signal and
                                       row['MACD'] < row['Signal_Line'] and
                                       row['SMA_Slope_1m'] < 0 and
                                       row['Close'] < row['EMA_200'])

                    if is_signal_short and self.cash > 0.01:
                        self._execute_short(row, self.initial_capital)

            self.last_macd = row['MACD']
            self.last_signal = row['Signal_Line']

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


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

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


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

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


class AllInOneHybridStrategy(BaseStrategy):
    def __init__(self, initial_capital, stop_loss_percent):
        super().__init__(initial_capital, stop_loss_percent)
        self.strategy_name = "All-In-One Hybrid (1/5th Split)"

        fifth_capital = initial_capital / 5.0

        # Initialize sub-strategies
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

        # Overwrite BaseStrategy's tracking with combined tracking
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
        # Run all sub-strategies
        for sub in self.sub_strategies:
            sub.run_strategy(analysis_df.copy())

        # Consolidate results
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
            # Ensure series have a proper index to allow for summing
            history[sub.strategy_name] = pd.Series(sub.portfolio_value_history, index=sub.portfolio_dates)

        # Combine all portfolio history series
        combined_history = sum(history.values()).sort_index()

        self.portfolio_dates = combined_history.index.tolist()
        self.portfolio_value_history = combined_history.tolist()

        # Calculate final total value
        final_value = sum(sub._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close']) for sub in self.sub_strategies)
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
        from random import randint
        current_hour = current_time.hour
        current_minute = current_time.minute

        # Set a random minute for the current hour block
        if current_hour >= 10 and current_hour < 15:
            self.target_minute = randint(0, 59)
            return

        # Set a random minute for the final hour block
        if current_hour == 15:
            self.target_minute = randint(0, 59)
            return

        self.target_minute = -1

    def run_strategy(self, analysis_df: pd.DataFrame):
        from random import random
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue

            current_time = index.time()
            self._update_portfolio(index, row)

            if current_time < time(10, 0):
                continue

            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                continue

            if self._check_stop_loss(row):
                continue

            is_target_minute = (current_time.minute == self.target_minute)

            if is_target_minute:
                # Set next target minute (for the next full hour block)
                current_hour_block = index.hour
                if current_hour_block >= 10 and current_hour_block < 15:
                    # In a real-time system, this logic would need to be time-based,
                    # but for this backtest we set the next target once the current one is hit.
                    self._set_next_target_minute(time(current_hour_block + 1, 0))
                elif current_hour_block == 15:
                    self.target_minute = -1 # End of day

                # Execute trade logic
                if random() < 0.5: # 50% chance to sell if in a position
                    if self.shares > 0.01:
                        self._execute_sell(row, "HOURLY RANDOM SELL")
                elif not self.trade_today: # Otherwise, 50% chance to buy
                    if self.cash * self.allocation_fraction > 0.01:
                        investment_amount = self.cash * self.allocation_fraction
                        # Temporarily override trade_today to allow the buy, then set it back inside _execute_buy
                        self.trade_today = False
                        self._execute_buy(row, investment_amount)

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


class RandomTraderStrategy(BaseStrategy):
    def __init__(self, initial_capital=1000.0, stop_loss_percent=0.01, trade_chance=0.01):
        super().__init__(initial_capital=initial_capital, stop_loss_percent=stop_loss_percent)
        self.strategy_name = "Random Trader (1% Chance)"
        self.trade_chance = trade_chance

    def run_strategy(self, analysis_df: pd.DataFrame):
        from random import random
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

            if self.shares > 0.01:
                if random() < self.trade_chance:
                    self._execute_sell(row, "RANDOM SIGNAL")
            else:
                if not self.trade_today and random() < self.trade_chance:
                    if self.cash > 0.01:
                        self._execute_buy(row, self.cash)

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


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

                # Dynamic support is the minimum low of the current swing
                dynamic_support = min(self.price_history) if self.price_history else row['Low']
                if row['Close'] < dynamic_support:
                    self._execute_sell(row, "SWING-LOW BREAK")
                    self.price_history = []

            else:
                # Buy signal based on uptrend conditions
                if (not self.trade_today and
                        row['SMA_Slope_1m'] > 0 and
                        row['Close'] > row['EMA_200']):

                    if self.cash > 0.01:
                        self._execute_buy(row, self.cash)
                        self.price_history = [row['Low']]

        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


class PerfectForesightStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_name = "Perfect Foresight (Max Return Long Benchmark)"

    def run_strategy(self, analysis_df: pd.DataFrame):
        self.portfolio_value_history = []
        self.portfolio_dates = []

        trading_hours_df = analysis_df[
            (analysis_df.index.time >= time(10, 0)) &
            (analysis_df.index.time <= time(16, 0))
            ].copy()

        if trading_hours_df.empty:
            return self.initial_capital

        # --- Corrected Logic: Find Max Profit Pair (Buy Low, Sell High, Buy before Sell) ---

        best_profit = -float('inf')
        buy_index_time = None
        sell_index_time = None

        # Iterate through all possible buy points (as lowest low)
        for i in range(len(trading_hours_df) - 1):
            current_buy_row = trading_hours_df.iloc[i]
            buy_price = current_buy_row['Close']

            # Look for the highest sell price *after* the current buy point
            # Use .iloc[i+1:] to ensure the sell is strictly after the buy
            future_df = trading_hours_df.iloc[i + 1:]

            if future_df.empty:
                continue

            highest_future_close = future_df['Close'].max()

            if highest_future_close > buy_price:
                current_profit = (highest_future_close - buy_price) / buy_price

                if current_profit > best_profit:
                    best_profit = current_profit
                    buy_index_time = current_buy_row.name

                    # Find the corresponding sell index time
                    # We take the *first* highest close after the buy point for simplicity,
                    # but taking the last one is also valid for maximizing the position time.
                    sell_index_time = future_df[future_df['Close'] == highest_future_close].index[0]

        # If no profitable trade is found, return initial capital
        if best_profit <= 0 or buy_index_time is None:
            return self.initial_capital

        # --- End Corrected Logic ---

        for index, row in analysis_df.iterrows():
            self._update_portfolio(index, row)

            # Check if current time is the optimal buy time
            if index == buy_index_time and self.shares < 0.01:
                # Perfect Buy
                shares_bought = self.initial_capital / row['Close']  # Use the actual close price at the time
                self.trade_log.append({
                    'Time': buy_index_time, 'Type': 'BUY (LONG)', 'Price': row['Close'],
                    'Shares': shares_bought, 'Value': self.initial_capital, 'Strategy': self.strategy_name
                })
                self.buy_dates.append(buy_index_time)
                self.cash = 0.0
                self.shares = shares_bought
                self.invested_value = self.initial_capital
                self.peak_invested_value = self.initial_capital

            # Check if current time is the optimal sell time
            elif index == sell_index_time and self.shares > 0.01:
                # Perfect Sell
                cash_received = self.shares * row['Close']  # Use the actual close price at the time
                self.trade_log.append({
                    'Time': sell_index_time, 'Type': 'SELL (PERFECT)', 'Price': row['Close'],
                    'Shares': self.shares, 'Value': cash_received, 'Strategy': self.strategy_name
                })
                self.sell_dates_signal.append(sell_index_time)
                self.cash += cash_received
                self.shares = 0.0
                self.invested_value = 0.0
                self.peak_invested_value = 0.0

        # Ensure final update
        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])


class PerfectShortForesightStrategy(InverseBaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_name = "Perfect Foresight (Max Return Short Benchmark)"
        self.initial_capital_base = self.initial_capital

    def run_strategy(self, analysis_df: pd.DataFrame):
        self.portfolio_value_history = []
        self.portfolio_dates = []

        trading_hours_df = analysis_df[
            (analysis_df.index.time >= time(10, 0)) &
            (analysis_df.index.time <= time(16, 0))
            ].copy()

        if trading_hours_df.empty:
            return self.initial_capital

        # --- Corrected Logic: Find Max Profit Pair (Short High, Cover Low, Short before Cover) ---

        best_profit = -float('inf')
        short_index_time = None
        cover_index_time = None

        # Iterate through all possible short points (as highest high)
        for i in range(len(trading_hours_df) - 1):
            current_short_row = trading_hours_df.iloc[i]
            short_price = current_short_row['Close']

            # Look for the lowest cover price *after* the current short point
            # Use .iloc[i+1:] to ensure the cover is strictly after the short
            future_df = trading_hours_df.iloc[i + 1:]

            if future_df.empty:
                continue

            lowest_future_close = future_df['Close'].min()

            # Profit in a short is the initial high price minus the later low price
            if short_price > lowest_future_close:
                current_profit = (short_price - lowest_future_close) / short_price

                if current_profit > best_profit:
                    best_profit = current_profit
                    short_index_time = current_short_row.name

                    # Find the corresponding cover index time
                    cover_index_time = future_df[future_df['Close'] == lowest_future_close].index[0]

        # If no profitable trade is found, or if shorting would lead to a loss, return initial capital
        if best_profit <= 0 or short_index_time is None:
            return self.initial_capital

        # --- End Corrected Logic ---

        for index, row in analysis_df.iterrows():
            self._update_portfolio(index, row)

            if index == short_index_time and self.shares > -0.01:
                # Perfect Short (Sell)
                shares_shorted = self.initial_capital_base / row['Close']  # Use the actual close price at the time
                self.trade_log.append({
                    'Time': short_index_time, 'Type': 'SELL (SHORT)', 'Price': row['Close'],
                    'Shares': shares_shorted, 'Value': self.initial_capital_base, 'Strategy': self.strategy_name
                })
                self.sell_dates_signal.append(short_index_time)
                self.shares = -shares_shorted
                self.cash += self.initial_capital_base
                self.entry_value = self.initial_capital_base
                self.invested_value = abs(self.shares) * row['Close']
                self.peak_invested_value = self.invested_value

            elif index == cover_index_time and self.shares < -0.01:
                # Perfect Cover (Buy)
                cash_used_to_cover = abs(self.shares) * row['Close']  # Use the actual close price at the time
                self.trade_log.append({
                    'Time': cover_index_time, 'Type': 'BUY (COVER: PERFECT)', 'Price': row['Close'],
                    'Shares': abs(self.shares), 'Value': cash_used_to_cover, 'Strategy': self.strategy_name
                })
                self.buy_dates.append(cover_index_time)
                self.cash -= cash_used_to_cover
                self.shares = 0.0
                self.invested_value = 0.0
                self.peak_invested_value = 0.0

        # Ensure final update
        return self._calculate_current_portfolio_value(analysis_df.iloc[-1]['Close'])

# --- Q-Learning Specific Utility ---

def get_state(df_row: pd.Series, ticker: str = 'SPXUSD') -> tuple:
    """
    Discretizes the continuous indicators into a state tuple.
    NOTE: This must exactly match the state definition used during training.
    """
    # 1. EMA_8 vs SMA_55 Cross (Indicators must be present in df_row index)
    ema8 = df_row['EMA_8']
    sma55 = df_row['SMA_55']
    ema_cross_state = 0
    if ema8 > sma55 * 1.0005:  # 0.05% buffer for Bullish
        ema_cross_state = 1
    elif ema8 < sma55 * 0.9995:  # 0.05% buffer for Bearish
        ema_cross_state = -1

    # 2. MACD Histogram (Positive/Negative)
    macd_histo = df_row['MACD_Histogram']
    macd_state = 0
    if macd_histo > 0.005:  # Small threshold for 'Positive'
        macd_state = 1
    elif macd_histo < -0.005:  # Small threshold for 'Negative'
        macd_state = -1

    # 3. SMA Slope (Positive/Negative)
    sma_slope = df_row['SMA_Slope_1m']
    slope_state = 0
    if sma_slope > 0.005:  # Small percentage threshold
        slope_state = 1
    elif sma_slope < -0.005:  # Small percentage threshold
        slope_state = -1

    return (ema_cross_state, macd_state, slope_state)


# --- Q-Learning Strategy Implementation ---

class QLearningStrategy(BaseStrategy):
    """
    A trading strategy that uses a loaded Q-Table to make buy/sell decisions.
    """

    def __init__(self, model_ticker: str, initial_capital=1000.0, stop_loss_percent=0.01,
                 model_dir='qlearning_results'):
        super().__init__(initial_capital=initial_capital, stop_loss_percent=stop_loss_percent)

        self.strategy_name = f"Q-Agent ({model_ticker})"
        self.model_ticker = model_ticker
        self.q_table = self._load_q_table(model_dir, model_ticker)

        self.action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        self.last_action = 0

    def _load_q_table(self, model_dir, model_ticker):
        """Loads the Q-Table CSV file into a dictionary."""
        # Note: We hardcode the trained ticker as SPXUSD/MOCK since the Q-table name depends on the training data.
        TICKER_SYMBOL_TRAINED = 'SPXUSD'  # Change to 'MOCK' if you trained on mock data
        model_filename = os.path.join(model_dir, f'q_table_{TICKER_SYMBOL_TRAINED}_model.ai')

        if not os.path.exists(model_filename):
            print(f"FATAL: Q-Table not found at {model_filename}. Q-Agent defaulting to HOLD.")
            return {}

        try:
            q_table_df = pd.read_csv(model_filename, index_col=0)
            q_table = {ast.literal_eval(k): v.values for k, v in q_table_df.iterrows()}
            print(f"✅ Q-Table loaded successfully for Q-Agent testing.")
            return q_table
        except Exception as e:
            print(f"ERROR: Failed to parse Q-Table CSV: {e}")
            return {}

    def run_strategy(self, analysis_df: pd.DataFrame):
        """
        Executes the strategy based on Q-Table lookups (Exploitation).
        """
        for index, row in analysis_df.iterrows():
            if pd.isna(row['Close']):
                continue
            if index.time() < time(10, 0):
                continue
            # 1. Update Portfolio and Check Stop Loss
            self._update_portfolio(index, row)

            # End of Day Check (Close out position at 15:59 ET)
            if index.time() == time(15, 59):
                if self.shares > 0.01:
                    self._execute_sell(row, "EOD")
                continue

            if self._check_stop_loss(row):
                self.last_action = 2
                continue

            # 2. Determine Current State
            current_state = get_state(row, self.model_ticker)

            # 3. Choose Action (Exploitation)
            if current_state in self.q_table:
                action_index = np.argmax(self.q_table[current_state])
            else:
                action_index = 0  # Unseen state, default to HOLD

            action = self.action_map[action_index]

            # 4. Execute Action
            if action == 'Buy':
                if self.shares < 0.01 and not self.trade_today:
                    if self.cash > 0.01:
                        self._execute_buy(row, self.cash)
                        self.last_action = 1

            elif action == 'Sell':
                if self.shares > 0.01:
                    self._execute_sell(row, "SIGNAL")
                    self.last_action = 2

        # Final close out if not done at 15:59
        final_close = analysis_df.iloc[-1]
        if self.shares > 0.01:
            self._execute_sell(final_close, "EOD")

        return self._calculate_current_portfolio_value(final_close['Close'])