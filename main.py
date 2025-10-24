import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import time, timedelta
from random import choice, randint
from matplotlib import gridspec
import numpy as np
import os
from trainLSTM import plot_price_path_comparison
from leaderboardd import generate_profit_plot_and_leaderboard

from strats import *


def plot_comparison_results(analysis_df: pd.DataFrame, all_strategies: list, ticker_symbol: str, plot: bool):
    """
    Generates a multi-panel plot comparing price action, indicators, and strategy performance.
    Saves the plot to a file if 'plot' is False, otherwise displays it.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    if analysis_df.index.empty:
        print("No data to plot.")
        return

    display_date = analysis_df.index[0].strftime('%Y-%m-%d')

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1, 2], hspace=0.1, wspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[:, 1])

    ax1.plot(analysis_df.index, analysis_df['Close'], linewidth=2, color='#1f77b4', label='Close Price')
    ax1.plot(analysis_df.index, analysis_df['EMA_200'], linewidth=1.5, color='#ff7f0e', label='EMA 200')
    ax1.plot(analysis_df.index, analysis_df['SMA_50'], linewidth=1.5, color='#d62728', linestyle=':', alpha=0.7,
             label='SMA 50')
    ax1.plot(analysis_df.index, analysis_df['SMA_55'], linewidth=1.5, color='purple', linestyle='-', alpha=0.7,
             label='SMA 55')
    ax1.plot(analysis_df.index, analysis_df['SMA_21'], linewidth=1.0, color='grey', linestyle='--', alpha=0.7,
             label='SMA 21')
    ax1.plot(analysis_df.index, analysis_df['SMA_13'], linewidth=1.0, color='lightgreen', linestyle='--', alpha=0.7,
             label='SMA 13')
    ax1.plot(analysis_df.index, analysis_df['EMA_8'], linewidth=1.0, color='cyan', linestyle='-', alpha=0.7,
             label='EMA 8')

    def filter_trades_by_date(trade_dates, display_date):
        return [date for date in trade_dates if pd.notna(date) and date.strftime('%Y-%m-%d') == display_date]

    strategy_colors = {}
    color_pool = plt.cm.get_cmap('hsv', len(all_strategies) + 1)

    for i, strategy in enumerate(all_strategies):
        color = color_pool(i)
        strategy_colors[strategy.strategy_name] = color

        label_suffix = ' (Short)' if not strategy.is_long_only else ' (Long)'
        strategy_short_name = strategy.strategy_name.replace(" (100% Cash)", "").replace("Perfect Foresight", "Perfect")

        buy_dates = filter_trades_by_date(strategy.buy_dates, display_date)
        if buy_dates:
            ax1.scatter(buy_dates, analysis_df.loc[buy_dates]['Low'], marker='^', color=color, s=40, zorder=5,
                        label=f'{strategy_short_name} Buy{label_suffix}')

        sell_dates = filter_trades_by_date(strategy.sell_dates_signal + strategy.sell_dates_eod, display_date)
        if sell_dates:
            ax1.scatter(sell_dates, analysis_df.loc[sell_dates]['High'], marker='v', color=color, s=40, zorder=5,
                        label=f'{strategy_short_name} Sell{label_suffix}')

        sl_dates = filter_trades_by_date(strategy.sell_dates_stoploss, display_date)
        if sl_dates:
            ax1.scatter(sl_dates, analysis_df.loc[sl_dates]['Close'], marker='x', color='red', s=70, zorder=5,
                        label=f'{strategy_short_name} Stop-Loss')

    ax1.set_title(f'{ticker_symbol} Price Action & Trade Signals on {display_date}',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', which='both', bottom=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    analysis_df['MACD'].plot(ax=ax2, linewidth=2, color='#0000FF', label='MACD Line')
    analysis_df['Signal_Line'].plot(ax=ax2, linewidth=1.5, color='#FF00FF', label='Signal Line')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    ax2.set_title('MACD Indicator (12, 26, 9)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('MACD Value', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax2.get_xticklabels(), visible=False)

    bar_colors = ['green' if x >= 0 else 'red' for x in analysis_df['MACD_Histogram']]

    ax3.bar(analysis_df.index, analysis_df['MACD_Histogram'],
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
    ax3.set_xlim(analysis_df.index.min(), analysis_df.index.max())

    plt.setp(ax3.get_xticklabels(), rotation=0, ha='center', fontsize=9)

    def get_portfolio_series(strategy, plot_index):
        if not strategy.portfolio_dates:
            if not plot_index.empty:
                return pd.Series(strategy.initial_capital, index=[plot_index.min()]).reindex(plot_index, method='ffill')
            return pd.Series(dtype=float)

        history_df = pd.DataFrame({'Value': strategy.portfolio_value_history}, index=strategy.portfolio_dates)
        history_df = history_df[~history_df.index.duplicated(keep='last')]
        history_df.sort_index(inplace=True)

        return history_df['Value'].reindex(plot_index, method='ffill')

    plot_index = analysis_df.index

    strategy_plot_colors = {
        "Q-Agent": 'darkgreen',
        "Perfect Benchmark": 'black',
        "Perfect Foresight (Max Return Long Benchmark)": 'darkred',
        "Perfect Foresight (Max Return Short Benchmark)": 'brown',
        "EMA Crossover (100% Cash)": '#9370DB',
        "SMA Crossover (100% Cash)": '#008080',
        "Triple SMA (13/21/55) + 8 EMA Crossover": 'darkorange',
        "MACD Crossover (100% Cash)": '#4c72b0',
        "Inverse MACD Crossover (100% Cash)": '#FF4500',
        "MACD Histo Derivative (100% Cash)": '#00FFFF',
        "SMA Derivative Trade": '#FFA500',
        "All-In-One Hybrid (1/5th Split)": '#008000',
        "Random Trader (1% Chance)": '#800000',
        "Random Trader (12% Hourly)": '#8B4513',
        "Swing Low Support (15m)": '#32CD32'
    }

    initial_capital = all_strategies[0].initial_capital if all_strategies else 1000.0
    ax4.axhline(y=initial_capital, color='black', linestyle=':', linewidth=1.5, alpha=0.7,
                label=f'Initial Capital (${initial_capital:.0f})')

    for strategy in all_strategies:




        series = get_portfolio_series(strategy, plot_index)

        if not series.empty:
            name = strategy.strategy_name.replace(" (SPXUSD)", "")
            color = strategy_plot_colors.get(name, strategy_colors.get(strategy.strategy_name, np.random.rand(3, )))
            linestyle = '-'
            linewidth = 3 if 'Perfect' in strategy.strategy_name else 1.5

            if 'Q-Agent' in strategy.strategy_name:
                name = 'Q-Learning Agent'
                linewidth = 2.5
                linestyle = '-.'

            series.plot(ax=ax4, linewidth=linewidth, color=color, linestyle=linestyle, label=name)


    ax4.set_title(f'Portfolio Value Comparison Over Time (One Day) - {ticker_symbol}',
                  fontsize=16, fontweight='bold')
    ax4.set_ylabel('Portfolio Value (USD)', fontsize=12)
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    ax4.legend(loc='best', fontsize=9, ncol=2)
    ax4.set_xlabel('Time (ET)', fontsize=12)

    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.set_xlim(analysis_df.index.min(), analysis_df.index.max())

    plt.setp(ax4.get_xticklabels(), rotation=0, ha='center', fontsize=9)

    plt.tight_layout()


    if plot:
        plt.show()
    else:

        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"{ticker_symbol}_{display_date}_comparison.png"
        save_path = os.path.join(output_dir, filename)

        fig.savefig(save_path)
        plt.close(fig)
        print(f"\n" + "=" * 80)
        print(f"📈 Plot saved to file: '{save_path}'")
        print("=" * 80)


def print_trade_summary(strategies):

    """
    Prints a chronological summary of all trades executed by all strategies.
    """
    all_trades = []
    for strategy in strategies:
        all_trades.extend(strategy.trade_log)

    if not all_trades:
        print("\n--- No Trades Executed ---")
        return

    trades_df = pd.DataFrame(all_trades).sort_values(by='Time').reset_index(drop=True)

    def clean_strategy_name(name):
        if 'Hybrid' in name:
            return name.replace(' Hybrid Sub', '').replace('Hybrid ', 'Hybrid (') + ')'
        return name

    required_initial_cols = ['Time', 'Strategy', 'Type', 'Price', 'Shares', 'Value']
    for col in required_initial_cols:
        if col not in trades_df.columns:
            trades_df[col] = pd.NA

    if 'P/L' in trades_df.columns:
        trades_df = trades_df.drop(columns=['P/L'])

    trades_df['Strategy'] = trades_df['Strategy'].apply(clean_strategy_name)
    trades_df['Time'] = trades_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
    trades_df['Price'] = trades_df['Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
    trades_df['Value'] = trades_df['Value'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "")
    trades_df['Shares'] = trades_df['Shares'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    final_cols = ['Time', 'Strategy', 'Type', 'Price', 'Shares', 'Value']

    trades_df = trades_df[final_cols]
    trades_df.columns = ['Time (ET)', 'Strategy', 'Event/Trade', 'Price', 'Shares', 'Value']

    print("\n" + "=" * 85)
    print("                      📊  CHRONOLOGICAL TRADE/EVENT SUMMARY  📊")
    print("=" * 85)
    print(trades_df.to_string())
    print("=" * 85)


RESULTS_LOG_FILE = 'run_results_log.csv'


def append_run_results(leaderboard_data: list[dict]):

    """
    Appends the Strategy Name and Return% of all strategies to the end of a CSV file.
    A new file is created with a header if it doesn't exist.
    """
    log_data = []
    for data in leaderboard_data:
        log_data.append({
            'Strategy Name': data['name'],
            'Return%': f"{data['return_percent']:.4f}%"
        })

    results_df = pd.DataFrame(log_data)

    write_header = not os.path.exists(RESULTS_LOG_FILE)

    results_df.to_csv(
        RESULTS_LOG_FILE,
        mode='a',
        header=write_header,
        index=False
    )

    print("\n" + "=" * 80)
    print("                  ✅ RUN RESULTS LOG UPDATED ✅")
    print(f"   {len(leaderboard_data)} new strategy results appended to '{RESULTS_LOG_FILE}'")
    print("=" * 80)


def amazingprogram(ticker, plot=True):
    """
    Main function to download data, calculate indicators, run strategies, and plot results.
    """
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



        df['SMA_13'] = calculate_sma(df, period=13)
        df['SMA_21'] = calculate_sma(df, period=21)
        df['SMA_55'] = calculate_sma(df, period=55)
        df['EMA_8'] = calculate_ema(df, period=8)
        df['SMA_50'] = calculate_sma(df, period=50)
        df['EMA_200'] = calculate_ema_200(df)
        macd_results = calculate_macd(df)
        df = df.join(macd_results)
        df['SMA_Slope_1m'] = calculate_sma_slope(df, sma_period=50)


        df['Price_Change'] = df['Close'].pct_change() * 100
        df['Close_vs_EMA200'] = (df['Close'] - df['EMA_200']) / df['EMA_200'] * 100
        df['MACD_vs_Signal'] = (df['MACD'] - df['Signal_Line']) / df['Signal_Line'].abs().replace(0, 1e-6) * 100
        df['EMA8_vs_SMA55'] = (df['EMA_8'] - df['SMA_55']) / df['SMA_55'] * 100
        df['Volatility_Range'] = (df['High'] - df['Low']) / df['Close']

        analysis_df = df.dropna()

        if analysis_df.empty:
            print(
                "Error: DataFrame is empty after calculating indicators and dropping NaNs. Need more historical data.")
            return

        trading_dates = analysis_df.index.normalize().unique().sort_values()

        if len(trading_dates) < 2:
            print(f"Not enough trading days available ({len(trading_dates)}). Need at least 2 for robust testing.")
            return

        test_date = trading_dates[1]
        single_day_df = analysis_df[analysis_df.index.normalize() == test_date].copy()

        if single_day_df.empty:
            print(f"Error: No data found for the test date {test_date.strftime('%Y-%m-%d')}.")
            return

        initial_capital = 1000.0
        stop_loss = 0.01

        MODEL_TICKER = 'SPY'

        print(f"\nTrading and plotting results for: {test_date.strftime('%Y-%m-%d')} (Trading starts at 09:30 AM ET)")



        strategies = [
            QLearningStrategy(model_ticker=MODEL_TICKER, initial_capital=initial_capital, stop_loss_percent=stop_loss),
            DQNStrategy(model_ticker=MODEL_TICKER, initial_capital=initial_capital,
                        stop_loss_percent=stop_loss, model_dir='dqn_results'),
            LSTMStrategy(model_ticker=MODEL_TICKER, initial_capital=initial_capital,
                         stop_loss_percent=stop_loss, model_dir='lstm_results_v2'),
            MACDStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            EMACrossoverStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            SMACrossoverStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            TripleSMACrossoverStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            MACDHistoDerivativeStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            SMAPolynomialStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            AllInOneHybridStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            TimedRandomTraderStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            RandomTraderStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss, trade_chance=0.01),
            SwingLowSupportStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            InverseMACDStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),

            PerfectForesightStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss),
            PerfectShortForesightStrategy(initial_capital=initial_capital, stop_loss_percent=stop_loss)
        ]


        leaderboard_data = []
        for strategy in strategies:
            final_value = strategy.run_strategy(single_day_df.copy())

            leaderboard_data.append({
                'name': strategy.strategy_name,
                'final_value': final_value,
                'return_percent': ((final_value - initial_capital) / initial_capital) * 100,
                'initial_capital': initial_capital
            })

        leaderboard_data.sort(key=lambda x: x['final_value'], reverse=True)


        if plot:
            print(f"\n" + "=" * 60)
            print(f"🥇 STRATEGY LEADERBOARD (Ranked by Final Value) 🥇")
            print(f"Initial Capital: ${initial_capital:.2f} on {test_date.strftime('%Y-%m-%d')}")
            print("=" * 60)

            max_name_len = max(len(d['name']) for d in leaderboard_data)

            header_format = "{:<4} {:<" + str(max_name_len + 5) + "} {:>15} {:>10}"
            print(header_format.format("Rank", "Strategy Name", "Final Value", "Return %"))
            print("-" * (6 + max_name_len + 5 + 15 + 10))

            for rank, data in enumerate(leaderboard_data, 1):
                rank_str = f"#{rank}"
                if rank == 1:
                    rank_str = "🥇"
                elif rank == 2:
                    rank_str = "🥈"
                elif rank == 3:
                    rank_str = "🥉"

                print(header_format.format(
                    rank_str,
                    data['name'],
                    f"${data['final_value']:.2f}",
                    f"{data['return_percent']:.2f}%"
                ))

            print("=" * 60)


        if plot:
            print_trade_summary(strategies)

        plot_comparison_results(single_day_df, strategies, ticker_symbol, plot=plot)



        append_run_results(leaderboard_data)
        generate_profit_plot_and_leaderboard()

    except Exception as e:
        print(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    stock_tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "CSCO",
        "JPM", "V", "MA", "XOM", "CVX", "LLY", "JNJ", "WMT", "COST", "HD",
        "NFLX", "T", "SPY", "QQQ", "DIA"
    ]



    amazingprogram(choice(stock_tickers))