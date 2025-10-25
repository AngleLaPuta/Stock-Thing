import pandas as pd
import matplotlib.pyplot as plt
import os
LOG_FILEPATH = 'run_results_log.csv'
INITIAL_CAPITAL = 10000.0
PLOT_OUTPUT_FILE = 'strategy_profit_plot.png'
TIME_INDEX_STRATEGY = "Perfect Foresight (Max Return Short Benchmark)"
OUTPUT_CHART_FOLDER = 'Return_Analysis_Charts'
def generate_profit_plot_and_leaderboard():
    if not os.path.exists(LOG_FILEPATH):
        print(f"ERROR: Log file not found at {LOG_FILEPATH}")
        return
    try:
        df = pd.read_csv(LOG_FILEPATH,
                         names=['Strategy', 'DailyReturnPercent'],
                         header=None,
                         skiprows=1)
    except Exception as e:
        print(f"Error reading or parsing the log file: {e}")
        return
    df['DailyReturnDecimal'] = df['DailyReturnPercent'].str.replace('%', '').astype(float) / 100.0
    df['DayIndex'] = df[df['Strategy'] == TIME_INDEX_STRATEGY].groupby('Strategy').cumcount().add(1)
    df['DayIndex'] = df['DayIndex'].fillna(method='ffill').fillna(1).astype(int)
    df['Multiplier'] = 1.0 + df['DailyReturnDecimal']
    daily_returns_df = df.groupby(['DayIndex', 'Strategy'])['Multiplier'].prod().unstack(fill_value=1.0)
    daily_percent_returns = daily_returns_df - 1.0
    cumulative_multiplier = daily_returns_df.cumprod()
    equity_df = cumulative_multiplier * INITIAL_CAPITAL
    print("\n" + "=" * 60)
    print("📈 FINAL TRADING STRATEGY LEADERBOARD")
    print(f"(Based on {equity_df.shape[0]} Simulated Days)")
    print("=" * 60)
    if equity_df.empty:
        print("No valid data found after processing log entries.")
        print("=" * 60 + "\n")
        return
    final_equity = equity_df.iloc[-1].sort_values(ascending=False)
    leaderboard = []
    for name, final_value in final_equity.items():
        total_return_decimal = (final_value / INITIAL_CAPITAL) - 1
        total_return_percent = total_return_decimal * 100.0
        leaderboard.append({
            'Strategy': name,
            'Final Equity': final_value,
            'Total Return %': total_return_percent
        })
    for rank, entry in enumerate(leaderboard, 1):
        rank_str = f"#{rank: <3}"
        name_str = f"{entry['Strategy']: <35}"
        equity_str = f"${entry['Final Equity']:>10,.2f}"
        return_str = f"{entry['Total Return %']:>10.2f}%"
        color_start = "\033[92m" if entry['Total Return %'] >= 0 else "\033[91m"
        color_end = "\033[0m"
        print(f"{rank_str} | {name_str} | Final Value: {equity_str} | Return: {color_start}{return_str}{color_end}")
    print("=" * 60 + "\n")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in equity_df.columns:
        ax.plot(equity_df.index, equity_df[column], label=column, linewidth=2, alpha=0.8)
    ax.axhline(INITIAL_CAPITAL, color='gray', linestyle='--', linewidth=1,
               label=f'Initial Capital (${INITIAL_CAPITAL:,.0f})')
    ax.set_title('Cumulative Trading Strategy Performance (Equity) Over Simulated Days', fontsize=16, fontweight='bold')
    ax.set_xlabel('Simulated Day Index', fontsize=12)
    ax.set_ylabel(f'Equity Value (USD)', fontsize=12)
    y_formatter = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(OUTPUT_CHART_FOLDER,PLOT_OUTPUT_FILE))
    print(f"✅ Plot successfully generated and saved to: {PLOT_OUTPUT_FILE}")
    try:
        if not os.path.exists(OUTPUT_CHART_FOLDER):
            os.makedirs(OUTPUT_CHART_FOLDER)
    except OSError as e:
        print(f"Error creating directory {OUTPUT_CHART_FOLDER}: {e}")
        return
    max_returns = daily_percent_returns.max()
    avg_returns = daily_percent_returns.mean()
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_max, ax_max = plt.subplots(figsize=(12, 7))
    sorted_max_returns = max_returns.sort_values(ascending=False)
    colors = plt.cm.viridis(sorted_max_returns.rank(method='first') / len(sorted_max_returns))
    bars = ax_max.bar(sorted_max_returns.index, sorted_max_returns, color=colors)
    ax_max.set_title('Maximum Daily Return by Strategy', fontsize=18, fontweight='bold', color='#333333')
    ax_max.set_xlabel('Strategy', fontsize=14, color='#555555')
    ax_max.set_ylabel('Max Return (%)', fontsize=14, color='#555555')
    ax_max.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    ax_max.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax_max.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, '{:.2%}'.format(yval),
                    ha='center', va='bottom', fontsize=9, color='black')
    plt.tight_layout()
    max_return_file = os.path.join(OUTPUT_CHART_FOLDER, 'max_daily_return_chart.png')
    plt.savefig(max_return_file, dpi=300)
    plt.close(fig_max)
    print(f"✅ Generated Max Return chart: {max_return_file}")
    fig_avg, ax_avg = plt.subplots(figsize=(12, 7))
    sorted_avg_returns = avg_returns.sort_values(ascending=False)
    colors = plt.cm.plasma(sorted_avg_returns.rank(method='first') / len(sorted_avg_returns))
    bars = ax_avg.bar(sorted_avg_returns.index, sorted_avg_returns, color=colors)
    ax_avg.set_title('Average Daily Return by Strategy', fontsize=18, fontweight='bold', color='#333333')
    ax_avg.set_xlabel('Strategy', fontsize=14, color='#555555')
    ax_avg.set_ylabel('Average Return (%)', fontsize=14, color='#555555')
    ax_avg.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    ax_avg.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax_avg.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, '{:.2%}'.format(yval),
                    ha='center', va='bottom', fontsize=9, color='black')
    plt.tight_layout()
    avg_return_file = os.path.join(OUTPUT_CHART_FOLDER, 'average_daily_return_chart.png')
    plt.savefig(avg_return_file, dpi=300)
    plt.close(fig_avg)
    print(f"✅ Generated Average Return chart: {avg_return_file}")
if __name__ == "__main__":
    generate_profit_plot_and_leaderboard()