import os
from datetime import time, timedelta, datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

TICKER_SYMBOL_TRAINED = 'SPXUSD'
SAVE_DIR = 'lstm_results_v2'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'weights'), exist_ok=True)


class StockLSTM_V2(nn.Module):
    """LSTM Model with 5 Outputs: Direction, Rel High, Rel Low, Time High, Time Low."""

    def __init__(self, input_dim, hidden_dim, num_layers, output_classes=2, output_regs=4):
        super(StockLSTM_V2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_cls = nn.Linear(hidden_dim, output_classes)
        self.fc_reg = nn.Linear(hidden_dim, output_regs)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        final_state = hn[-1]
        cls_output = self.fc_cls(final_state)
        reg_output = self.fc_reg(final_state)
        return cls_output, reg_output


def prepare_lstm_data_v2(df: pd.DataFrame, ticker: str, sequence_length: int = 60):
    """
    Transforms the indicator DataFrame into LSTM sequences (X) and 5 targets (Y).
    Y: [Direction, Relative High %, Relative Low %, Time to High (min), Time to Low (min)]
    """
    relevant_metrics = [
        'Close', 'Price_Change', 'Close_vs_EMA200', 'EMA8_vs_SMA55',
        'MACD', 'Signal_Line', 'MACD_Histogram', 'SMA_Slope_1m'
    ]
    data = df[ticker].copy()
    data = data.loc[:, relevant_metrics]
    data.index = data.index.tz_convert('America/New_York')
    all_days = data.index.normalize().unique().tolist()
    feature_cols = [col for col in relevant_metrics if col != 'Close']
    X_sequences = []
    Y_targets = []
    print(f"Preparing data for {len(all_days)} days...")
    for day in all_days:
        date_str = day.strftime('%Y-%m-%d')
        start_time_x = datetime.combine(day.date(), time(9, 30)).replace(tzinfo=day.tz)
        end_time_x = datetime.combine(day.date(), time(10, 30)).replace(tzinfo=day.tz) - timedelta(minutes=1)
        start_time_y = datetime.combine(day.date(), time(10, 30)).replace(tzinfo=day.tz)
        end_time_y = datetime.combine(day.date(), time(15, 59)).replace(tzinfo=day.tz)
        df_x = data.loc[start_time_x:end_time_x]
        df_y = data.loc[start_time_y:end_time_y]
        if len(df_x) < sequence_length or df_y.empty:
            continue
        price_1030 = df_x['Close'].iloc[-1]
        close_price = df_y['Close'].iloc[-1]
        direction = 1 if close_price > price_1030 else 0
        rest_of_day_prices = df_y['Close']
        rest_of_day_high = rest_of_day_prices.max()
        rest_of_day_low = rest_of_day_prices.min()
        high_time_index = rest_of_day_prices[rest_of_day_prices == rest_of_day_high].index[0]
        low_time_index = rest_of_day_prices[rest_of_day_prices == rest_of_day_low].index[0]
        relative_high = ((rest_of_day_high - price_1030) / price_1030) * 100
        relative_low = ((rest_of_day_low - price_1030) / price_1030) * 100
        time_diff_high = high_time_index - start_time_y
        time_to_high_minutes = time_diff_high.total_seconds() / 60
        time_diff_low = low_time_index - start_time_y
        time_to_low_minutes = time_diff_low.total_seconds() / 60
        Y_targets.append([direction, relative_high, relative_low, time_to_high_minutes, time_to_low_minutes])
        X_sequence = df_x[feature_cols].values
        X_sequences.append(X_sequence)
    X = np.array(X_sequences)
    Y = np.array(Y_targets)
    num_samples, seq_len, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    mean = X_reshaped.mean(axis=0);
    std = X_reshaped.std(axis=0);
    train_mean = mean
    train_std = std
    std[std == 0] = 1e-6
    X_normalized = ((X_reshaped - mean) / std).reshape(num_samples, seq_len, num_features)
    Y_reg_mean = Y[:, 1:].mean(axis=0)
    Y_reg_std = Y[:, 1:].std(axis=0)
    Y_reg_std[Y_reg_std == 0] = 1.0
    Y_reg_normalized = (Y[:, 1:] - Y_reg_mean) / Y_reg_std
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    Y_cls_tensor = torch.tensor(Y[:, 0], dtype=torch.long)
    Y_reg_tensor = torch.tensor(Y_reg_normalized, dtype=torch.float32)
    print(f"Data preparation complete. {len(X_tensor)} sequences ready.")
    return (
        X_tensor,
        Y_cls_tensor,
        Y_reg_tensor,
        num_features,
        Y_reg_mean,
        Y_reg_std,
        train_mean,
        train_std
    )


def predict_stock_movement_v2(model: StockLSTM_V2, df_day: pd.DataFrame, ticker: str,
                              Y_reg_mean, Y_reg_std, train_mean, train_std) -> pd.DataFrame:
    """
    Uses the trained LSTM model to predict 5 key metrics, returns a DataFrame
    with prediction details, AND calls plot_price_path_comparison.
    """
    model.eval()
    day = df_day.index.normalize().min()
    start_time_x = datetime.combine(day.date(), time(9, 30)).replace(tzinfo=day.tz)
    end_time_x = datetime.combine(day.date(), time(10, 30)).replace(tzinfo=day.tz) - timedelta(minutes=1)
    data = df_day.copy().swaplevel(0, 1, axis=1)[ticker]
    relevant_metrics = [
        'Close', 'Price_Change', 'Close_vs_EMA200', 'EMA8_vs_SMA55',
        'MACD', 'Signal_Line', 'MACD_Histogram', 'SMA_Slope_1m'
    ]
    feature_cols = [col for col in relevant_metrics if col != 'Close']
    df_x = data.loc[start_time_x:end_time_x]
    if len(df_x) < 60:
        return pd.DataFrame({'Prediction': ['Error: Not enough 9:30-10:30 data.']}), None
    price_1030 = df_x['Close'].iloc[-1]
    X_sequence = df_x[feature_cols].values
    X_normalized = (X_sequence - train_mean) / train_std
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        cls_output, reg_output_normalized = model(X_tensor)
    reg_output_denorm = reg_output_normalized.squeeze().numpy() * Y_reg_std + Y_reg_mean
    predicted_relative_high_perc = reg_output_denorm[0]
    predicted_relative_low_perc = reg_output_denorm[1]
    predicted_time_to_high_min = max(0, reg_output_denorm[2])
    predicted_time_to_low_min = max(0, reg_output_denorm[3])
    predicted_direction_idx = torch.argmax(cls_output, dim=1).item()
    predicted_direction = "Higher Close" if predicted_direction_idx == 1 else "Lower/Same Close"
    predicted_high = price_1030 * (1 + predicted_relative_high_perc / 100)
    predicted_low = price_1030 * (1 + predicted_relative_low_perc / 100)
    start_time_y = datetime.combine(day.date(), time(10, 30)).replace(tzinfo=day.tz)
    predicted_high_time = start_time_y + timedelta(minutes=int(predicted_time_to_high_min))
    predicted_low_time = start_time_y + timedelta(minutes=int(predicted_time_to_low_min))
    if predicted_direction_idx == 1:
        predicted_close = price_1030 + (predicted_high - price_1030) / 2
    else:
        predicted_close = price_1030 + (predicted_low - price_1030) / 2
    end_of_day_time = datetime.combine(day.date(), time(15, 59)).replace(tzinfo=day.tz)
    predicted_high_time = min(predicted_high_time, end_of_day_time)
    predicted_low_time = min(predicted_low_time, end_of_day_time)
    result_df = pd.DataFrame({
        'Metric': [
            '10:30 AM Price',
            'Predicted Close Direction',
            'Predicted Absolute High ($)',
            'Predicted Time to High (ET)',
            'Predicted Absolute Low ($)',
            'Predicted Time to Low (ET)',
            'Approximate Final Price ($)'
        ],
        'Value': [
            f'{price_1030:.2f}',
            predicted_direction,
            f'{predicted_high:.2f}',
            predicted_high_time.strftime('%H:%M'),
            f'{predicted_low:.2f}',
            predicted_low_time.strftime('%H:%M'),
            f'{predicted_close:.2f}'
        ]
    })
    plot_data = {
        'start_price': price_1030,
        'start_time': start_time_y,
        'high_price': predicted_high,
        'high_time': predicted_high_time,
        'low_price': predicted_low,
        'low_time': predicted_low_time,
        'end_price': predicted_close,
        'end_time': end_of_day_time
    }
    date_str = day.strftime('%Y%m%d')
    plot_filename = f'{ticker}_prediction_path_{date_str}.png'
    plot_filepath = os.path.join(SAVE_DIR, plot_filename)
    plot_price_path_comparison(analysis_df=df_day,
                               plot_data=plot_data,
                               ticker_symbol=ticker,
                               file_path=plot_filepath)
    print(f"Plot saved to: {plot_filepath}")
    return result_df, plot_data


def plot_price_path_comparison(analysis_df: pd.DataFrame, plot_data: dict, ticker_symbol: str, file_path: str):
    """
    Plots the actual price action and a polynomial approximation of the predicted path.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    day = analysis_df.index.normalize().min()
    date_str = day.strftime('%Y-%m-%d')
    df_day_slice = analysis_df.loc[date_str].copy()
    close_series = df_day_slice.swaplevel(0, 1, axis=1)[ticker_symbol]['Close']
    plot_index = close_series.index
    start_time = plot_data['start_time']
    end_time = plot_data['end_time']
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
            linewidth=2, linestyle='--', color='red', label='Predicted Path (Approximation)')
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
