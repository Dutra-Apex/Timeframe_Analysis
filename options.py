import argparse

parser = argparse.ArgumentParser(description='Timeframe Analysis')

# data specifications
parser.add_argument('--unclean_data_path', type=str)
parser.add_argument('--daily_data_path', type=str)
parser.add_argument('--hourly_data_path', type=str)
parser.add_argument('--minute_data_path', type=str)

# Data wrangling options
parser.add_argument('--extract_timeframe', type=str)
parser.add_argument('--pad_missing', type=str, default='n')

# Training options
parser.add_argument('--x_range', type=int)
parser.add_argument('--y_range', type=int)
parser.add_argument('--time_column', type=str)
parser.add_argument('--values_column', type=str)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--timeframe', type=str)
parser.add_argument('--epochs', type=int, default=10)

# Results options
parser.add_argument('--results_path', type=str, default='')


args = parser.parse_args()
