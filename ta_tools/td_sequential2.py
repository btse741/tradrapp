# td_sequential_analyzer_final.py
import psycopg2
import pandas as pd

class TDSequentialAnalyzer:
    def __init__(self, db_credentials, table_name, symbol, config=None):
        self.db_credentials = db_credentials
        self.table_name = table_name
        self.symbol = symbol
        # Default configuration with all optional features enabled
        self.config = {
            'use_price_flip': True,
            'use_perfection_condition': True,
            'use_tdst_cancellation': True,
            'use_13_vs_8_deferral': True
        }
        if config:
            self.config.update(config)
        self.data = None

    def fetch_data(self):
        """Fetches OHLC data from PostgreSQL."""
        try:
            conn = psycopg2.connect(**self.db_credentials)
            query = f"SELECT date, open, high, low, close FROM {self.table_name} WHERE symbol='{self.symbol}' ORDER BY date"
            self.data = pd.read_sql(query, conn, index_col='date', parse_dates=['date'])
            conn.close()
            return True
        except (Exception, psycopg2.Error) as error:
            print(f"Error while connecting to PostgreSQL: {error}")
            return False

    def run_analysis(self):
        """Applies TD Sequential logic with configurable options."""
        if not self.fetch_data():
            return None
        
        data = self.data
        data['td_setup_buy'] = 0
        data['td_setup_sell'] = 0
        data['td_countdown_buy'] = 0
        data['td_countdown_sell'] = 0
        data['td_setup_trend_buy'] = None
        data['td_setup_trend_sell'] = None

        state = {
            'setup_buy': 0, 'setup_sell': 0,
            'countdown_buy': 0, 'countdown_sell': 0,
            'tdst_buy': None, 'tdst_sell': None,
            'setup_buy_perfected': False, 'setup_sell_perfected': False
        }
        
        for i in range(4, len(data)):
            # --- 1. Price Flip Check ---
            buy_price_flip = False
            sell_price_flip = False
            if self.config['use_price_flip']:
                if data['close'].iloc[i-1] > data['close'].iloc[i-5] and data['close'].iloc[i] < data['close'].iloc[i-4]:
                    buy_price_flip = True
                if data['close'].iloc[i-1] < data['close'].iloc[i-5] and data['close'].iloc[i] > data['close'].iloc[i-4]:
                    sell_price_flip = True

            # --- 2. Setup Logic ---
            # Buy Setup (Bearish)
            if data['close'].iloc[i] < data['close'].iloc[i-4]:
                if state['setup_buy'] == 0 and not buy_price_flip and self.config['use_price_flip']:
                    pass
                else:
                    state['setup_buy'] += 1
                    state['setup_sell'] = 0
            else:
                state['setup_buy'] = 0
            
            # Sell Setup (Bullish)
            if data['close'].iloc[i] > data['close'].iloc[i-4]:
                if state['setup_sell'] == 0 and not sell_price_flip and self.config['use_price_flip']:
                    pass
                else:
                    state['setup_sell'] += 1
                    state['setup_buy'] = 0
            else:
                state['setup_sell'] = 0

            # Store setup counts
            data['td_setup_buy'].iloc[i] = state['setup_buy'] if state['setup_buy'] >= 9 else 0
            data['td_setup_sell'].iloc[i] = state['setup_sell'] if state['setup_sell'] >= 9 else 0
            
            # --- 3. TDST Line & Perfection Check ---
            if data['td_setup_buy'].iloc[i] == 9:
                state['tdst_buy'] = data['low'].iloc[i-8:i+1].min() # Lowest low of bars 1-9
                if self.config['use_perfection_condition']:
                    low8 = data['low'].iloc[i-1]
                    low9 = data['low'].iloc[i]
                    if (low8 <= data['low'].iloc[i-3] and low8 <= data['low'].iloc[i-2]) or \
                       (low9 <= data['low'].iloc[i-3] and low9 <= data['low'].iloc[i-2]):
                        state['setup_buy_perfected'] = True
            if data['td_setup_sell'].iloc[i] == 9:
                state['tdst_sell'] = data['high'].iloc[i-8:i+1].max() # Highest high of bars 1-9
                if self.config['use_perfection_condition']:
                    high8 = data['high'].iloc[i-1]
                    high9 = data['high'].iloc[i]
                    if (high8 >= data['high'].iloc[i-3] and high8 >= data['high'].iloc[i-2]) or \
                       (high9 >= data['high'].iloc[i-3] and high9 >= data['high'].iloc[i-2]):
                        state['setup_sell_perfected'] = True
            data['td_setup_trend_buy'].iloc[i] = state['tdst_buy']
            data['td_setup_trend_sell'].iloc[i] = state['tdst_sell']

            # --- 4. Countdown Logic (Recycle, TDST Cancel, Deferral) ---
            # Recycle (Cancellation due to new opposite setup)
            if data['td_setup_buy'].iloc[i] >= 9 and state['countdown_sell'] > 0:
                state['countdown_sell'] = 0
            if data['td_setup_sell'].iloc[i] >= 9 and state['countdown_buy'] > 0:
                state['countdown_buy'] = 0

            # TDST Violation (Cancellation due to price crossing TDST line)
            if self.config['use_tdst_cancellation']:
                if state['tdst_buy'] is not None and data['close'].iloc[i] < state['tdst_buy'] and state['countdown_buy'] > 0:
                    state['countdown_buy'] = 0
                    state['tdst_buy'] = None
                if state['tdst_sell'] is not None and data['close'].iloc[i] > state['tdst_sell'] and state['countdown_sell'] > 0:
                    state['countdown_sell'] = 0
                    state['tdst_sell'] = None

            # Increment Countdown (Note: Original rules use close vs high/low 2 bars ago for count)
            if data['td_setup_buy'].iloc[i] >= 9 and state['countdown_buy'] == 0:
                state['countdown_buy'] = 1
            elif state['countdown_buy'] > 0 and data['close'].iloc[i] <= data['low'].iloc[i-2]:
                state['countdown_buy'] += 1

            if data['td_setup_sell'].iloc[i] >= 9 and state['countdown_sell'] == 0:
                state['countdown_sell'] = 1
            elif state['countdown_sell'] > 0 and data['close'].iloc[i] >= data['high'].iloc[i-2]:
                state['countdown_sell'] += 1
            
            # Deferral Check (for 13-count)
            if state['countdown_buy'] == 13 and self.config['use_13_vs_8_deferral']:
                if data['low'].iloc[i] > data['close'].iloc[i-5]: # 13th bar low vs 8th bar close
                    print(f"TD Sequential Buy Signal Deferred (+) for {self.symbol} on {data.index[i]}.")
                    state['countdown_buy'] = 13
                else:
                    print(f"TD Sequential Buy Signal Finalized for {self.symbol} on {data.index[i]}. Setup Perfected: {state['setup_buy_perfected']}")
                    state['countdown_buy'] = 0
                    state['setup_buy_perfected'] = False
            
            if state['countdown_sell'] == 13 and self.config['use_13_vs_8_deferral']:
                if data['high'].iloc[i] < data['close'].iloc[i-5]: # 13th bar high vs 8th bar close
                    print(f"TD Sequential Sell Signal Deferred (+) for {self.symbol} on {data.index[i]}.")
                    state['countdown_sell'] = 13
                else:
                    print(f"TD Sequential Sell Signal Finalized for {self.symbol} on {data.index[i]}. Setup Perfected: {state['setup_sell_perfected']}")
                    state['countdown_sell'] = 0
                    state['setup_sell_perfected'] = False

            data['td_countdown_buy'].iloc[i] = state['countdown_buy']
            data['td_countdown_sell'].iloc[i] = state['countdown_sell']

        return data

# Example Usage
if __name__ == '__main__':
    # Configure your PostgreSQL credentials
    db_credentials = {
        "user": "your_user",
        "password": "your_password",
        "host": "your_host",
        "port": "5432",
        "database": "your_database"
    }
    
    # Configure which rules to enable/disable (set to False to disable a rule)
    analysis_config = {
        'use_price_flip': True,
        'use_perfection_condition': True,
        'use_tdst_cancellation': True,
        'use_13_vs_8_deferral': True
    }

    # Initialize and run the analyzer for a symbol
    analyzer = TDSequentialAnalyzer(db_credentials, "ohlc_data", "AAPL", config=analysis_config)
    results = analyzer.run_analysis()
    
    if results is not None:
        print(results.tail(20)) # Print the last 20 rows to see recent activity