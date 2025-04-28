if __name__ == "__main__":
    # Import semua Library
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import seaborn as sns
    from statsmodels.tsa.seasonal import seasonal_decompose
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, GRU
    # Load Data
    df = pd.read_csv('BTC-USD.csv')
    df.head()
    ##  Basic Exploration
    print(df.info())

    # Statistik dasar
    print(df.describe())

    # Cek missing values
    print(df.isnull().sum())
    ## Visualize Data

    # Pastikan kolom Date jadi tipe datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Set Date jadi index
    df.set_index('Date', inplace=True)

    # Plot harga penutupan (Close)
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Close Price')
    plt.title('Bitcoin Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot semua harga
    plt.figure(figsize=(14,7))
    plt.plot(df['Open'], label='Open')
    plt.plot(df['High'], label='High')
    plt.plot(df['Low'], label='Low')
    plt.plot(df['Close'], label='Close')
    plt.title('Bitcoin Price (Open, High, Low, Close)')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Volume transaksi
    plt.figure(figsize=(14,5))
    plt.plot(df['Volume'], color='orange', label='Volume')
    plt.title('Bitcoin Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Heatmap Korelasi antar fitur
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='Blues')
    plt.title('Correlation Heatmap')
    plt.show()

    # Seasonal Decomposition
    result = seasonal_decompose(df['Close'], model='multiplicative', period=30)  # Asumsi period bulanan
    fig = result.plot()
    fig.set_size_inches(14, 10)
    plt.show()

    # Moving Average (30 hari)
    df['Close_MA30'] = df['Close'].rolling(window=30).mean()

    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['Close_MA30'], label='30 Days Moving Average', color='red')
    plt.title('Bitcoin Close Price with 30 Days MA')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.grid(True)
    plt.show()

    ## Data Preprocessing Untuk Modelling
    # Fokus pada beberapa fitur untuk multivariate prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features]

    # 2. Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    # Membentuk dataset windowed
    sequence_length = 60  # Menggunakan 60 hari sebagai input
    forecast_horizon = 30   # Memprediksi 7 hari ke depan

    X = []
    y = []
    for i in range(sequence_length, len(scaled_data) - forecast_horizon + 1):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i:i+forecast_horizon])

    X, y = np.array(X), np.array(y)
    ## Membuat Model Deep Learning
    # Membuat model LSTM multivariate multi-step
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(RepeatVector(forecast_horizon))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(len(features))))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 5. Melatih model
    history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1)


    # 6. Prediksi
    train_pred = model.predict(X)  # (samples, forecast_horizon, features)

    # Inverse transform manual per step
    train_pred_unscaled = []
    y_unscaled = []

    for i in range(train_pred.shape[0]):
        pred_window = scaler.inverse_transform(train_pred[i])
        true_window = scaler.inverse_transform(y[i])
        train_pred_unscaled.append(pred_window)
        y_unscaled.append(true_window)

    train_pred_unscaled = np.array(train_pred_unscaled)
    y_unscaled = np.array(y_unscaled)
    plt.figure(figsize=(14, 8))
    for idx, feat in enumerate(features):
        plt.subplot(len(features), 1, idx+1)
        plt.plot(y_unscaled[:, 29, idx], label=f'Actual {feat}')
        plt.plot(train_pred_unscaled[:, 29, idx], label=f'Predicted {feat}')
        plt.legend()
        plt.title(f'{feat} Prediction - Day {29+1}')
    plt.tight_layout()
    plt.show()

    class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('mae') < 0.055 and logs.get('val_mae') < 0.055):
                    self.model.stop_training = True
    
    callbacks = myCallback()

    model = Sequential()
    model.add(GRU(64, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(RepeatVector(forecast_horizon))
    model.add(GRU(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(len(features))))

    model.compile(optimizer='adam', loss='mae', metrics=["mae"])

    #  Melatih model
    history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1,
            callbacks=callbacks,)

    #  Prediksi dan Plot seperti sebelumnya
    train_pred = model.predict(X)
    plt.figure(figsize=(14, 8))
    for idx, feat in enumerate(features):
        plt.subplot(len(features), 1, idx+1)
        plt.plot(y_unscaled[:, 29, idx], label=f'Actual {feat}')
        plt.plot(train_pred_unscaled[:, 29, idx], label=f'Predicted {feat}')
        plt.legend()
        plt.title(f'{feat} Prediction - Day {29+1}')
    plt.tight_layout()
    plt.show()
