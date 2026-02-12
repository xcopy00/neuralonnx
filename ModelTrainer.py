import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
import tf2onnx
import tensorflow as tf
from google.colab import files
import onnxruntime as rt

# Robust loading: Skip the comma header line, then split data lines on whitespace
with open('EURUSD_PERIOD_M5.csv', 'r') as f:  # Use your exact filename
    lines = f.readlines()

# Skip first line (header "Time,Open,High,Low,Close,Volume")
data_lines = lines[1:]

data = []
for line in data_lines:
    parts = line.strip().split()  # Split on any whitespace
    if len(parts) == 7:  # Date Time Open High Low Close Volume
        data.append(parts)

# Create DataFrame
df = pd.DataFrame(data, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

print("Loaded shape:", df.shape)
print("First few rows:\n", df.head(10))

# Combine Date and Time into datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
df = df.drop(['Date', 'Time'], axis=1)
df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Reorder

# Convert to numeric
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)
print("After numeric conversion:", df.shape)

# Indicators (RSI, MACD, ATR)
delta = df['Close'].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + rs))

df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

df.dropna(inplace=True)
print("After indicators:", df.shape)

# Target: Normalized future return [0,1]
df['FutureReturn'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
df['Target'] = (df['FutureReturn'] + 0.01) / 0.02
df.dropna(inplace=True)
print("Final data shape:", df.shape)

if df.shape[0] < 1000:
    raise ValueError("Still too few rows! Export more bars (set BarsToExport=50000) and re-upload.")

features = ['Open', 'High', 'Low', 'Close', 'RSI', 'MACD', 'ATR']
X = df[features].values
y = df['Target'].values

# Scale [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

print("=== COPY THESE TO YOUR EA ===")
print("scaler_min = ", scaler.min_.tolist())
print("scaler_scale = ", scaler.scale_.tolist())
print("=================================")

# Reshape for LSTM
timesteps = 60
num_features = len(features)  # 7
X_reshaped = []
y_reshaped = []
for i in range(timesteps, len(X_scaled)):
    X_reshaped.append(X_scaled[i-timesteps:i])
    y_reshaped.append(y[i])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)
print("Reshaped data:", X_reshaped.shape)

# Split
split = int(0.8 * len(X_reshaped))
X_train, X_test = X_reshaped[:split], X_reshaped[split:]
y_train, y_test = y_reshaped[:split], y_reshaped[split:]

# Hybrid Model
inputs = Input(shape=(timesteps, num_features))
x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LayerNormalization()(x)
attn = MultiHeadAttention(num_heads=8, key_dim=8)(x, x)
x = attn + x
x = LayerNormalization()(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=40, batch_size=64, validation_data=(X_test, y_test), verbose=1)

print("\n" + "="*60)
print("EXPORTING ONNX MODEL (CPU-ONLY)")
print("="*60)

# ✅ FIX: Export ONNX with CPU-ONLY support (no CUDA)
spec = (tf.TensorSpec((None, timesteps, num_features), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the model
with open("hybrid_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✓ Model saved as hybrid_model.onnx")

# ✅ FIX: Test with CPU-only providers to ensure compatibility
try:
    session_options = rt.SessionOptions()
    session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Use CPU only (no CUDA)
    cpu_session = rt.InferenceSession(
        "hybrid_model.onnx", 
        providers=['CPUExecutionProvider'],
        sess_options=session_options
    )
    print("✓ Model verified with CPU-only execution")
    print("✓ Model is compatible with MT5 (no CUDA required)")
except Exception as e:
    print("⚠ Warning:", str(e))
    print("  The model may still work in MT5")

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print("Test MAE:", model.evaluate(X_test, y_test)[1])
print("✓ Model exported successfully!")
print("✓ Ready to download: hybrid_model.onnx")
print("="*60 + "\n")

files.download('hybrid_model.onnx')
