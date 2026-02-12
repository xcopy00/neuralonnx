#property copyright "JA's Hybrid AI EA 2026"
#property version   "1.00"
#property strict
#property description "Hybrid LSTM+Attention Trading Bot"

#include <Trade\Trade.mqh>

input string ModelFile = "hybrid_model.onnx";
input int Timesteps = 60;
input int NumFeatures = 7;
input double BuyThreshold = 0.55;   // Predicted return > this = Buy
input double SellThreshold = 0.45;  // < this = Sell
input double RiskPercent = 1.0;
input int StopLossPips = 50;
input int TakeProfitPips = 100;
input bool UseTrailing = true;
input int TrailingStart = 30;

long onnx_handle = INVALID_HANDLE;
CTrade trade;
datetime last_bar = 0;
int h_RSI, h_MACD, h_ATR;

// PASTE YOUR SCALER VALUES HERE (from Colab output)
double scaler_min[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Replace!
double scaler_scale[7] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // Replace!

int OnInit() {
   Print("Loading model ", ModelFile);
   onnx_handle = OnnxCreate(ModelFile, ONNX_DEFAULT);
   if (onnx_handle == INVALID_HANDLE) {
      Print("MODEL LOAD FAILED! Error: ", GetLastError(), " - Check Files folder");
      return INIT_FAILED;
   }
   long dims[3] = {1, Timesteps, NumFeatures};
   OnnxSetInputShape(onnx_handle, 0, dims);
   Print("Model loaded. Shape: [1,", Timesteps, ",", NumFeatures, "]");

   h_RSI = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
   h_MACD = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
   h_ATR = iATR(_Symbol, PERIOD_CURRENT, 14);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   if (onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
   IndicatorRelease(h_RSI); IndicatorRelease(h_MACD); IndicatorRelease(h_ATR);
}

void OnTick() {
   if (iTime(_Symbol, PERIOD_CURRENT, 0) == last_bar) return;
   last_bar = iTime(_Symbol, PERIOD_CURRENT, 0);

   if (Bars(_Symbol, PERIOD_CURRENT) < Timesteps + 20) return;

   double inputs[];
   ArrayResize(inputs, Timesteps * NumFeatures);
   int idx = 0;
   for (int i = Timesteps; i > 0; i--) {
      inputs[idx++] = iOpen(_Symbol, PERIOD_CURRENT, i);
      inputs[idx++] = iHigh(_Symbol, PERIOD_CURRENT, i);
      inputs[idx++] = iLow(_Symbol, PERIOD_CURRENT, i);
      inputs[idx++] = iClose(_Symbol, PERIOD_CURRENT, i);
      double rsi[1]; CopyBuffer(h_RSI, 0, i, 1, rsi); inputs[idx++] = rsi[0];
      double macd[1]; CopyBuffer(h_MACD, MAIN_LINE, i, 1, macd); inputs[idx++] = macd[0];
      double atr[1]; CopyBuffer(h_ATR, 0, i, 1, atr); inputs[idx++] = atr[0];
   }

   // Apply exact scaler
   for (int i = 0; i < ArraySize(inputs); i++) {
      int f = i % NumFeatures;
      inputs[i] = inputs[i] * scaler_scale[f] + scaler_min[f];
   }

   double output[1];
   if (!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, inputs, output)) {
      Print("Inference failed: ", GetLastError());
      return;
   }
   double pred = output[0];
   Print("Prediction: ", DoubleToString(pred, 4));

   if (PositionsTotal() > 0) { ManageTrailing(); return; }

   double point = _Point * 10;  // Adjust for 5-digit brokers
   double lot = NormalizeDouble(AccountInfoDouble(ACCOUNT_BALANCE) * RiskPercent / 100 / (StopLossPips * point * 10), 2);
   lot = MathMax(lot, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));

   if (pred > BuyThreshold) {
      trade.Buy(lot, _Symbol, 0, SymbolInfoDouble(_Symbol, SYMBOL_ASK) - StopLossPips * point, 
                SymbolInfoDouble(_Symbol, SYMBOL_ASK) + TakeProfitPips * point);
   } else if (pred < SellThreshold) {
      trade.Sell(lot, _Symbol, 0, SymbolInfoDouble(_Symbol, SYMBOL_BID) + StopLossPips * point, 
                 SymbolInfoDouble(_Symbol, SYMBOL_BID) - TakeProfitPips * point);
   }
}

void ManageTrailing() {
   // Simple trailing stop logic (expand as needed)
   // ... (add your preferred trailing code)
}
