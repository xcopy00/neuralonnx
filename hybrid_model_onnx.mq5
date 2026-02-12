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

// ✅ SCALER VALUES FROM YOUR TRAINED MODEL (DO NOT CHANGE)
double scaler_min[7] = {-49.50314201927093, -47.03578528827037, -51.10300800692481, -49.50314201927093, -1.3465984122097983, -0.1997447268777992, -1.0659793814432899};
double scaler_scale[7] = {41.89359028068695, 39.76143141153081, 43.280675178532704, 41.89359028068695, 0.02717184008527788, 704.9653907578305, 962.1993127147944};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   Print("========== HYBRID AI EA INITIALIZATION ==========");
   Print("Loading ONNX model: ", ModelFile);
   
   // Create ONNX model handle (CPU-only execution, no GPU)
   onnx_handle = OnnxCreate(ModelFile, ONNX_DEFAULT);
   
   if (onnx_handle == INVALID_HANDLE) {
      Print("❌ MODEL LOAD FAILED!");
      Print("   Error Code: ", GetLastError());
      Print("   Solution: Place hybrid_model.onnx in MQL5\\Files\\ folder");
      Print("   Path: C:\\Users\\[YourName]\\AppData\\Roaming\\MetaQuotes\\Terminal\\[Number]\\MQL5\\Files\\");
      return INIT_FAILED;
   }
   
   // Set input shape for the model
   long dims[3] = {1, Timesteps, NumFeatures};
   if (!OnnxSetInputShape(onnx_handle, 0, dims)) {
      Print("❌ Failed to set input shape. Error: ", GetLastError());
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return INIT_FAILED;
   }
   
   Print("✓ Model loaded successfully!");
   Print("✓ Input shape: [1, ", Timesteps, ", ", NumFeatures, "]");
   Print("✓ Execution mode: CPU (CUDA disabled for compatibility)");
   
   // Initialize technical indicators
   h_RSI = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
   h_MACD = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
   h_ATR = iATR(_Symbol, PERIOD_CURRENT, 14);
   
   if (h_RSI == INVALID_HANDLE || h_MACD == INVALID_HANDLE || h_ATR == INVALID_HANDLE) {
      Print("❌ Failed to initialize indicators");
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return INIT_FAILED;
   }
   
   Print("✓ RSI, MACD, ATR indicators initialized");
   Print("================================================");
   Print("EA ready for trading!");
   Print("Buy Threshold: ", BuyThreshold);
   Print("Sell Threshold: ", SellThreshold);
   Print("Risk: ", RiskPercent, "%");
   Print("================================================");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   Print("EA Deinitialized. Reason: ", reason);
   
   if (onnx_handle != INVALID_HANDLE) {
      OnnxRelease(onnx_handle);
      Print("ONNX model released");
   }
   
   IndicatorRelease(h_RSI);
   IndicatorRelease(h_MACD);
   IndicatorRelease(h_ATR);
   Print("All indicators released");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   // Execute only once per bar
   if (iTime(_Symbol, PERIOD_CURRENT, 0) == last_bar) return;
   last_bar = iTime(_Symbol, PERIOD_CURRENT, 0);

   // Ensure we have enough historical data
   if (Bars(_Symbol, PERIOD_CURRENT) < Timesteps + 20) {
      Print("Not enough bars. Need: ", Timesteps + 20, " Have: ", Bars(_Symbol, PERIOD_CURRENT));
      return;
   }

   // Prepare input data array
   double inputs[];
   ArrayResize(inputs, Timesteps * NumFeatures);
   int idx = 0;
   
   // Collect OHLC + Indicators for last Timesteps bars
   for (int i = Timesteps; i > 0; i--) {
      inputs[idx++] = iOpen(_Symbol, PERIOD_CURRENT, i);
      inputs[idx++] = iHigh(_Symbol, PERIOD_CURRENT, i);
      inputs[idx++] = iLow(_Symbol, PERIOD_CURRENT, i);
      inputs[idx++] = iClose(_Symbol, PERIOD_CURRENT, i);
      
      // RSI
      double rsi[1];
      CopyBuffer(h_RSI, 0, i, 1, rsi);
      inputs[idx++] = rsi[0];
      
      // MACD
      double macd[1];
      CopyBuffer(h_MACD, MAIN_LINE, i, 1, macd);
      inputs[idx++] = macd[0];
      
      // ATR
      double atr[1];
      CopyBuffer(h_ATR, 0, i, 1, atr);
      inputs[idx++] = atr[0];
   }

   // ✅ CRITICAL FIX: Apply correct MinMaxScaler inverse transform
   // Formula: normalized_value = (raw_value - scaler_min) / scaler_scale
   for (int i = 0; i < ArraySize(inputs); i++) {
      int feature_idx = i % NumFeatures;
      inputs[i] = (inputs[i] - scaler_min[feature_idx]) / scaler_scale[feature_idx];
   }

   // Run ONNX inference
   double output[1];
   if (!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, inputs, output)) {
      Print("❌ Inference failed. Error: ", GetLastError());
      return;
   }

   double pred = output[0];
   Print("═══════════════════════════════════════════");
   Print("TIME: ", TimeToString(TimeCurrent()), " | Prediction: ", DoubleToString(pred, 4));
   Print("═══════════════════════════════════════════");

   // If we have open positions, manage trailing stops
   if (PositionsTotal() > 0) {
      ManageTrailing();
      return;
   }

   // Calculate position sizing
   double point = _Point * 10;  // Adjust for 5-digit brokers
   double sl_points = StopLossPips * point;
   double tp_points = TakeProfitPips * point;
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double lot_size = NormalizeDouble(account_balance * RiskPercent / 100 / (StopLossPips * point * 10), 2);
   
   // Ensure minimum lot size
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   lot_size = MathMax(lot_size, min_lot);
   
   // Get current Bid/Ask prices
   double ask_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // BUY SIGNAL
   if (pred > BuyThreshold && PositionsTotal() == 0) {
      double buy_sl = ask_price - sl_points;
      double buy_tp = ask_price + tp_points;
      
      bool buy_result = trade.Buy(lot_size, _Symbol, ask_price, buy_sl, buy_tp);
      
      if (buy_result) {
         Print("✓ BUY ORDER PLACED");
         Print("  Lot: ", lot_size, " | SL: ", buy_sl, " | TP: ", buy_tp);
         Print("  Prediction: ", DoubleToString(pred, 4), " (Threshold: ", BuyThreshold, ")");
      } else {
         Print("❌ BUY ORDER FAILED. Error: ", GetLastError());
      }
   }
   
   // SELL SIGNAL
   else if (pred < SellThreshold && PositionsTotal() == 0) {
      double sell_sl = bid_price + sl_points;
      double sell_tp = bid_price - tp_points;
      
      bool sell_result = trade.Sell(lot_size, _Symbol, bid_price, sell_sl, sell_tp);
      
      if (sell_result) {
         Print("✓ SELL ORDER PLACED");
         Print("  Lot: ", lot_size, " | SL: ", sell_sl, " | TP: ", sell_tp);
         Print("  Prediction: ", DoubleToString(pred, 4), " (Threshold: ", SellThreshold, ")");
      } else {
         Print("❌ SELL ORDER FAILED. Error: ", GetLastError());
      }
   }
}

//+------------------------------------------------------------------+
//| Trailing Stop Management                                         |
//+------------------------------------------------------------------+
void ManageTrailing() {
   if (!UseTrailing || PositionsTotal() == 0) return;

   double ask_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (ticket == 0) continue;

      if (!PositionSelectByTicket(ticket)) continue;

      ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double current_sl = PositionGetDouble(POSITION_SL);
      double current_tp = PositionGetDouble(POSITION_TP);
      double point = _Point * 10;

      // BUY position trailing stop
      if (pos_type == POSITION_TYPE_BUY) {
         double new_sl = bid_price - TrailingStart * point;
         if (new_sl > current_sl && new_sl > 0) {
            trade.PositionModify(ticket, new_sl, current_tp);
         }
      }
      // SELL position trailing stop
      else if (pos_type == POSITION_TYPE_SELL) {
         double new_sl = ask_price + TrailingStart * point;
         if (new_sl < current_sl) {
            trade.PositionModify(ticket, new_sl, current_tp);
         }
      }
   }
}

//+------------------------------------------------------------------+
// END OF EA
//+------------------------------------------------------------------+
