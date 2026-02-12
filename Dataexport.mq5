#property copyright "JA's Data Exporter 2026"
#property version   "1.00"
#property script_show_inputs

input string InSymbol = "EURUSD";
input ENUM_TIMEFRAMES Timeframe = PERIOD_M5;
input int BarsToExport = 50000;  // More bars = better training data

void OnStart()
{
   Print("========== DATA EXPORT STARTED ==========");
   Print("Symbol: ", InSymbol);
   Print("Timeframe: ", EnumToString(Timeframe));
   Print("Bars to export: ", BarsToExport);
   
   // Create filename with date format matching trainer
   string filename = InSymbol + "_" + EnumToString(Timeframe) + ".csv";
   
   // Open file for writing
   int file = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(file == INVALID_HANDLE)
   {
      Print("❌ FILE OPEN FAILED! Error: ", GetLastError());
      Print("   Solution: Ensure MQL5\\Files\\ folder exists");
      Print("   Path: C:\\Users\\[YourName]\\AppData\\Roaming\\MetaQuotes\\Terminal\\[Number]\\MQL5\\Files\\");
      return;
   }

   Print("✓ File opened successfully");

   // Write header with correct format (matches trainer expectations)
   // Format: Date Time Open High Low Close Volume
   FileWrite(file, "Date,Time,Open,High,Low,Close,Volume");
   Print("✓ Header written");

   int bars_written = 0;
   int bars_failed = 0;

   // Export bars from newest to oldest
   for(int i = BarsToExport - 1; i >= 0; i--)
   {
      // Get OHLCV data
      datetime time = iTime(InSymbol, Timeframe, i);
      double open_price = iOpen(InSymbol, Timeframe, i);
      double high = iHigh(InSymbol, Timeframe, i);
      double low = iLow(InSymbol, Timeframe, i);
      double close_price = iClose(InSymbol, Timeframe, i);
      long volume = iVolume(InSymbol, Timeframe, i);

      // Validate data
      if(time == 0 || open_price == 0 || close_price == 0)
      {
         bars_failed++;
         continue;
      }

      // Format date and time separately (YYYY.MM.DD HH:MM)
      string date_str = TimeToString(time, TIME_DATE);      // YYYY.MM.DD format
      string time_str = TimeToString(time, TIME_MINUTES);   // HH:MM format

      // Write CSV row with proper formatting
      FileWrite(file, 
                date_str,                                    // Date
                time_str,                                    // Time
                DoubleToString(open_price, _Digits),        // Open
                DoubleToString(high, _Digits),              // High
                DoubleToString(low, _Digits),               // Low
                DoubleToString(close_price, _Digits),       // Close
                IntegerToString(volume));                    // Volume

      bars_written++;

      // Progress indicator every 5000 bars
      if(bars_written % 5000 == 0)
      {
         Print("   Exported ", bars_written, " bars...");
      }
   }

   FileClose(file);

   Print("========== DATA EXPORT COMPLETED ==========");
   Print("✓ File: ", filename);
   Print("✓ Bars exported: ", bars_written);
   Print("✓ Bars skipped: ", bars_failed);
   Print("✓ Total bars: ", bars_written + bars_failed);
   Print("✓ Location: MQL5\\Files\\", filename);
   Print("=========================================");
   Print("Next steps:");
   Print("1. Go to File → Open Data Folder");
   Print("2. Navigate to MQL5\\Files\\");
   Print("3. Download ", filename);
   Print("4. Upload to Colab");
   Print("5. Run trainer script");
   Print("=========================================");
}
