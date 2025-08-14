# Stock-Price-Prediction-A-Performance-Comparison-of-LSTM-vs.-GRU
Comparative study using LSTM and GRU models to predict stock prices, analyzing accuracy and performance on the same dataset.
技術棧

Python、TensorFlow/Keras、NumPy/Pandas、scikit-learn、pandas-ta、Matplotlib（Colab）。



目標

以 AAPL 下一交易日收盤價為目標變數，建立可迭代的預測流程與實驗框架；同時建立持續性基線（明日=今日）以避免被高自相關資料的 R²「虛高」誤導，改以 RMSE/MAE 為主做效能判讀。



流程

原始欄位：Date、Open、High、Low、Close、Adj Close、Volume。
資料切分：70/10/20（Train/Val/Test），Val/Test 向前銜接 window=60 ，避免時間斷裂。
縮放：MinMaxScaler 僅以訓練集擬合；推論後再反標準化回原尺度。
視覺化與實驗報告：訓練後自動輸出指標、超參數與早停 epoch。


關鍵問題與修復

時間連續性被破壞 → 早期 LSTM R²≈-15。改為在 Val/Test 造序列時拼接上一分割尾窗，訊號恢復。
評估口徑不一致 → 統一使用反標準化後的真值與預測計算 MSE/RMSE/MAE/R²（含 Adjusted R²）。
環境與套件：改用 tensorflow.keras 匯入、字型/legend 參數修正、pandas-ta 與 NumPy 版本相容處理。


模型調整

LSTM 基線：2 層（50→50）表現不穩；增至 100/128 並加 Dropout(0.2)，波動降低；基於實驗精神增至500 ，直接過擬合。
CNN-LSTM 假設驗證：Conv1D+Pool 在本資料增益不顯著，R²約 0.37–0.55，淘汰。
核心替換為 GRU：Stacked GRU(128→64)+Dropout(0.2/0.2)，Adam(1e-3)，EarlyStopping+Checkpoint。
結果：R²≈0.84；RMSE/MAE 顯著下降，訓練更穩定（早停約 26 epochs）。


特徵工程（由單變量到多變量）

在 Close 基礎上引入 Volume、SMA_10、SMA_20、RSI_14（pandas-ta / rolling 計算）。
多特徵輸入形狀：(window, num_features)；反標準化時以多維佈局 inverse_transform 再取 Close 一維。
控制共線風險：MA 條數循序擴展並做消融，保留帶來實質降誤差者。
固定隨機種子（NumPy/TensorFlow）、版本鎖定（TF/NumPy/pandas-ta）。
回調機制：EarlyStopping(restore_best_weights=True)、ModelCheckpoint，可選 ReduceLROnPlateau。
訓練完自動列印超參數、最佳 epoch、指標表，便於審計與橫向比較。


結論

持續性基線的 R² 會極高，但不代表可交易優勢；本專案以 RMSE/MAE 為核心指標，R²/Adjusted R² 為輔。
在此場景 GRU 優於 LSTM；效能提升主要來自資料管線修復 + 正確評估口徑 + 合理容量 + 多特徵，而非單純加深模型。


後續規劃

調參自動化：KerasTuner/Hyperband（units、dropout、LR、window）。
驗證穩健性：Walk-forward / TimeSeriesSplit。
任務重述：改預測 log-returns/差分、或做方向/分位數預測；探索 EMA/MACD/ATR、簡易宏觀特徵。
方差抑制：多種子輕量 Ensemble。
