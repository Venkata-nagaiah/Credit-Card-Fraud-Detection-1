# credit-card-fraud-detection-1

**Credit Card Fraud Detection** using Combined **Random Forest (99.81%)** & **Gradient Boosting (99.85%)** Machine Learning Security Engine.

## Team
- **Team Leader**: Ambati Venkatesh
- **Team Members**: Mallapuram Venkata Rao, Nunavath Ramesh, Vineeth

## Features
- **Combined Single Prediction Output**: Evaluates transaction features across Random Forest and Gradient Boosting models to produce a single unified verdict (`FRAUD DETECTED` or `TRANSACTION LEGITIMATE`).
- **Balance Delta Verification (`OrgDelta` & `DestDelta`)**: Verifies sender and recipient before/after balances against the transaction amount.
- **Audio Buzzer Feedback**: Web Audio API synthesizer generates an audible alarm buzzer on fraud detection and a verification chime for safe transactions.
- **Real-Time Detection Logs**: Connected dataset history log with CSV export.
- **Interactive Benchmarks**: Multi-metric algorithm comparison and ROC curve visualization.

## Quick Start
```bash
pip install -r requirements.txt
python train_models.py
python app.py
```
Then open `http://127.0.0.1:5050` in your browser.
