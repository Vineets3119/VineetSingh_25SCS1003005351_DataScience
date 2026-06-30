# ============================================================
# DATA SCIENCE PROJECT - VINEET SINGH
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

print("="*50)
print("FINAL DS PROJECT RUNNING")
print("="*50)

# ========================
# STEP 1 - LOAD DATA
# ========================
df = pd.read_excel("Delhi_Pollution_Combined.xlsx", header=16)

df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.rename(columns={'From Date': 'Date'}, inplace=True)

KEEP = [c for c in ['Date','PM2.5','PM10','NO','NO2','NOx',
                     'NH3','SO2','CO','Ozone','Station']
        if c in df.columns]
df = df[KEEP].copy()

print("Columns:", list(df.columns))
print("Shape:", df.shape)

# ========================
# STEP 2 - CLEANING
# ========================
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

for col in df.columns:
    if col not in ['Date','Station']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['PM2.5'], inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)
df = df[df['PM2.5'] >= 0]

# Outlier removal
Q1, Q3 = df['PM2.5'].quantile(0.25), df['PM2.5'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['PM2.5'] >= Q1-3*IQR) & (df['PM2.5'] <= Q3+3*IQR)]
df.reset_index(drop=True, inplace=True)

print("\n===== CLEANED DATASET =====")
print("Total Records :", len(df))
print(df[['PM2.5','NO2','CO','NOx']].describe().round(2))

# ========================
# STEP 3 - FEATURES
# ========================
df['Month'] = df['Date'].dt.month.fillna(1).astype(int)

def aqi(pm):
    if pm<=30: return 'Good'
    elif pm<=60: return 'Satisfactory'
    elif pm<=90: return 'Moderate'
    elif pm<=120: return 'Poor'
    elif pm<=250: return 'Very Poor'
    else: return 'Severe'

df['AQI_Category'] = df['PM2.5'].apply(aqi)

# ========================
# STEP 4 - VISUALIZATION
# ========================
corr = np.corrcoef(df['NO2'], df['PM2.5'])[0,1]
print("\nCorrelation (NO2 vs PM2.5):", round(corr,2))

# Graph 1
plt.figure(figsize=(7,4))
plt.scatter(df['NO2'], df['PM2.5'], alpha=0.4)
m,b = np.polyfit(df['NO2'], df['PM2.5'], 1)
x = np.linspace(df['NO2'].min(), df['NO2'].max(), 100)
plt.plot(x, m*x+b, 'r-')
plt.xlabel("NO2")
plt.ylabel("PM2.5")
plt.title("NO2 vs PM2.5")
plt.tight_layout()
plt.savefig("graph1.png")
plt.show()

# Graph 2
mon = df.groupby('Month')['PM2.5'].mean()
plt.figure(figsize=(8,4))
plt.bar(mon.index, mon.values)
plt.title("Monthly PM2.5")
plt.tight_layout()
plt.savefig("graph2.png")
plt.show()

# Graph 3
plt.figure(figsize=(6,4))
df['AQI_Category'].value_counts().plot(kind='bar')
plt.title("AQI Distribution")
plt.tight_layout()
plt.savefig("graph3.png")
plt.show()

# ========================
# STEP 5 - REGRESSION
# ========================
FEAT = [f for f in ['NO2','NOx','CO','NH3','Month'] if f in df.columns]

data = df[FEAT + ['PM2.5']].dropna()
X = data[FEAT]
y = data['PM2.5']
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression().fit(Xtr, ytr)
yp_lr = lr.predict(Xte)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
yp_rf = rf.predict(Xte)

print("\nREGRESSION RESULTS")
print("LR R2:", round(r2_score(yte,yp_lr),4))
print("RF R2:", round(r2_score(yte,yp_rf),4))
print("MAE:", round(mean_absolute_error(yte,yp_rf),2))
print("RMSE:", round(np.sqrt(mean_squared_error(yte,yp_rf)),2))

cv = cross_val_score(rf, X, y, cv=5)
print("CV Score:", round(cv.mean(),4))

# Graph 4
plt.figure(figsize=(6,4))
plt.bar(['Linear','Random Forest'], [r2_score(yte,yp_lr), r2_score(yte,yp_rf)],
        color=['#3498db','#27ae60'], edgecolor='black')
plt.title("Model Comparison - R2")
plt.tight_layout()
plt.savefig("graph4.png")
plt.show()

# ========================
# STEP 6 - FEATURE IMPORTANCE (FIXED)
# ========================
fi = pd.DataFrame({'Feature':FEAT,'Importance':rf.feature_importances_}).sort_values('Importance')

plt.figure(figsize=(7,4))
plt.barh(fi['Feature'], fi['Importance'], color='steelblue', edgecolor='black')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Score")
plt.tight_layout()
plt.savefig("graph5.png")
plt.show()

# ========================
# STEP 7 - CLASSIFICATION (FIXED)
# ========================
le = LabelEncoder()
dc = df[FEAT + ['AQI_Category']].dropna()

Xc = dc[FEAT]
yc = le.fit_transform(dc['AQI_Category'])

Xtr,Xte,ytr,yte = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)

dt = DecisionTreeClassifier(max_depth=8, random_state=42).fit(Xtr,ytr)
rf_c = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr,ytr)

acc_dt = accuracy_score(yte, dt.predict(Xte))
acc_rf = accuracy_score(yte, rf_c.predict(Xte))

print("\nCLASSIFICATION")
print("DT:", round(acc_dt*100,2),"%")
print("RF:", round(acc_rf*100,2),"%")

plt.figure(figsize=(6,4))
plt.bar(['Decision Tree','RF Classifier'], [acc_dt*100, acc_rf*100],
        color=['#e74c3c','#27ae60'], edgecolor='black')
plt.title("Classification Accuracy")
plt.tight_layout()
plt.savefig("graph6.png")
plt.show()

# ========================
# STEP 8 - SIMULATION
# ========================
print("\nSIMULATION")

base_f = pd.DataFrame([{f: df[f].mean() for f in FEAT}])
base_pm = rf.predict(base_f)[0]

reds = []

for shift in [10, 20, 30]:
    sim = base_f.copy()

    sim.loc[0, 'NO2'] *= (1 - shift/100)
    sim.loc[0, 'CO'] *= (1 - shift/100)

    if 'NOx' in sim.columns:
        sim.loc[0, 'NOx'] *= (1 - shift/100)

    p = rf.predict(sim)[0]

    r = (base_pm - p) / base_pm * 100
    reds.append(r)

    print(f"{shift}% shift -> Reduction: {round(r,1)} %")
plt.figure(figsize=(7,4))
plt.bar(['10%','20%','30%'], reds,
        color=['#f39c12','#e67e22','#e74c3c'], edgecolor='black')
plt.title("Pollution Reduction Simulation")
plt.ylabel("% Reduction")
plt.tight_layout()
plt.savefig("graph7.png")
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY")