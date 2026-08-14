import pandas as pd
import numpy as np

# Load Data
events_df = pd.read_csv('data/processed/events_with_car.csv')
mc_df = pd.read_csv('data/processed/market_caps.csv')
events_df = events_df.merge(mc_df, left_on='ISSUERTRADINGSYMBOL', right_on='Ticker', how='left')

# Prepare Variables
events_df['LogTransactionValue'] = np.log(events_df['TransactionValue'])
events_df['LogFirmSize'] = np.log(events_df['MarketCap'].replace(0, np.nan))

events_df['BuyDummy'] = (events_df['TRANS_CODE'] == 'P').astype(int)
events_df['Interaction'] = events_df['BuyDummy'] * events_df['LogFirmSize']

# Dummy for Insider Role
rel = events_df['RPTOWNER_RELATIONSHIP'].fillna('')
events_df['IsCEO'] = rel.str.contains('CEO', case=False).astype(int)
events_df['IsCFO'] = rel.str.contains('CFO', case=False).astype(int)
events_df['IsDirector'] = rel.str.contains('Director', case=False).astype(int)
events_df['Is10pctOwner'] = rel.str.contains('10%', case=False).astype(int)
events_df['IsOtherOfficer'] = (rel.str.contains('Officer', case=False) & ~rel.str.contains('CEO|CFO', case=False)).astype(int)

cols = ['CAR', 'LogTransactionValue', 'LogFirmSize', 'BuyDummy', 'Interaction', 
        'IsCEO', 'IsCFO', 'IsDirector', 'Is10pctOwner', 'IsOtherOfficer']
reg_data = events_df[cols].dropna()

print("Number of observations:", len(reg_data))
print("\nVariances of variables:")
print(reg_data.var())

print("\nValue counts for dummies:")
for col in ['BuyDummy', 'IsCEO', 'IsCFO', 'IsDirector', 'Is10pctOwner', 'IsOtherOfficer']:
    print(f"\n{col}:")
    print(reg_data[col].value_counts())

print("\nCorrelation matrix:")
print(reg_data.corr().round(3))
