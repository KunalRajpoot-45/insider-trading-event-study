import pandas as pd
import yfinance as yf
import requests
import zipfile
import io
import os
import time

def download_sec_form4_data(year, quarter, output_dir="data/raw"):
    """
    Download SEC Form 4 data for a given year and quarter by scraping the SEC page.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{year}q{quarter}_form4.zip" # keep our local naming simple
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"File {filename} already exists. Skipping download.")
        return filepath
        
    page_url = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
    headers = {
        'User-Agent': 'Antigravity IDE User (antigravity@example.com)',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.sec.gov'
    }
    
    print(f"Fetching links from {page_url}...")
    page_resp = requests.get(page_url, headers=headers)
    if page_resp.status_code != 200:
        print(f"Failed to fetch SEC page. Status: {page_resp.status_code}")
        return None
        
    import re
    # Find all zip links
    zip_links = re.findall(r'href="([^"]+\.zip)"', page_resp.text)
    
    # Match the specific year and quarter
    target_pattern = f"{year}q{quarter}"
    target_link = None
    for link in zip_links:
        if target_pattern in link:
            target_link = link
            break
            
    if not target_link:
        print(f"Could not find dataset for {year} Q{quarter} on the SEC page.")
        return None
        
    if not target_link.startswith('http'):
        target_link = "https://www.sec.gov" + target_link
        
    print(f"Downloading {target_link}...")
    response = requests.get(target_link, headers=headers)
    
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
        time.sleep(0.2)
        return filepath
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")
        return None

def extract_and_load_nonderiv(zip_filepath):
    """
    Extracts and merges SUBMISSION, REPORTINGOWNER, and NONDERIV_TRANS from the zip file.
    """
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        if 'NONDERIV_TRANS.tsv' not in namelist:
            print(f"NONDERIV_TRANS.tsv not found in {zip_filepath}")
            return pd.DataFrame()
            
        with zip_ref.open('SUBMISSION.tsv') as f:
            sub_df = pd.read_csv(f, sep='\t', low_memory=False, usecols=['ACCESSION_NUMBER', 'ISSUERTRADINGSYMBOL'])
            
        with zip_ref.open('REPORTINGOWNER.tsv') as f:
            own_df = pd.read_csv(f, sep='\t', low_memory=False, usecols=['ACCESSION_NUMBER', 'RPTOWNERNAME', 'RPTOWNER_RELATIONSHIP'])
            
        with zip_ref.open('NONDERIV_TRANS.tsv') as f:
            trans_df = pd.read_csv(f, sep='\t', low_memory=False, usecols=['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_CODE', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE'])
            
        # Merge on ACCESSION_NUMBER
        df = trans_df.merge(sub_df, on='ACCESSION_NUMBER', how='left')
        df = df.merge(own_df, on='ACCESSION_NUMBER', how='left')
        
        return df

def process_sec_data(df, target_tickers):
    """
    Process SEC data: filter by tickers, open-market transactions, minimum value.
    """
    # Relevant fields
    columns = [
        'ISSUERTRADINGSYMBOL', 'RPTOWNERNAME', 'RPTOWNER_RELATIONSHIP',
        'TRANS_DATE', 'TRANS_CODE', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE'
    ]
    # Check if all columns exist in df
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns].copy()
    
    # Filter for target tickers
    if target_tickers is not None and 'ISSUERTRADINGSYMBOL' in df.columns:
        df = df[df['ISSUERTRADINGSYMBOL'].isin(target_tickers)]
        
    # Open market purchases (P) and sales (S)
    if 'TRANS_CODE' in df.columns:
        df = df[df['TRANS_CODE'].isin(['P', 'S'])]
    
    # Convert numeric columns
    if 'TRANS_SHARES' in df.columns:
        df['TRANS_SHARES'] = pd.to_numeric(df['TRANS_SHARES'], errors='coerce')
    if 'TRANS_PRICEPERSHARE' in df.columns:
        df['TRANS_PRICEPERSHARE'] = pd.to_numeric(df['TRANS_PRICEPERSHARE'], errors='coerce')
    
    # Drop NaNs in essential columns
    subset_to_dropna = [c for c in ['TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'TRANS_DATE'] if c in df.columns]
    df = df.dropna(subset=subset_to_dropna)
    
    # Calculate transaction value
    if 'TRANS_SHARES' in df.columns and 'TRANS_PRICEPERSHARE' in df.columns:
        df['TransactionValue'] = df['TRANS_SHARES'] * df['TRANS_PRICEPERSHARE']
        # Filter minimum value > 10,000 as per PDF
        df = df[df['TransactionValue'] > 10000]
    
    return df

def download_stock_prices(tickers, start_date, end_date):
    """
    Download daily adjusted closing prices using yfinance.
    """
    # Download all tickers at once
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    # We want Adjusted Close
    if 'Adj Close' in data:
        adj_close = data['Adj Close']
    else:
        adj_close = pd.DataFrame()
    return adj_close

def download_market_benchmark(start_date, end_date):
    """
    Download S&P 500 benchmark.
    """
    data = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=False)
    if 'Adj Close' in data:
        return data['Adj Close']
    return pd.Series()
