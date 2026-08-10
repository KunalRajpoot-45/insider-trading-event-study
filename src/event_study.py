import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def estimate_market_model(firm_returns, market_returns, event_date, estimation_window=(-250, -30)):
    """
    Estimate market model regression over a clean estimation window.
    R(i,t) = alpha(i) + beta(i) * R(m,t) + epsilon(i,t)
    """
    # Align dates
    df = pd.DataFrame({'firm_ret': firm_returns, 'mkt_ret': market_returns}).dropna()
    
    # Get index of event_date
    if event_date not in df.index:
        try:
            # Find nearest date
            idx = df.index.get_indexer([event_date], method='nearest')[0]
            event_date_idx = df.index[idx]
        except Exception:
            return None, None
    else:
        event_date_idx = event_date
        
    pos = df.index.get_loc(event_date_idx)
    
    start_pos = pos + estimation_window[0]
    end_pos = pos + estimation_window[1]
    
    if start_pos < 0 or end_pos >= len(df):
        return None, None
        
    est_data = df.iloc[start_pos:end_pos+1]
    
    X = sm.add_constant(est_data['mkt_ret'])
    y = est_data['firm_ret']
    
    model = sm.OLS(y, X).fit()
    alpha = model.params.get('const', 0)
    beta = model.params.get('mkt_ret', 0)
    
    return alpha, beta

def compute_abnormal_returns(firm_returns, market_returns, event_date, alpha, beta, event_window=(-5, 5)):
    """
    Compute abnormal returns for the event window.
    AR(i,t) = R(i,t) - [alpha_hat(i) + beta_hat(i) * R(m,t)]
    """
    df = pd.DataFrame({'firm_ret': firm_returns, 'mkt_ret': market_returns}).dropna()
    
    if event_date not in df.index:
        try:
            idx = df.index.get_indexer([event_date], method='nearest')[0]
            event_date_idx = df.index[idx]
        except Exception:
            return None
    else:
        event_date_idx = event_date
        
    pos = df.index.get_loc(event_date_idx)
    
    start_pos = pos + event_window[0]
    end_pos = pos + event_window[1]
    
    if start_pos < 0 or end_pos >= len(df):
        return None
        
    event_data = df.iloc[start_pos:end_pos+1].copy()
    event_data['expected_ret'] = alpha + beta * event_data['mkt_ret']
    event_data['AR'] = event_data['firm_ret'] - event_data['expected_ret']
    
    return event_data['AR']

def run_event_study_tests(car_purchases, car_sales):
    """
    Run t-tests and Wilcoxon tests on Cumulative Abnormal Returns (CARs).
    """
    results = {}
    
    car_purchases = np.array(car_purchases)
    car_sales = np.array(car_sales)
    
    # 1. One-sample t-test
    t_stat_p, p_val_p = stats.ttest_1samp(car_purchases, 0) if len(car_purchases) > 1 else (np.nan, np.nan)
    t_stat_s, p_val_s = stats.ttest_1samp(car_sales, 0) if len(car_sales) > 1 else (np.nan, np.nan)
    
    # Confidence intervals
    def mean_confidence_interval(data, confidence=0.95):
        if len(data) < 2:
            return np.nan, np.nan, np.nan
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h
        
    mean_p, ci_low_p, ci_high_p = mean_confidence_interval(car_purchases)
    mean_s, ci_low_s, ci_high_s = mean_confidence_interval(car_sales)
    
    results['One-Sample'] = {
        'Purchases': {'Mean CAR': mean_p, 't-stat': t_stat_p, 'p-val': p_val_p, '95% CI': (ci_low_p, ci_high_p)},
        'Sales': {'Mean CAR': mean_s, 't-stat': t_stat_s, 'p-val': p_val_s, '95% CI': (ci_low_s, ci_high_s)}
    }
    
    # 2. Two-sample (Welch) t-test
    if len(car_purchases) > 1 and len(car_sales) > 1:
        t_stat_welch, p_val_welch = stats.ttest_ind(car_purchases, car_sales, equal_var=False)
    else:
        t_stat_welch, p_val_welch = np.nan, np.nan
    results['Two-Sample (Welch)'] = {'t-stat': t_stat_welch, 'p-val': p_val_welch}
    
    # 3. Non-parametric robustness check (Wilcoxon)
    w_stat_p, w_p_val_p = stats.wilcoxon(car_purchases) if len(car_purchases) > 0 else (np.nan, np.nan)
    w_stat_s, w_p_val_s = stats.wilcoxon(car_sales) if len(car_sales) > 0 else (np.nan, np.nan)
    
    results['Wilcoxon'] = {
        'Purchases': {'w-stat': w_stat_p, 'p-val': w_p_val_p},
        'Sales': {'w-stat': w_stat_s, 'p-val': w_p_val_s}
    }
    
    return results
