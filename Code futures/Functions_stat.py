
def test_linear_regression_assumptions(df, target_col, feature_cols=None, plot=False):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan
    from scipy.stats import shapiro
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    import seaborn as sns

    results = {}

    # Drop NA for testing
    data = df[[target_col] + (feature_cols if feature_cols else [])].dropna()

    # 1. ADF Test (stationarity)
    adf_result = adfuller(data[target_col])
    results['ADF Test Statistic'] = adf_result[0]
    results['ADF p-value'] = adf_result[1]
    results['ADF Stationary'] = adf_result[1] < 0.05

    # 2. Durbin-Watson (autocorrelation) - needs residuals
    if feature_cols:
        X = sm.add_constant(data[feature_cols])
        y = data[target_col]
        model = sm.OLS(y, X).fit()
        dw_stat = durbin_watson(model.resid)
        results['Durbin-Watson'] = dw_stat
    else:
        results['Durbin-Watson'] = 'N/A (no features provided)'

    # 3. Shapiro-Wilk (normality of target)
    shapiro_stat, shapiro_p = shapiro(data[target_col])
    results['Shapiro-Wilk Statistic'] = shapiro_stat
    results['Shapiro-Wilk p-value'] = shapiro_p
    results['Normally Distributed'] = shapiro_p > 0.05

    # 4. Breusch-Pagan (heteroscedasticity) — only if features provided
    if feature_cols:
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        results['Breusch-Pagan p-value'] = bp_test[1]
        results['Homoscedastic'] = bp_test[1] > 0.05
    else:
        results['Breusch-Pagan p-value'] = 'N/A'
        results['Homoscedastic'] = 'N/A'

    # 5. Optional plots
    if plot and feature_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(model.resid, bins=30, kde=True, ax=axes[0])
        axes[0].set_title('Residual Histogram')

        sm.qqplot(model.resid, line='s', ax=axes[1])
        axes[1].set_title('QQ Plot of Residuals')
        plt.tight_layout()
        plt.show()

    return results



def run_xgboost_regression(y, X, learning_rate=0.1, max_depth=3):
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import pandas as pd

    # Combine and drop missing values
    data = pd.concat([X, y], axis=1).dropna()
    X_clean = data[X.columns]
    y_clean = data[y.name]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    # Model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        verbosity=0  # to suppress training logs
    )
    # Fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'R2: {r2}, MSE: {mse}')
    return pd.DataFrame({'observed': y_test, 'forecast': y_pred}).sort_index(), model


def run_lasso_regression(y, X, cv=5):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import pandas as pd

    # Combine and drop missing values
    data = pd.concat([X, y], axis=1).dropna()
    X_clean = data[X.columns]
    y_clean = data[y.name]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Fit Lasso with cross-validation
    model = LassoCV(cv=cv, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best alpha (from CV): {model.alpha_:.6f}")
    print(f"R2: {r2:.4f}, MSE: {mse:.4f}")

    return pd.DataFrame({'observed': y_test, 'forecast': y_pred}).sort_index(), model


def run_pca(X, n_components=None, scale=True):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Optionally scale features
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create DataFrame with PCA components
    pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

    # Explained variance
    explained_var = pca.explained_variance_ratio_

    return pca_df, explained_var, pca
