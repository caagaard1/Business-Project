#Create a regression model on %change premium and %change DA rolling vol

#Percentage change
target_df = pd.DataFrame({'target': futures_premium_reg_df['M+1'].rolling(window = 7).mean().dropna().pct_change()})

feature_df = pd.DataFrame({
    'f1_DA_7MVA' : futures_premium_reg_df['DA MVA 7'].pct_change(),
    'f2_DA_30MVA' : futures_premium_reg_df['DA MVA 30'].pct_change(),
    'f3_DA_60MVA' : futures_premium_reg_df['DA MVA 60'].pct_change(),
    'f4_DAVol_7MVA' : futures_premium_reg_df['DA rolling vol 7'].pct_change(),
    'f5_DAVol_30MVA' : futures_premium_reg_df['DA rolling vol 30'].pct_change(),
    'f6_DAVol_60MVA' : futures_premium_reg_df['DA rolling vol 60'].pct_change(),
    'f7_FUVol_7MVA' : futures_premium_reg_df['M+1'].rolling(window=7).std().dropna().pct_change(),
    'f8_FUVol_30MVA' : futures_premium_reg_df['M+1'].rolling(window=30).std().dropna().pct_change(),
    'f9_FUVol_60MVA' : futures_premium_reg_df['M+1'].rolling(window=60).std().dropna().pct_change(),
    'f10_FU_7MVA_VAL' : futures_premium_reg_df['M+1'].rolling(window = 7).mean().dropna(),
    'f11_FU_30MVA_VAL' : futures_premium_reg_df['M+1'].rolling(window = 30).mean().dropna(),
    'f12_FU_60MVA_VAL' : futures_premium_reg_df['M+1'].rolling(window = 60).mean().dropna(),
    'f13_DA_7MVA_VAL' : futures_premium_reg_df['DA MVA 7'],
    'f14_DA_30MVA_VAL' : futures_premium_reg_df['DA MVA 30'],
    'f15_DA_60MVA_VAL' : futures_premium_reg_df['DA MVA 60'],
}).shift(1).dropna()

regression_df = target_df.join(feature_df, how = 'left').dropna()

correl_matrix_3 = regression_df.corr()

#Absolute relationship
target_df = pd.DataFrame({'target': futures_premium_reg_df['M+1, premium']})

feature_df = pd.DataFrame({
    'f1_DA_7MVA' : futures_premium_reg_df['DA MVA 7'],
    'f2_DA_30MVA' : futures_premium_reg_df['DA MVA 30'],
    'f3_DA_60MVA' : futures_premium_reg_df['DA MVA 60'],
    'f4_DAVol_7MVA' : futures_premium_reg_df['DA rolling vol 7'],
    'f5_DAVol_30MVA' : futures_premium_reg_df['DA rolling vol 30'],
    'f6_DAVol_60MVA' : futures_premium_reg_df['DA rolling vol 60'],
    'f7_FUVol_7MVA' : futures_premium_reg_df['M+1'].rolling(window=7).std().dropna(),
    'f8_FUVol_30MVA' : futures_premium_reg_df['M+1'].rolling(window=30).std().dropna(),
    'f9_FUVol_60MVA' : futures_premium_reg_df['M+1'].rolling(window=60).std().dropna(),
    'f10_FU_7MVA_VAL' : futures_premium_reg_df['M+1'].rolling(window = 7).mean().dropna(),
    'f11_FU_30MVA_VAL' : futures_premium_reg_df['M+1'].rolling(window = 30).mean().dropna(),
    'f12_FU_60MVA_VAL' : futures_premium_reg_df['M+1'].rolling(window = 60).mean().dropna(),
    'f13_DA_7MVA_VAL' : futures_premium_reg_df['DA MVA 7'],
    'f14_DA_30MVA_VAL' : futures_premium_reg_df['DA MVA 30'],
    'f15_DA_60MVA_VAL' : futures_premium_reg_df['DA MVA 60'],
}).shift(1).dropna()

regression_df = target_df.join(feature_df, how = 'left').dropna()

correl_matrix_3 = regression_df.corr()










test_linear_regression_assumptions(regression_df, 'target', feature_cols=list(feature_df.columns), plot=False)
temp, model = run_xgboost_regression(regression_df['target'], regression_df[feature_df.columns], max_depth=6, learning_rate=0.1)


pca_df, explained_var, pca_model = run_pca(X=feature_df, n_components=6)
regression_df_pca = target_df.join(pca_df, how = 'left').dropna()
temp, model = run_xgboost_regression(regression_df_pca['target'], regression_df_pca[pca_df.columns], max_depth=6, learning_rate=0.1)




pred_values = pd.DataFrame({'forecast' : model.predict(regression_df_pca[pca_df.columns])}, index=regression_df_pca.index)
temp_df = pd.DataFrame(regression_df['target'])
temp_df['forecast'] = pd.DataFrame(pred_values)

plt.figure()
plt.subplot(2,1,1)
plt.plot(temp_df, label = temp_df.columns)
plt.legend()

plt.subplot(2,1,2)
plt.plot(temp_df['target'] - temp_df['forecast'], label='residuals')
plt.show()



pred_values = pd.DataFrame({'forecast' : model.predict(regression_df[feature_df.columns])}, index=regression_df.index)
temp_df = pd.DataFrame(regression_df['target'])
temp_df['forecast'] = pd.DataFrame(pred_values)

plt.figure()
plt.subplot(2,1,1)
plt.plot(temp_df, label = temp_df.columns)
plt.legend()

plt.subplot(2,1,2)
plt.plot(temp_df['target'] - temp_df['forecast'], label='residuals')
plt.show()


'--------------------------------'


pred_values = pd.DataFrame({'forecast %': model.predict(regression_df[feature_df.columns])}, index=regression_df.index)
pred_values += 1
pred_values['t-1 value'] = pd.Series(futures_premium_reg_df['M+1'].shift(1)[futures_premium_reg_df.index >= min(regression_df.index)])
pred_values['forecast'] = pred_values['forecast %'] * pred_values['t-1 value']

observed = pd.Series(futures_premium_reg_df['M+1'][futures_premium_reg_df.index >= min(regression_df.index)])
temp_df = pd.DataFrame({'target': observed})
temp_df['forecast'] = pd.DataFrame(pred_values['forecast'])

plt.figure()
plt.subplot(2,1,1)
plt.plot(temp_df, label = temp_df.columns)
plt.legend()

plt.subplot(2,1,2)
plt.plot(temp_df['target'] - temp_df['forecast'], label='residuals')
plt.show()