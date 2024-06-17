import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, shapiro
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('../big-cities-health/BigCitiesHealth.csv')

# Using the pivot function to reshape your data
wide_df = df.pivot_table(index='geo_label_state', columns='metric_item_label_subtitle', values='value')

wide_df.reset_index(inplace=True)
# Saving the transposed data to a new file
wide_df.to_csv('../big-cities-health/BigCitiesHealth_wide.csv', index=False)

wide_df.set_index('geo_label_state', inplace=True)

wide_df_clean = wide_df.dropna(axis=1)

# Splitting the data into features and label
X = wide_df_clean.drop(columns=['Life expectancy at birth (years, age-adjusted, 5-year estimate)'])
y = wide_df_clean['Life expectancy at birth (years, age-adjusted, 5-year estimate)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

scaler = StandardScaler()

# Fitting and transforming the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Only transforming the test data using the parameters learned from the training data
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Calculating correlation coefficients only on the training data
# Calculating correlation coefficients only on the training data
correlation_coefficients = {}
correlation_test = {}
for column in X_train.columns:
    # Checking if it's normally distributed
    stat, p_value = shapiro(X_train_scaled[column])
    if p_value > 0.05:
        # Normally distributed => Pearson's correlation
        correlation = np.corrcoef(X_train_scaled[column], y_train)[0, 1]
        correlation_type = "Pearson"
    else:
        # Not normally distributed => Spearman's rank correlation
        correlation = spearmanr(X_train_scaled[column], y_train)[0]
        correlation_type = "Spearman"
    correlation_coefficients[column] = (correlation, correlation_type)

for feature, (corr,correlation_type) in correlation_coefficients.items():
    print(f"Correlation between {feature} and label: {corr} (type: {correlation_type})")

sorted_features = sorted(correlation_coefficients.items(), key=lambda x: abs(x[1][0]), reverse=True)

print("\nTop 5 features with the highest correlation coefficients (by magnitude):")
for feature, (corr,correlation_type) in sorted_features[:5]:
    print(f"Feature: {feature}, Correlation coefficient: {corr}(type: {correlation_type})")

# Getting the top 5 features
top5 = [feature for feature, corr in sorted_features[:5]]

# Selecting the top features for X_train and X_test
X_train_best = X_train_scaled[top5]
X_test_best = X_test_scaled[top5]

######Random Forest Regression#######
# Defining the parameter grid for Random Forest
param_grid_rfr= {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initializing the grid search
grid_search_rfr = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid_rfr, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

X_train_best, X_test_best, y_train, y_test = (
    np.array(X_train_best),
    np.array(X_test_best),
    np.array(y_train),
    np.array(y_test),
)

grid_search_rfr.fit(X_train_best, y_train)
print("##### Random Forest Regression #####")
# Getting the best parameters
best_params = grid_search_rfr.best_params_
print("Best parameters:", best_params)


# Re-fitting the model with the best parameters
best_rfr = RandomForestRegressor(**best_params)
best_rfr.fit(X_train_best, y_train)

# Performing cross-validation
y_pred_cv_rfr = cross_val_predict(best_rfr, X_train_best, y_train, cv=5)

# Evaluating the model using cross-validated predictions
mae_cv_rfr = mean_absolute_error(y_train, y_pred_cv_rfr)
r2_cv_rfr = r2_score(y_train, y_pred_cv_rfr)
mse_cv_rfr = mean_squared_error(y_train, y_pred_cv_rfr)

print("Cross-validated MAE:", mae_cv_rfr)
print("Cross-validated R^2:", r2_cv_rfr)
print("Cross-validated MSE:", mse_cv_rfr)

# Predicting on the test set
y_pred_test_rfr = best_rfr.predict(X_test_best)

# Evaluating the model on the test set
mae_test_rfr = mean_absolute_error(y_test, y_pred_test_rfr)
r2_test_rfr = r2_score(y_test, y_pred_test_rfr)
mse_test_rfr = mean_squared_error(y_test, y_pred_test_rfr)

print("Test MAE:", mae_test_rfr)
print("Test R^2:", r2_test_rfr)
print("Test MSE:", mse_test_rfr)

# Plotting the results
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_cv_rfr,y_train,  color='blue', alpha=0.5, label='Predicted vs True (CV)')
plt.scatter(y_pred_test_rfr,y_test,  color='red', marker='x', alpha=0.5, label='Predicted vs True (Test)')
min_val = min(min(y_test),min(y_pred_test_rfr),min(y_train),min(y_pred_cv_rfr))
max_val = max(max(y_test),max(y_pred_test_rfr),max(y_train),max(y_pred_cv_rfr))
plt.plot([min_val,max_val], [min_val,max_val], color='green', linestyle='--', label='Perfect Prediction')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Cross-validated Predictions vs Test Predictions')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../output/rfr_pred_vs_actual.png')


####### Linear Regression #######
def linear_reg_eval(X_test,y_test,clf_LR):
    print(f'Coefficients: {clf_LR.coef_}')
    print(f'Intercept: {clf_LR.intercept_}')

    y_pred=clf_LR.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = mse**(1/2)
    mae= mean_absolute_error(y_test,y_pred)

    return r2,mse,rmse,mae,clf_LR.coef_,y_pred

#Model 1 only 5 best features
clf_LR = LinearRegression()
clf_LR.fit(X_train_best,y_train)

y_pred_cv_LR = cross_val_predict(clf_LR, X_train_best, y_train, cv=5)

print("##### Linear Regression #####")
r2_LR_cv,mse_LR_cv,rmse_LR_cv,mae_LR_cv,_,_=linear_reg_eval(X_train_best,y_pred_cv_LR,clf_LR)

print('\n----------------------------------------------------------------------------------------------------------')
print('Cross validation (best 5) Training performance:')
print(f'R squared: {r2_LR_cv:.3f} | Mean squared error: {mse_LR_cv:.3f} | Root mean squared error: {rmse_LR_cv:.3f} | Mean absolute error: {mae_LR_cv:.3f}')
print('----------------------------------------------------------------------------------------------------------\n')

r2_LR,mse_LR,rmse_LR,mae_LR,_,y_pred=linear_reg_eval(X_test_best,y_test,clf_LR)
print('\n----------------------------------------------------------------------------------------------------------')
print('Linear Regression (best 5) Test performance:')
print(f'R squared: {r2_LR:.3f} | Mean squared error: {mse_LR:.3f} | Root mean squared error: {rmse_LR:.3f} | Mean absolute error: {mae_LR:.3f}')
print('----------------------------------------------------------------------------------------------------------\n')
y_train_pred=clf_LR.predict(X_train_best)


#Visualization
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_cv_LR,y_train,  color='blue', alpha=0.5, label='Predicted vs True (CV)')
plt.scatter(y_pred, y_test,  color='red', marker='x', alpha=0.5, label='Predicted vs True (Test)')
min_val = min(min(y_test),min(y_pred),min(y_train),min(y_pred_cv_LR))
max_val = max(max(y_test),max(y_pred),max(y_train),max(y_pred_cv_LR))
plt.plot([min_val,max_val], [min_val,max_val], color='green', linestyle='--', label='Perfect Prediction')
plt.xlim(left=72, right=84)
plt.ylim(bottom=72, top=84)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Cross-validated Prediction vs Test Predictions')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../output/lr_pred_vs_actual.png')

####### Support Vector Regression #######

svr = SVR()

param_grid_svr = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1,0.2,0.3,0.4 ,0.5,0.6,0.7,0.8,0.9,1 ], #100, 1000
    'epsilon': [0.1, 0.2, 0.3,0.4,0.5,0.7, 1,2],
}
grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, scoring='neg_mean_squared_error', cv=5)
grid_search_svr.fit(X_train_best, y_train)

# Extracting the best kernel from the grid search results
best_kernel_svr = grid_search_svr.best_params_['kernel']
best_params_svr = grid_search_svr.best_params_
print("#####Support Vector Regression#####")
print(f"Best kernel: {best_kernel_svr}")
print("Best score found: ", grid_search_svr.best_score_)
print(f"Best C: {best_params_svr['C']}")
print(f"Best epsilon: {best_params_svr['epsilon']}")

# Reinitializing SVR model with the best kernel

svr_best = SVR(kernel=best_kernel_svr,C=best_params_svr['C'],epsilon=best_params_svr['epsilon'])

svr_best.fit(X_train_best, y_train)

y_pred_cv_svr = cross_val_predict(svr_best, X_train_best, y_train, cv=5)
y_pred_svr = svr_best.predict(X_test_best)


mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

mae_svr_cv = mean_absolute_error(y_train, y_pred_cv_svr)
r2_svr_cv = r2_score(y_train, y_pred_cv_svr)
mse_svr_cv = mean_squared_error(y_train, y_pred_cv_svr)
rmse_svr_cv = np.sqrt(mse_svr_cv)

print(f'Top 5 features selected: {top5}')

print("Cross-validated MAE:", mae_svr_cv)
print("Cross-validated R^2:", r2_svr_cv)
print("Cross-validated MSE:", mse_svr_cv)

print(f'Mean Squared Error (MSE): {mse_svr}')
print(f'Root Mean Squared Error (RMSE): {rmse_svr}')
print(f'Mean Absolute Error (MAE): {mae_svr}')
print(f'R-squared (R²): {r2_svr}')

plt.figure(figsize=(6, 6))
plt.scatter( y_pred_cv_svr,y_train, color='blue', alpha=0.5, label='Predicted vs True (CV)')
plt.scatter(y_pred_svr,y_test,  color='red', marker='x', alpha=0.5, label='Predicted vs True (Test)')
min_val = min(min(y_test),min(y_pred),min(y_train),min(y_pred_cv_svr))
max_val = max(max(y_test),max(y_pred),max(y_train),max(y_pred_cv_svr))
plt.plot([min_val,max_val], [min_val,max_val], color='green', linestyle='--', label='Perfect Prediction')


plt.xlim(left=72, right=84)
plt.ylim(bottom=72, top=84)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVR Predictions vs Actual Values')
plt.legend()
plt.grid(True)
plt.savefig("../output/svr_pred_vs_actual.png")

#######K-Nearest Neighbour#######

# Finding the optimal k using cross-validation
k_range = range(1, 10)
k_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train_best, y_train, cv=10, scoring='neg_mean_squared_error')
    k_scores.append(scores.mean())

# Convert negative MSE to positive
k_scores = [-score for score in k_scores]

# Plotting the results
plt.plot(k_range, k_scores)
plt.xlabel("Value of 'k'")
plt.ylabel('Mean Squared Error')
plt.title("KNN: selecting 'k' with 10-fold Cross Validation")
plt.savefig("../output/knn_k.png")

# Selecting the best k
best_k = k_range[np.argmin(k_scores)]
print(f'The optimal number of neighbors is {best_k}')

# implement and fit KNN regressor with optimal k
knn_regressor = KNeighborsRegressor(n_neighbors=best_k)
knn_regressor.fit(X_train_best, y_train)

y_pred_cv_knn = cross_val_predict(knn_regressor, X_train_best, y_train, cv=5)
y_pred_knn = knn_regressor.predict(X_test_best)

# model evaluation
mse_knn = mean_squared_error(y_test, y_pred)
r2_knn = r2_score(y_test, y_pred)
print("#####K-Nearest Neighbour#####")
print(f'Mean Squared Error: {mse_knn:.2f}')
print(f'R-squared: {r2_knn:.2f}')

mse_cv_knn = mean_squared_error(y_train, y_pred_cv_knn)
r2_cv_knn = r2_score(y_train, y_pred_cv_knn)
print(f'Mean Squared Error cv: {mse_cv_knn:.2f}')
print(f'R-squared cv: {r2_cv_knn:.2f}')

plt.figure(figsize=(6, 6)) 
plt.scatter(y_pred_cv_knn, y_train, color='blue', alpha=0.5, label='Predicted vs True (CV)')
plt.scatter(y_pred_knn,y_test,  color='red', marker='x', alpha=0.5, label='Predicted vs True (Test)')
plt.title('Cross-validated Prediction vs Test Predictions')
plt.xlabel('Predicted')
plt.ylabel('True')
min_val = min(min(y_test),min(y_pred),min(y_train),min(y_pred_cv_knn))
max_val = max(max(y_test),max(y_pred),max(y_train),max(y_pred_cv_knn))
plt.plot([min_val,max_val], [min_val,max_val], color='green', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.axis('equal')
plt.grid('true')
plt.savefig("../output/knn_pred_vs_actual.png")
