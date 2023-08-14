# Boston_Housing
This project delves into the exploration and modeling of the Boston Housing dataset. It employs a combination of statistical analysis, regression modeling, and decision tree classification to extract insights and make predictions on the median house value (MEDV).

## OLSE Milti-linear regression 
<img width="677" alt="Screenshot 2023-08-14 at 3 27 05 PM" src="https://github.com/WillNaf/Boston_Housing/assets/118142412/fdd566a6-84dc-4be2-87b3-83876e4c5adf">

Overall Statistics:

RMSE: 4.735998462783738: This is the Root Mean Squared Error, a measure of the average error made by the model in predicting the dependent variable. A lower RMSE suggests a better fit to the data.
R-squared: 0.734: R-squared is the proportion of the variance in the dependent variable that is predictable from the independent variables. This means approximately 73.4% of the variability in MEDV is explained by the predictors in the model. This is relatively high, indicating a good fit.
Adj. R-squared: 0.728: This is the adjusted R-squared which accounts for the number of predictors in the model. It's slightly lower than the R-squared, but still relatively high.
F-statistic & Prob (F-statistic): These values test the overall significance of the regression model. The low Prob (F-statistic) (close to zero) suggests that the regression model predicts the dependent variable better than if we'd used the mean of the dependent variable.
Coefficients Table:

coef: These are the estimated coefficients for each predictor. For example, holding all other predictors constant, a one-unit increase in RM is associated with an increase of 3.6581 in MEDV.
std err: Standard error of the coefficient estimates. This provides a measure of the precision of the coefficient estimates.
t: The t-statistic is a ratio of the departure of the estimated value of a parameter from its hypothesized value to its standard error.
P>|t|: The p-value associated with each predictor tests the null hypothesis that the predictor's coefficient is zero (no effect). A low p-value (< 0.05) indicates that you can reject the null hypothesis.
[0.025 0.975]: These are the 95% confidence intervals for the coefficients. If the range includes zero, then that predictor is not statistically significant at the 5% level.
Other Statistics:

Omnibus & Prob(Omnibus): A test of the skewness and kurtosis of the residual (errors). A significant result suggests the residuals may not be normally distributed, which is an assumption of OLS.
Durbin-Watson: Tests for homoscedasticity. Values between 1 and 2 suggest that there is no auto-correlation in the residuals.
Jarque-Bera (JB) & Prob(JB): Another test of the normality of the residuals. A significant result indicates the residuals are not normally distributed.
Cond. No.: This provides an indication of multicollinearity in the data. The note suggests there might be strong multicollinearity, which could make some of the coefficients unreliable or unstable.
Interpretation:

Variables like INDUS and AGE have p-values greater than 0.05, suggesting that they might not be significant predictors for MEDV in the presence of other variables.
Variables like RM, NOX, DIS, PTRATIO, and LSTAT (among others) have p-values less than 0.05, making them significant predictors for the model.

## Histogram or Density Plot of Target Variable (MEDV):
Provides an insight into the distribution of the target variable.
![histogram](https://github.com/WillNaf/Boston_Housing/assets/118142412/8d4e0fe6-c4a8-4ed2-9faf-ee2bc5651b97)

## Correlation Matrix or Heatmap:
Helps in understanding the relationships between predictor variables.
![correlation_matrix](https://github.com/WillNaf/Boston_Housing/assets/118142412/d7e0f99f-877c-41b1-bd16-6527cf455808)

## Scatter Plots or Residual Plots:
For assessing the linearity assumption and spotting outliers.
Helps in understanding which features contribute the most to the model's decision-making process.
![scatter](https://github.com/WillNaf/Boston_Housing/assets/118142412/08d0d6b2-31ce-4f06-8c59-d80bb6d8e76a)


## Confusion Matrix Visualization:
Better visual representation of the confusion matrix.
Helps in visualizing the relationship between predictors and the target variable.
![confusion](https://github.com/WillNaf/Boston_Housing/assets/118142412/91841a46-ccb3-4edd-a8ca-ac1472b0f948)

## Boxplots for each Predictor vs. MEDV:
Helps in visualizing the relationship between predictors and the target variable
![box_AGE](https://github.com/WillNaf/Boston_Housing/assets/118142412/8c5e46f5-5ed3-4412-b96f-508a40107f1d)
![box_CHAS](https://github.com/WillNaf/Boston_Housing/assets/118142412/8c200fe6-420a-4f1a-9f35-3e805ca3826f)
![box_CRIM](https://github.com/WillNaf/Boston_Housing/assets/118142412/242d2ea2-e1cb-4c32-bccf-546ad78e5824)
![box_DIS](https://github.com/WillNaf/Boston_Housing/assets/118142412/fa40b5f5-8038-47a3-a6f1-29d98463083d)
![box_INDUS](https://github.com/WillNaf/Boston_Housing/assets/118142412/7cf933b5-5930-4cdc-ba0d-2492ca04aca6)
![box_LSTAT](https://github.com/WillNaf/Boston_Housing/assets/118142412/780d5292-e647-416b-9eb5-4476fbf7a22d)
![box_NOX](https://github.com/WillNaf/Boston_Housing/assets/118142412/e1d96952-75d2-41fc-bbed-b23775939c17)
![box_PTRATIO](https://github.com/WillNaf/Boston_Housing/assets/118142412/09580606-93f6-4bc7-8368-4cbeb8d1d370)
![box_RAD](https://github.com/WillNaf/Boston_Housing/assets/118142412/6c67f467-226f-4c64-841e-5eaa12c13bf2)
![box_RM](https://github.com/WillNaf/Boston_Housing/assets/118142412/62cc72c7-ad74-484f-b35c-8e522ac14633)
![box_TAX](https://github.com/WillNaf/Boston_Housing/assets/118142412/5a1e31a8-f40c-401d-adaa-a9570aea75f9)
![box_ZN](https://github.com/WillNaf/Boston_Housing/assets/118142412/1f2756ff-0021-4d18-a4bb-8d2fb96aef5d)

## Decision Tree
A machine learning algorithm that is used for both classification (labeling items into groups) and regression (predicting a numerical value). It works by recursively splitting the data into subsets based on the value of input features. 
![Tree](https://github.com/WillNaf/Boston_Housing/assets/118142412/15d667a3-c591-4422-a42a-eadd33c5c39d)

## Feature Importance (for Decision Tree):
Helps in understanding which features contribute the most to the model's decision-making process.
![tree_importance](https://github.com/WillNaf/Boston_Housing/assets/118142412/23e4840b-b64d-4b7f-aadb-d241d81828c1)

## Decision Tree Depth vs. Accuracy Plot:
Helps in understanding how the depth of the tree affects the model's accuracy and avoiding overfitting.
![tree_accuracy](https://github.com/WillNaf/Boston_Housing/assets/118142412/2b060623-e80b-4c1c-9273-93b8219ad396)

In essence, this project provides a comprehensive exploration and analysis of the Boston Housing dataset, offering both regression and classification insights. The combination of EDA, regression modeling, and decision tree classification equips stakeholders with a multifaceted understanding of housing variables' relationships and their impact on median house values.
