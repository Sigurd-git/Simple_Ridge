# Simple_Ridge
A simple version of ridge implemented in sklearn or torch, with nested-validation
1. Generate toy examples by example.py.
2. Inherit from sklearn.base.BaseEstimator and sklearn.base.RegressorMixin to implement a simple ridge regression class called Simple_ridge.
3. Wrap the class with sklearn.model_selection.GridSearchCV and sklearn.model_selection.KFold to perform nested-validation. 
4. Add a parameter to control whether we need to reshape a multi-dimention tensor to a matrix?