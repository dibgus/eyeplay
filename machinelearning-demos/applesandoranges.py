from sklearn import tree
features = [[300, 10], [200, 8000], [750, 750], [3000, 1000], [450, 20]] #fuel left and horsepower/fuel
labels = [1, 0, 0, 1, 1] #if enough gas is left
learntree = tree.DecisionTreeClassifier()
learntree = learntree.fit(features, labels)
print learntree.predict([[10, 2000], [500, 20]])