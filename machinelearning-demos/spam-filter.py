from sklearn import tree
treeclassifier = tree.DecisionTreeClassifier()
data = ["Claim your free reward", "Free cats in denmark", "information concerning your cat", "programming question"]
isspam = [1, 1, 0, 0]

treeclassifier.fit(data, isspam)
treeclassifier.predict("Free insurance claims!")