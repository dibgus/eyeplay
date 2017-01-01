#this code was written in inspiration of ep 4 of machine learning recipies
#This code does not work because the classifiers in sklearn that I am using do not support 2d arrays
from sklearn.neighbors import KNeighborsClassifier
points = [
        [[3, 5], [4, 6], [5, 7]],
        [[4, 4], [3,2], [5,6]],
        [[-1, -2], [0, -4], [-2, 0]],
        [[-7, -9], [-6, -6], [-5, -3]],
        [[0, 0], [1, 1], [2, 4]],
        [[-1, -2], [-2, -5], [-7, -9]],
        [[2, 5], [1, 3], [3, 9]]
          ]
islinear = [1, 1, 1, 1, 0, 0, 0]
treelearn = KNeighborsClassifier()
treelearn.fit(points, islinear)
treelearn.predict([[100, 100], [105, 150], [106, 110]])

