"""
Authors :
    1) Vandit Maheshwari
    2) Aviral Upadhyay
    3) Rishika Verma
"""

import DecisionTree as dt
from pprint import pprint

data = dt.pd.read_csv("dt_data.txt")#Modified data file - dt_data3.txt
testcase = dt.pd.read_csv("testcases.txt")#Given test case and additional test cases - testcases.txt

def main():
    
    total_entropy = dt.entropy_of_list(data[' Enjoy'])
    #print("Total entropy",total_entropy)
    attribute_names = list(data.columns)
    attribute_names.remove(' Enjoy')
    tree = dt.id3(data, ' Enjoy', attribute_names)
    print("")
    print("Decision Tree:")
    pprint(tree)
    testcase['precdiction'] = testcase.apply(dt.classify, axis=1, args=(tree,'Yes') )
    print("")
    print("Prediction:")
    print(testcase)

def defaultLib():
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from sklearn import preprocessing
    from sklearn import tree
    le = preprocessing.LabelEncoder()
    ddata = data.apply(le.fit_transform)
    features = list(ddata.keys())
    x = ddata[features[0:-1]]
    y = ddata[features[-1]]
    tree1 = DecisionTreeClassifier()
    tree1.fit(x, y)
    dtest = testcase.apply(le.fit_transform)
    print("Prediction made using python Library:")
    print(tree1.predict(dtest))
    tree.plot_tree(tree1.fit(x,y))

defaultLib()
main()