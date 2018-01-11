#!/usr/bin/python

import pickle
import sys

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'expenses',
                 'total_payments',
                 'bonus',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'salary'
                 ]  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## Task 2: Remove outliers
del data_dict['TOTAL']  # remove TOTAL key

for val in data_dict.values():
    if val['salary'] == 'NaN': val['salary'] = 0.0
    if val['bonus'] == 'NaN': val['bonus'] = 0.0
    if val['total_payments'] == 'NaN': val['total_payments'] = 0.0
    if val['from_this_person_to_poi'] == 'NaN': val['from_this_person_to_poi'] = 0.0
    if val['from_poi_to_this_person'] == 'NaN': val['from_poi_to_this_person'] = 0.0
    if val['shared_receipt_with_poi'] == 'NaN': val['shared_receipt_with_poi'] = 0.0
    if val['expenses'] == 'NaN': val['expenses'] = 0.0
    if val['long_term_incentive'] == 'NaN': val['long_term_incentive'] = 0.0
    if val['exercised_stock_options'] == 'NaN': val['exercised_stock_options'] = 0.0
    if val['total_payments'] == 'NaN': val['total_payments'] = 0.0
    if val['total_stock_value'] == 'NaN': val['total_stock_value'] = 0.0
    if val['deferral_payments'] == 'NaN': val['deferral_payments'] = 0.0
    if val['restricted_stock'] == 'NaN': val['restricted_stock'] = 0.0

# replace NaNs with 0.
data_dict = {k: v for k, v in data_dict.items() if True
             and (v['salary'] < 450000 or v['salary'] == 'NaN')
             and (v['expenses'] < 150000 or v['expenses'] == 'NaN')
             and (v['bonus'] < 2500000 or v['bonus'] == 'NaN')
             and (v['bonus'] > 0 or v['bonus'] == 'NaN')
             and (v['from_poi_to_this_person'] == 'NaN' or v['from_poi_to_this_person'] < 300)
             and (v['from_this_person_to_poi'] == 'NaN' or v['from_this_person_to_poi'] < 49)
             and (v['total_payments'] == 'NaN' or v['total_payments'] < 0.3e7)
             and not (v['bonus'] > 150000 and v['expenses'] > 80000)
             }

# print len(data_dict)
# import matplotlib.pyplot

# for point in data_dict.values():
#     x = point['bonus']
#     y = point['expenses']
#     if point['poi'] == 1: matplotlib.pyplot.scatter(x, y, c='red')
#     if point['poi'] == 0: matplotlib.pyplot.scatter(x, y, c='black')
#
# matplotlib.pyplot.xlabel('bonus')
# matplotlib.pyplot.ylabel("expenses")
# matplotlib.pyplot.show()


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

randomForest = RandomForestClassifier(max_depth=2,
                                      # n_estimators=2,
                                      max_features=5,
                                      min_samples_split=4,
                                      min_samples_leaf=4,
                                      max_leaf_nodes=3,
                                      oob_score=True
                                      )

# parameters = {'min_samples_leaf': [3, 4, 13, 15],
#               'min_samples_split': [2, 3, 4, 5, 10],
#               'max_features': [1, 2, 3, 4, 5, 6, 7, 'sqrt', 'log2', None],
#               'max_depth': [2, 3, 5, 10],
#               # 'n_estimators':[2,3, 4],
#               # 'max_leaf_nodes':[3,13,20,50],
#               # 'min_impurity_decrease':[0, 0.01],
#               # 'min_impurity_decrease':[.1,.2,.3,.4,1,10,50]
#               'oob_score': [True, False]
#               }

pipeRandomForest = Pipeline(steps=[('minMaxScaler', scaler), ('randomForest', randomForest)])

# clf = GridSearchCV(randomForest, parameters)
clf = pipeRandomForest

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# pca_res = pca.fit(features_train)

# clf.fit(features_train, labels_train)
# predicted_labels = clf.predict(features_test)
# print clf.best_params_
# print clf.best_score_

# from sklearn.metrics import accuracy_score

# print accuracy_score(labels_test, predicted_labels)

# from sklearn.metrics import f1_score

# print f1_score(labels_test, predicted_labels)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
