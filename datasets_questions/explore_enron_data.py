#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
poi_data = {k:v for k,v in enron_data.iteritems() if v["poi"] == 1 }
salaryNotNaN = {k:v for k,v in enron_data.iteritems() if v["salary"] != 'NaN' }
emailNotNaN = {k:v for k,v in enron_data.iteritems() if v["email_address"] != 'NaN' }
totalPaymentNaN = {k:v for k,v in enron_data.iteritems() if v["total_payments"] == 'NaN' }
totalPayment_POI_NaN = {k:v for k,v in enron_data.iteritems() if v["total_payments"] == 'NaN' and v["poi"] == 1 }

print len(totalPayment_POI_NaN)
print len(totalPaymentNaN)
print len(enron_data)
#print len(salaryNotNaN)
#print len(emailNotNaN)

#print len(poi_data)
#print enron_data['SKILLING JEFFREY K']['total_payments']
#print enron_data['LAY KENNETH L']['total_payments']
#print enron_data['FASTOW ANDREW S']['total_payments']
