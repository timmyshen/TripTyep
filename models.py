
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab inline')
import pandas as pd
import seaborn as sns


# In[165]:

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


# In[167]:

train = pd.read_csv('./train.csv',
                    dtype={'TripType':object,
                           'VisitNumber':int,
                           'Weekday':object,
                           'Upc':float,
                           'ScanCount':int,
                           'DepartmentDescription':object
                           })


# In[121]:

#train.TripType = train.TripType.astype('category')
#train.VisitNumber = train.VisitNumber.astype('category')
#train.Weekday = train.Weekday.astype('category')


# In[168]:

#train.head()


# In[169]:

#train.dtypes


# In[170]:

#train.shape


# In[67]:

#train.describe()


# In[78]:

#train.Weekday.describe()


# In[68]:

#visitNumCounts = train.VisitNumber.value_counts()

# In[99]:

#departmentDescriptionCounts = train.DepartmentDescription.value_counts()
#departmentDescriptionCounts.plot()


# In[96]:

#departmentDescriptionCounts.shape


# In[112]:

#finelineNumberCounts = train.FinelineNumber.value_counts()


# In[113]:

#finelineNumberCounts.shape


# In[171]:

train_sub = train.dropna()


# In[172]:

#train_sub.shape


# In[203]:

#X = train.drop(['TripType', 'Upc', 'DepartmentDescription', 'FinelineNumber'], axis=1)
X = train_sub[['VisitNumber', 'DepartmentDescription', 'Weekday', 'Upc', 'ScanCount', 'FinelineNumber']]


# In[204]:

#X.head()


# In[205]:

#X.dtypes


# In[206]:

X_dummies = pd.get_dummies(X)


# In[207]:

#X_dummies.head()


# In[208]:

y = train_sub.TripType


# In[209]:

#y.head()


# In[210]:

X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.2, random_state=42)


# In[218]:

rf_classifier = RandomForestClassifier(n_estimators=50)


# In[219]:

X_train.head()


# In[ ]:

rf_classifier = rf_classifier.fit(X_train, y_train)


# In[214]:

y_hat_prob = rf_classifier.predict_proba(X_test)


# In[215]:

y_hat_prob


# In[216]:

y_test.head()


# In[217]:

log_loss(y_test, y_hat_prob)


# In[ ]:



