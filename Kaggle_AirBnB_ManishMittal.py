import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time 

import xgboost as xgb # for xboost algorithm
from sklearn import preprocessing 
from sklearn import cross_validation
from sklearn import linear_model # for regression models
from sklearn import metrics # accuracy
from sklearn import ensemble # from random forest

# %matplotlib qt -- use this command for a separate figure windown in ipython
# %reset # to clear ipython's memory

# below two lines enable autoadjust of space for labels
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# seaborn aethetics
sns.set_context("talk")
sns.set_style("white")

print('Reading the training file ......')
dftrain = pd.read_csv('train_users_2.csv')
dfclean = dftrain.copy()

# these columns are nominal so would be treated diffrently in all plots etc
nominal = ['age','action','action_detail','action_type','secs_elapsed'] 
ordinal = ['date_account_created_year','date_account_created_month','date_account_created_day',
          'date_account_created_weekday','date_account_created','timestamp_first_active',
          'timestamp_first_active_year','timestamp_first_active_month','timestamp_first_active_day',
          'timestamp_first_active_weekday']
ignore = ['id','country_destination']

### Data cleanup for the train_users file
print('Starting data cleanup ......')
dfclean = dfclean.drop('date_first_booking',1)
# age submodule. encode missing and obviously wrong age with number 1
dfclean.loc[(dfclean['age']>100) | (dfclean['age']<14),'age'] = 1
dfclean.loc[dfclean['age'].isnull() ,'age'] = 1
dfclean['age_bins'] = np.ceil(dfclean['age']/10) # bin age in sets of 10


# split date account created into year, month, date, weekday
colname = 'date_account_created'
dfclean[colname] = pd.to_datetime(dftrain[colname],format='%Y-%m-%d')
dfclean['date_account_created_year'] = pd.DatetimeIndex(dfclean[colname]).year
dfclean['date_account_created_month'] = pd.DatetimeIndex(dfclean[colname]).month
dfclean['date_account_created_day'] = pd.DatetimeIndex(dfclean[colname]).day
dfclean['date_account_created_weekday'] = pd.DatetimeIndex(dfclean[colname]).weekday

# split time first active into year, month, date, weekday
colname = 'timestamp_first_active'
dfclean[colname] = pd.to_datetime(dftrain[colname],format='%Y%m%d%H%M%S')
dfclean[colname] = dfclean[colname].apply(lambda x: x.replace(hour=0,minute=0,second=0))
dfclean['timestamp_first_active_year'] = pd.DatetimeIndex(dfclean[colname]).year
dfclean['timestamp_first_active_month'] = pd.DatetimeIndex(dfclean[colname]).month
dfclean['timestamp_first_active_day'] = pd.DatetimeIndex(dfclean[colname]).day
dfclean['timestamp_first_active_weekday'] = pd.DatetimeIndex(dfclean[colname]).weekday

# fill nans in first_affiliate_tracked
colname = 'first_affiliate_tracked'
dfclean[colname].fillna('empty',inplace=True)

# Split data into pre-1/1/2014 and post 1/1/2014 due to sessions file
dfcleanPre = dfclean.loc[dfclean['date_account_created']<pd.to_datetime('20140101'),:]
dfcleanPost = dfclean.loc[~(dfclean['date_account_created']<pd.to_datetime('20140101')),:]

# drop some columns
dfclean = dfclean.drop('date_account_created',1)
dfclean = dfclean.drop('timestamp_first_active',1)
dfcleanPre = dfcleanPre.drop('date_account_created',1)
dfcleanPre = dfcleanPre.drop('timestamp_first_active',1)
dfcleanPost = dfcleanPost.drop('date_account_created',1)
dfcleanPost = dfcleanPost.drop('timestamp_first_active',1)
dftrain=None
dfclean=None

### Import data from the sessions file and do a join with train_users over id
newdf = pd.DataFrame()
print('Starting to read the sessions file in chunks......')
csv_chunks = pd.read_csv('sessions.csv',dtype={'user_id':np.str,'action':np.str,'action_type':np.str,'action_detail':np.str,'device_type':np.str,'secs_elapsed':np.float64},chunksize=64*1024)
i=0;
lastid = None # to check if subsequent chunk is overlapping with previous
lastgrouped = pd.Series()
ATDs = ['ArequestedTsubmitDpost_checkout_action',
'ArequestedTviewDp5',
'ApendingTbooking_requestDpending',
'AitineraryTviewDguest_itinerary',
'A12Tmessage_postDmessage_post',
'Aagree_terms_checkT-unknown-D-unknown-',
'Aphone_verification_successTclickDphone_verification_success',
'Amessage_to_host_changeTclickDmessage_to_host_change',
'Aat_checkpointTbooking_requestDat_checkpoint',
'Aqt_withTdataDlookup_message_thread',
'Amessage_to_host_focusTclickDmessage_to_host_focus',
'Atravel_plans_currentTviewDyour_trips',
'AverifyT-unknown-D-unknown-',
'Alanguages_multiselectT-unknown-D-unknown-',
'Aajax_google_translate_reviewsTclickDtranslate_listing_reviews',
'Achange_currencyT-unknown-D-unknown-',
'Aother_hosting_reviewsT-unknown-D-unknown-',
'Aajax_google_translate_descriptionT-unknown-D-unknown-',
'Atop_destinationsT-unknown-D-unknown-',
'AsaluteT-unknown-D-unknown-',
'Aother_hosting_reviews_firstT-unknown-D-unknown-',
'AshowTdataDtranslations',
'Acancellation_policiesTviewDcancellation_policies']

for dfsession in csv_chunks:
    i=i+1
    print('Reading chunk number:', i)
    # fill NaN userid by "empty". This wil anyway get discarded since no such data in training file
    dfsession = dfsession.fillna(value={'user_id':'empty'})
    
    grouped = dfsession.groupby('user_id')
    grouped_agg = grouped.aggregate({'secs_elapsed':'sum', 'action':'count', 'action_detail':'count','action_type':'count'})

    dfsession = dfsession.fillna(value={'action':'empty'})
    dfsession = dfsession.fillna(value={'action_type':'empty'})
    dfsession = dfsession.fillna(value={'action_detail':'empty'})
    dfsession['ATD'] = 'A'+dfsession.loc[:,'action']+'T'+dfsession.loc[:,'action_type']+'D'+dfsession.loc[:,'action_detail']

    grouped = pd.crosstab(dfsession['user_id'],dfsession['ATD'],values=dfsession['secs_elapsed'],aggfunc=np.sum) # crosstab to sum up secs_elapsed  of ATD by user_id
    grouped1 = pd.crosstab(dfsession['user_id'],dfsession['ATD'])    

    start_time= time.clock()
    for item in ATDs:
        try: 
            grouped_agg.loc[:,'SECS'+item] = grouped.loc[:,item]
            grouped_agg.loc[:,item] = grouped1.loc[:,item]            
        except: 
            grouped_agg.loc[:,'SECS'+item] = 0
            grouped_agg.loc[:,item] = 0            
    #print(time.clock()-start_time,"seconds")

    # if there is an overlap between this chunk and previous chunk
    if dfsession.loc[0,'user_id'] == lastid:
        grouped_agg.fillna(value=0) # just in case there is a nan
        lastgrouped.fillna(value=0) # just in case there is a nan
        grouped_agg.loc[lastid,:] += lastgrouped[:] # add the data from previous chunk
        newdf = newdf.drop(lastid,0) # drop the lastid from previous list before appending

    newdf=newdf.append(grouped_agg) # append the grouped data to create a DF

    # save the values that will be used to account for chunk overlap
    lastid = dfsession.loc[len(dfsession)-1,'user_id']
    lastgrouped = grouped_agg.loc[lastid,:]

dfcleanPost = dfcleanPost.join(newdf, how='left', on='id') # do the left join

# Fill empty numeric data from sessions with 0
dfcleanPost['action'].fillna(0,inplace=True)
dfcleanPost['action_type'].fillna(0,inplace=True)
dfcleanPost['action_detail'].fillna(0,inplace=True)
dfcleanPost['secs_elapsed'].fillna(0,inplace=True)
for col in ATDs:
    dfcleanPost.loc[:,'BIN'+col] = 1
    dfcleanPost.loc[(dfcleanPost[col]==0),'BIN'+col] = 0
    dfcleanPost[col].fillna(0,inplace=True)
    dfcleanPost['SECS'+col].fillna(0,inplace=True)

dfsession = None
grouped = None
grouped_agg = None
newdf = None
dfcleanPre = None

model='xgboost' # can replace this with 'tree' or 'reg' to try another model

collist = ['ArequestedTsubmitDpost_checkout_action',
'ArequestedTviewDp5',
'ApendingTbooking_requestDpending',
'AitineraryTviewDguest_itinerary',
'A12Tmessage_postDmessage_post',
'Aagree_terms_checkT-unknown-D-unknown-',
'Aphone_verification_successTclickDphone_verification_success',
'Amessage_to_host_changeTclickDmessage_to_host_change',
'Aat_checkpointTbooking_requestDat_checkpoint',
'Aqt_withTdataDlookup_message_thread',
'Amessage_to_host_focusTclickDmessage_to_host_focus',
'Atravel_plans_currentTviewDyour_trips',
'AverifyT-unknown-D-unknown-',
'Alanguages_multiselectT-unknown-D-unknown-',
'Aajax_google_translate_reviewsTclickDtranslate_listing_reviews',
'Achange_currencyT-unknown-D-unknown-',
'Aother_hosting_reviewsT-unknown-D-unknown-',
'Aajax_google_translate_descriptionT-unknown-D-unknown-',
'Atop_destinationsT-unknown-D-unknown-',
'AsaluteT-unknown-D-unknown-',
'Aother_hosting_reviews_firstT-unknown-D-unknown-',
'AshowTdataDtranslations',
'Acancellation_policiesTviewDcancellation_policies']
collist.extend(['SECS' + s for s in collist])
collist.extend(['date_account_created_year','secs_elapsed','action','action_detail','action_type'])

#collist = ['date_account_created_year']
### Normalize some columns
for colname in collist:    
    dfcleanPost.loc[~(dfcleanPost[colname]==0),colname] = np.log10(dfcleanPost.loc[~(dfcleanPost[colname]==0),colname])
    dfcleanPost.loc[(dfcleanPost[colname]==0),colname] = -1
    scaler = preprocessing.StandardScaler()
    dfcleanPost[colname] = scaler.fit_transform(dfcleanPost[colname])

### Preprocessing data by encoding categorical variables
collist = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider',
           'first_affiliate_tracked', 'signup_app', 'signup_flow', 'first_device_type', 
           'first_browser']
for colname in collist:
    le=preprocessing.LabelEncoder()
    le.fit(dfcleanPost[colname].unique())
    dfcleanPost.loc[:,colname] = le.transform(dfcleanPost[colname])

# one hot encode columns
for colname in collist:
    dfcleanPost = pd.concat([dfcleanPost,pd.get_dummies(dfcleanPost[colname],prefix=colname)],axis=1)
    dfcleanPost = dfcleanPost.drop(colname,axis=1)

### Split data into train and test
print('Splitting data between Test and Train and pre-processing it....')
train_X, test_X = cross_validation.train_test_split(dfcleanPost, test_size=0.2, random_state=42)
train_Y = train_X.loc[:,['country_destination']]
test_Y = test_X.loc[:,['country_destination']]

# Drop columns before training
collist = ['date_account_created_day', 'date_account_created_weekday', 'timestamp_first_active_year',
           'timestamp_first_active_month', 'timestamp_first_active_day','timestamp_first_active_weekday',
           'id','country_destination','age']
for colname in collist:
    train_X = train_X.drop(colname,axis=1)
    test_X = test_X.drop(colname,axis=1)

### Train three models: Booking vs NDF, US vs non-US, country classsfication
train_Y_save = train_Y.copy() # save a copy for future
test_Y_save = test_Y.copy() # save a copy for future

# get the boolean series for indices for all three models
colname = 'country_destination'
m1_index_NDF = (train_Y[colname] == 'NDF')
m1_index_booking = ~(m1_index_NDF)
m2_index = m1_index_booking

## Train the booking vs NDF model first
# replace country by "booking" for binary classification
colname = 'country_destination' # encode the country destination column 
train_Y.loc[m1_index_booking,colname] = 'booking' # done to complete booking vs NDF classification
le_dest_m1=preprocessing.LabelEncoder()
le_dest_m1.fit(train_Y[colname].unique())
train_Y.loc[:,colname] = le_dest_m1.transform(train_Y[colname]) # done to train the model

print('\nTraining model to differentiate between booking and NDF....\n')
if model == 'tree':
    dtc_train_m1 = ensemble.RandomForestClassifier(n_estimators=50,verbose=0)
    dtc_train_m1 = dtc_train_m1.fit(train_X,train_Y)
    feature_imp_m1 = pd.DataFrame(dtc_train_m1.feature_importances_, index=train_X.columns)
elif model == 'reg':
    dtc_train_m1 = linear_model.LogisticRegression(C=0.1,random_state=42,verbose=1)
    dtc_train_m1 = dtc_train_m1.fit(train_X,train_Y)
    feature_imp_m1 = pd.DataFrame(dtc_train_m1.coef_, columns=train_X.columns).transpose()
elif model == 'xgboost':
    params = {'eta': 0.3,'max_depth': 10,'subsample': 0.5,'colsample_bytree': 0.5,'objective': 'multi:softprob','num_class':2}
    dtrain = xgb.DMatrix(train_X, train_Y)
    dtc_train_m1 = xgb.train(params=params, dtrain=dtrain, num_boost_round=10)    

## Train the multiclass country model
train_Y = train_Y_save.copy() # retrieve the correct dataset
colname = 'country_destination'
train_Y1 = train_Y.loc[m2_index,:].copy()
le_dest_m2=preprocessing.LabelEncoder()
le_dest_m2.fit(train_Y.loc[m2_index,colname].unique())
train_Y1.loc[:,colname] = le_dest_m2.transform(train_Y1.loc[:,colname]) # encode the categorical data for training

print('\nTraining model to differentiate between countries....\n')
if model == 'tree':
    dtc_train_m2 = ensemble.RandomForestClassifier(n_estimators=50,verbose=0)
    dtc_train_m2 = dtc_train_m2.fit(train_X.loc[m2_index,:],train_Y1)
    feature_imp_m2 = pd.DataFrame(dtc_train_m2.feature_importances_, index=train_X.columns)
elif model == 'reg':
    dtc_train_m2 = linear_model.LogisticRegression(C=0.1,random_state=42,verbose=1,multi_class='ovr')
    dtc_train_m2 = dtc_train_m2.fit(train_X.loc[m2_index,:],train_Y1)
    feature_imp_m2 = pd.DataFrame(dtc_train_m2.coef_,columns=train_X.columns,index=le_dest_m2.classes_).transpose()
elif model == 'xgboost':
    params = {'eta': 0.3,'max_depth': 10,'subsample': 0.5,'colsample_bytree': 0.5,'objective': 'multi:softprob','num_class': 11}
    dtrain = xgb.DMatrix(train_X.loc[m2_index,:], train_Y1)
    dtc_train_m2 = xgb.train(params=params, dtrain=dtrain, num_boost_round=10)

train_Y1 = None
m1_index_NDF = None
m1_index_booking = None
m2_index = None
train_Y_save = None

### Run the 2 models on test data now ###

## Run the booking vs NDF model
colname = 'country_destination' 
if model == 'xgboost':
    dtc_m1_probs = dtc_train_m1.predict(xgb.DMatrix(test_X))
    dtc_test_Y_m1 = np.argmax(dtc_m1_probs,axis=1)
    dtc_test_Y_m1 = le_dest_m1.inverse_transform(dtc_test_Y_m1) # transform back to categorical variable
    dtc_test_Y_m1 = pd.DataFrame(dtc_test_Y_m1,index=test_Y.index,columns=test_Y.columns)
    dtc_m1_probs = pd.DataFrame(dtc_m1_probs,index=test_X.index,columns=le_dest_m1.classes_)    
else: 
    dtc_test_Y_m1 = dtc_train_m1.predict(test_X)
    dtc_test_Y_m1 = le_dest_m1.inverse_transform(dtc_test_Y_m1) # transform back to categorical variable
    dtc_test_Y_m1 = pd.DataFrame(dtc_test_Y_m1,index=test_Y.index,columns=test_Y.columns)
    dtc_m1_probs = dtc_train_m1.predict_proba(test_X) # prob of being in a certain class
    dtc_m1_probs = pd.DataFrame(dtc_m1_probs,index=test_X.index,columns=le_dest_m1.classes_)

# Encode the test data in same way as the training data for comparison
test_Y.loc[~(test_Y[colname] == 'NDF'),colname] = 'booking' 

# Calculate the scores and probability of classification
Score = metrics.accuracy_score(test_Y,dtc_test_Y_m1)
print('\nAccuracy on NDF vs booking model = ',Score)
print('\nConfusion matrix: ')
print(metrics.confusion_matrix(test_Y,dtc_test_Y_m1))
print(le_dest_m1.classes_,'\n\n')

# determine which entires in test set were classified to be "booking" and NDF
bkIndex_fromdtc = (dtc_test_Y_m1[colname]=='booking')

## Run the multiclass country classification model
colname = 'country_destination' 
test_Y = test_Y_save.copy()
if model == 'xgboost':
    dtc_m2_probs = dtc_train_m2.predict(xgb.DMatrix(test_X))
    dtc_test_Y_m2 = np.argmax(dtc_m2_probs,axis=1)
    dtc_test_Y_m2 = le_dest_m2.inverse_transform(dtc_test_Y_m2) # transform back to categorical variable
    dtc_test_Y_m2 = pd.DataFrame(dtc_test_Y_m2,index=test_Y.index,columns=test_Y.columns)
    dtc_m2_probs = pd.DataFrame(dtc_m2_probs,index=test_X.index,columns=le_dest_m2.classes_)     
else:    
    dtc_test_Y_m2 = dtc_train_m2.predict(test_X)
    dtc_test_Y_m2 = le_dest_m2.inverse_transform(dtc_test_Y_m2) # transform back to categorical variable
    dtc_test_Y_m2 = pd.DataFrame(dtc_test_Y_m2,index=test_Y.index,columns=test_Y.columns)
    dtc_m2_probs = dtc_train_m2.predict_proba(test_X) # prob of being in a certain class
    dtc_m2_probs = pd.DataFrame(dtc_m2_probs,index=test_X.index,columns=le_dest_m2.classes_)

dtc_test_Y_m2.loc[~(bkIndex_fromdtc),colname] = 'NDF' # combine the new results with previous model
bkIndex_fromdtc = None

# Calculate the scores and probability of classification
Score = metrics.accuracy_score(test_Y,dtc_test_Y_m2)
print('\nAccuracy of country classification model = ',Score)
print('\nConfusion matrix: ')
print(metrics.confusion_matrix(test_Y,dtc_test_Y_m2))
print(le_dest_m2.classes_,'\n\n')

### Combine the 2 models probability prediction into one matrix
print('Picking top 5 countries....')
net_probs = pd.merge(dtc_m1_probs.drop('booking',1),dtc_m2_probs.multiply(dtc_m1_probs['booking'],axis='index'),left_index=True,right_index=True)

NDCG=0
# go thorough all items and calculate the NDCG score
for i in range(0,len(net_probs)):
    #print(i)
    row = net_probs.iloc[i,:] # extract the row    
    top = list(row.sort_values(ascending=False).head(5).index.values) # take top 5 items
    top = np.array(top) # convert to array
    
    try: 
        # try to prevent error in case the actual result is not in any of the guesses. 
        ItemIndex = np.where(top==test_Y.iloc[i].values)[0][0]+1
        NDCG = NDCG + 1/(np.log2(ItemIndex+1)) # update the NDCG score based on index position
    except:
        ItemIndex = 0

NDCG = NDCG/len(net_probs)
print('\n ********** NDCG score = ',NDCG,' ***********\n')
dtc_m1_probs = None
dtc_m2_probs = None
