#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
year_df = pd.read_csv('/home/wzy/Documents/account-paper/etc/年数据.csv', encoding = 'GBK')
day_df = pd.read_csv('/home/wzy/Documents/account-paper/etc/日数据.csv', encoding = 'GBK')
day_df.head(5)
basic_df = pd.read_csv('/home/wzy/Documents/account-paper/etc/基础数据.csv', encoding = 'GBK')
year_df.info()
day_df.columns
year_df.columns
day_merge_data = day_df.groupby(by = ['股票编号', '年']).tail(1)
day_merge_data = day_merge_data[['股票编号', '年', '收盘价', '总市值', '120日平均换手率']]
day_merge_data.rename(columns = {'年':'年份（年末）'}, inplace = True)
day_merge_data
year_df = pd.merge(year_df, day_merge_data, how = 'inner', on = ['股票编号', '年份（年末）'])
day_df.info()
basic_df.info()
year_df.columns
year_df.groupby(by = '股票编号').mean().head(10)
year_df['gsz'] = year_df['每股送转']>0.5
year_df['gsz'] = year_df['gsz'].map({False:0, True:1})
date = '10月1日'
df_HighTransDate = year_df['高转送除权日']
def get_date(date, df):
    y = []
    df2str = df.fillna('缺失数据')
    for i in range(len(df2str)):
        if(df2str[i] == '缺失数据'):
            y.append(np.nan)
        elif(df2str[i]> = 'date'):
            y.append(1)
        else:
            y.append(0)
    return y
year_df['HighTransDate'] = get_date(date, df_HighTransDate)
## 处理缺失值数据
index_num = year_df.iloc[:, 0:-2].dropna(how = 'all', thresh = 8, axis = 0).index
index_num = year_df.iloc[:]
year_df_dropna = year_df.dropna(how = 'all', thresh = 12, axis = 0)
year_df_dropna.dropna(how = 'all', thresh = 300, axis = 1, inplace = True)
year_df_dropna.shape
basic_df
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
basic_df['所属行业'] = encoder.fit_transform(basic_df['所属行业'])
#basic_df_policy = basic_df['所属概念板块']
#from future_encoders import OneHotEncoder
#cat_encoder = OneHotEncoder()
#basic_df_one_hot = cat_encoder.transform(basic_df_policy)
train_data = pd.merge(year_df_dropna, basic_df, how = 'inner', on = '股票编号')
train_data.info()
def judge_new_stock(df_name):
    medium_variable = []
    for i in range(len(df_name)):
        current_feature_list = str(df_name['所属概念板块'][i]).split(';')
        if "次新股" in current_feature_list:
            medium_variable.append(1)
        else:
            medium_variable.append(0)
    return medium_variable
basic_df['是否为次新股'] = judge_new_stock(basic_df)
basic_df['是否为次新股'].value_counts()
train_data = pd.merge(year_df_dropna, basic_df, how = 'inner', on = '股票编号')
train_data['上市年限'] = train_data['上市年限'] + train_data['年份（年末）']-7
train_data = train_data[train_data['上市年限']>0]
from datetime import datetime
from dateutil.parser import parse
stamp = datetime(1900, 10, 1)
a = str(train_data['高转送除权日'][4])
a
year_df
train_data.columns
day_merge_data.columns
train_data_concrete = train_data[['基本每股收益', '每股净资产(元/股)', '每股营业收入(元/股)', '每股资本公积(元/股)', '每股未分配利润(元/股)', '每股企业自由现金流量(元/股)', '营业收入同必增长(%)', '所属行业', '是否为次新股', '上市年限', '120日平均换手率', '总市值', '收盘价', 'gsz']]
train_data_complete = train_data[['股票编号', '年份（年末）', '基本每股收益', '每股净资产(元/股)', '每股营业收入(元/股)', '每股资本公积(元/股)', '每股未分配利润(元/股)', '每股企业自由现金流量(元/股)', '营业收入同必增长(%)', '所属行业', '是否为次新股', '上市年限', 'HighTransDate', '120日平均换手率', '总市值', '收盘价', 'gsz']]
train_current = train_data_concrete.dropna(how = 'any', axis = 0)
train_current
x_traindata = train_current.iloc[:, :-1]
y_traindata = train_current[['gsz']]
x_traindata
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
x_train_data_std = std_scaler.fit_transform(x_traindata)
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x_train_data_std, y_traindata)
a = pd.DataFrame(model.feature_importances_.tolist(), index = x_traindata.columns, columns = ['importance'])
print(pd.DataFrame(model.feature_importances_.tolist(), index = x_traindata.columns, columns = ['importance']))
a['importance'].sort_values(ascending = False)
variable_list = list(a['importance'].sort_values(ascending = False).index[:7])
variable_list
x_traindata.corr()
svcSVC(C = 1.0, class_weight = 'balanced', kernel = 'linear', probability = True)
rfecvRFECV(estimator = svc,  step = 1,  cv = StratifiedKFold(2),
              scoring = 'accuracy')
X_trainScalepreprocessing.scale(x_traindata)
rfecv.fit(x_train_data_std, y_traindata)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1,  len(rfecv.grid_scores_)  +  1),  rfecv.grid_scores_)
plt.show()
def get_prediction(x_traindata, y_traindata, x_testdata, standard = 'scale'):
    if standard  ==  'scale':
        #均值方差标准化
        std_scaler = StandardScaler()
        X_trainScalestd_scaler.fit_transform(x_traindata)
        X_testScalestd_scaler.fit_transform(x_testdata)
    elif standard  == 'minmax':
        #min_max标准化
        min_max_scalerpreprocessing.MinMaxScaler()
        X_trainScalemin_max_scaler.fit_transform(x_traindata)
        X_testScalemin_max_scaler.transform(x_testdata)
    elif standard  == 'no':
        #不标准化
        X_trainScalex_traindata
        X_testScalex_testdata
    ###考虑到样本中高送转股票与非高送转股票样本的不平衡问题，这里选用调整的class_weight
    modelLogisticRegression(class_weight = 'balanced', C = 1e9)
    model.fit(X_trainScale,  y_traindata)
    predict_ymodel.predict_proba(X_testScale)

    return predict_y
def assess_classification_result(traindata, testdata, variable_list, q123_sz_data, functionget_prediction):
    x_traindatatraindata.loc[:, variable_list]
    y_traindatatraindata.loc[:, 'gsz']
    x_testdatatestdata.loc[:, variable_list]
    y_testdatatestdata.loc[:, 'gsz']
    totaltestdata.loc[:, ['股票编号', 'gsz']]
    for method in ['scale', 'minmax', 'no']:
        predict_yfunction(x_traindata, y_traindata, x_testdata, standard = method)
        total['predict_prob_' + method]predict_y[:, 1]

    ###过滤今年前期已经送转过的股票
    q123_stockq123_sz_data['股票编号'].tolist()
    total_filtertotal.loc[total['股票编号'].isin(q123_stock) == False]

    ###衡量不同选股个数、不同标准化方法下的 预测准度
    result_dict  = {}
    for stock_num in [10, 25, 50, 100, 200]:
        accuracy_list[]
        for column in total.columns[2:]:
            total.sort_values(column, inplaceTrue, ascendingFalse)
            ddtotal[:stock_num]
            accuracylen(dd[dd['gsz'] == 1])/len(dd)
            accuracy_list.append(accuracy)
        result_dict[stock_num]accuracy_list

    resultpd.DataFrame(result_dict, index  = ['accuracy_scale', 'accuracy_minmax', 'accuracy_no'])

    return result, total
train_data_complete.columns
train_data['高转送除权日'].fillna('2月1日', inplace = True)
first_year_data = train_data_complete[train_data_complete['年份（年末）'] == 1]
second_year_data = train_data_complete[train_data_complete['年份（年末）'] == 2]
train_data_exper = pd.concat([first_year_data, second_year_data], axis = 0)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 3]
variable_list = list(a['importance'].sort_values(ascending = False).index[:7])
train_data_exper.dropna(inplace = True)
test_data_exper.dropna(inplace = True)
q123_sz_data = train_data_complete[(train_data_complete['年份（年末）'] == 3)&(train_data_complete['gsz'] == 1)&(train_data_complete['HighTransDate'] == 0)]
result_3, total_3 assess_classification_result(train_data_exper, test_data_exper, variable_list, q123_sz_data)
print (result_3)
second_year_data = train_data_complete[train_data_complete['年份（年末）'] == 2]
third_year_data = train_data_complete[train_data_complete['年份（年末）'] == 3]
train_data_exper = pd.concat([second_year_data, third_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 4]
test_data_exper.dropna(inplace = True)
variable_list = list(a['importance'].sort_values(ascending = False).index[:7])
result_4, total_4 assess_classification_result(train_data_exper, test_data_exper, variable_list, q123_sz_data)
print (result_4)
train_data_exper
third_year_data = train_data_complete[train_data_complete['年份（年末）'] == 3]
fourth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 4]
train_data_exper = pd.concat([third_year_data, fourth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 5]
test_data_exper.dropna(inplace = True)
result_5, total_5 assess_classification_result(train_data_exper, test_data_exper, variable_list, q123_sz_data)
print (result_5)
fourth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 4]
fifth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 5]
train_data_exper = pd.concat([fifth_year_data, fourth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 6]
test_data_exper.dropna(inplace = True)
result_6, total_6 assess_classification_result(train_data_exper, test_data_exper, variable_list, q123_sz_data)
print (result_6)
sixth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 6]
fifth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 5]
train_data_exper = pd.concat([fifth_year_data, sixth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 7]
test_data_exper.dropna(inplace = True)
variable_list = list(a['importance'].sort_values(ascending = False).index[:7])
result_7, total_7 assess_classification_result(train_data_exper, test_data_exper, variable_list, q123_sz_data)
print (result_7)
from sklearn.svm import SVC
def get_prediction_SVM(x_traindata, y_traindata, x_testdata, standard = 'scale'):
    if standard  ==  'scale':
        #均值方差标准化
        standard_scalerpreprocessing.StandardScaler()
        X_trainScalestandard_scaler.fit_transform(x_traindata)
        X_testScalestandard_scaler.transform(x_testdata)
    elif standard  == 'minmax':
        #min_max标准化
        min_max_scalerpreprocessing.MinMaxScaler()
        X_trainScalemin_max_scaler.fit_transform(x_traindata)
        X_testScalemin_max_scaler.transform(x_testdata)
    elif standard  == 'no':
        #不标准化
        X_trainScalex_traindata
        X_testScalex_testdata
    ###考虑到样本中高送转股票与非高送转股票样本的不平衡问题，这里选用调整的class_weight

    clfSVC(C = 1.0, class_weight = 'balanced', gamma = 'auto', kernel = 'rbf', probability = True)
    clf.fit(X_trainScale,  y_traindata)
    predict_y = clf.predict_proba(X_testScale)
    return predict_y
third_year_data = train_data_complete[train_data_complete['年份（年末）'] == 3]
fourth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 4]
train_data_exper = pd.concat([third_year_data, fourth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 5]
test_data_exper.dropna(inplace = True)
result_5, total_5 assess_classification_result(train_data_exper, test_data_exper, variable_list, q123_sz_data, function = get_prediction_SVM)
print (result_5)
fourth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 4]
fifth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 5]
train_data_exper = pd.concat([fifth_year_data, fourth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 6]
test_data_exper.dropna(inplace = True)
result_6, total_6 assess_classification_result(train_data_exper, test_data_exper, variable_list, q123_sz_data, function = get_prediction_SVM)
print (result_6)
sixth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 6]
fifth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 5]
train_data_exper = pd.concat([fifth_year_data, sixth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 7]
test_data_exper.dropna(inplace = True)
variable_list = list(a['importance'].sort_values(ascending = False).index[:7])
result_7, total_7 assess_classification_result(train_data_exper, test_data_exper, variable_list, q123_sz_data, function = get_prediction_SVM)
print (result_7)
def assess_unite_logit_SVM(traindata, testdata, variable_list, q123_sz_data, method_use):
    ###Logit 部分
    traindata.dropna(inplaceTrue)
    testdata.dropna(inplaceTrue)
    x_traindatatraindata[variable_list]
    y_traindatatraindata[['gsz']]
    x_testdatatestdata[variable_list]
    y_testdatatestdata[['gsz']]

    total_logittestdata[['股票编号', 'gsz']].copy()
    for method in ['scale', 'minmax', 'no']:
        predict_yget_prediction(x_traindata, y_traindata, x_testdata, standard = method)
        total_logit['predict_prob_' + method]predict_y[:, 1]


    ###########SVM部分
    traindata.loc[traindata['gsz'] == 0, 'gsz'] = -1   #pandas>1.0.0
    testdata.loc[testdata['gsz'] == 0, 'gsz'] = -1
    #traindata.ix[traindata['gsz'] == 0, 'gsz'] = -1   #pandas<1.0.0
    #testdata.ix[testdata['gsz'] == 0, 'gsz'] = -1
    x_traindatatraindata[variable_list]
    y_traindatatraindata[['gsz']]
    x_testdatatestdata[variable_list]
    y_testdatatestdata[['gsz']]
    total_SVMtestdata[['股票编号', 'gsz']].copy()
    for method in ['scale', 'minmax', 'no']:
        predict_yget_prediction_SVM(x_traindata, y_traindata, x_testdata, standard = method)
        total_SVM['predict_prob_' + method]predict_y[:, 1]

    ###合并
    columns['股票编号', 'gsz', 'predict_prob_scale', 'predict_prob_minmax', 'predict_prob_no']
    totaltotal_logit[columns].merge(total_SVM[['股票编号', 'predict_prob_scale', 'predict_prob_minmax',                                                   'predict_prob_no']], on = ['股票编号'])
    for method in ['scale', 'minmax', 'no']:
        total['score_logit']total['predict_prob_' + method + '_x'].rank(ascendingFalse)
        total['score_SVM']total['predict_prob_' + method + '_y'].rank(ascendingFalse)
        total['score_'  +  method]total['score_logit'] + total['score_SVM']
    ###过滤今年前期已经送转过的股票
    #q123_stockq123_sz_data['stock'].tolist()
    #total_filtertotal.loc[total['stock'].isin(q123_stock) == False]

    ###过滤ST股票
    #stock_listtotal_filter['stock'].tolist()
    #st_datapd.DataFrame(get_extras('is_st', stock_list ,  start_date = date1,  end_date = date2,  df = True).iloc[-1, :])
    #st_data.columns  = ['st_signal']
    #st_listst_data[st_data['st_signal'] == True]
    #total_filtertotal_filter[total_filter['stock'].isin(st_list) == False]

    result_dict  = {}
    for stock_num in [10, 25, 50, 100, 200]:
        accuracy_list[]
        for column in ['score_scale', 'score_minmax', 'score_no']:
            total.sort_values(column, inplaceTrue, ascendingTrue)
            ddtotal[:stock_num]
            accuracylen(dd[dd['gsz'] == 1])/len(dd)
            accuracy_list.append(accuracy)
        result_dict[stock_num]accuracy_list

    resultpd.DataFrame(result_dict, index  = ['score_scale', 'score_minmax', 'score_no'])

    return result

third_year_data = train_data_complete[train_data_complete['年份（年末）'] == 3]
fourth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 4]
train_data_exper = pd.concat([third_year_data, fourth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 5]
test_data_exper.dropna(inplace = True)
result_5_svm_logit =   assess_unite_logit_SVM(train_data_exper, test_data_exper, variable_list, q123_sz_data, 'minmax')
print (result_5_svm_logit)
fourth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 4]
fifth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 5]
train_data_exper = pd.concat([fifth_year_data, fourth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 6]
test_data_exper.dropna(inplace = True)
result_6_svm_logit assess_unite_logit_SVM(train_data_exper, test_data_exper, variable_list, q123_sz_data, 'minmax')
print (result_6_svm_logit)
sixth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 6]
fifth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 5]
train_data_exper = pd.concat([fifth_year_data, sixth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 7]
test_data_exper.dropna(inplace = True)
variable_list = list(a['importance'].sort_values(ascending = False).index[:7])
result_7_svm_logit assess_unite_logit_SVM(train_data_exper, test_data_exper, variable_list, q123_sz_data, 'minmax')
print (result_7_svm_logit)
sixth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 6]
fifth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 5]
train_data_exper = pd.concat([fifth_year_data, sixth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 7]
test_data_exper.dropna(inplace = True)
variable_list = list(a['importance'].sort_values(ascending = False).index[:7])
###Logit 部分
x_traindatatrain_data_exper[variable_list]
y_traindatatrain_data_exper[['gsz']]
x_testdatatest_data_exper[variable_list]

total_logittest_data_exper[['股票编号']].copy()
method = 'scale'
predict_yget_prediction(x_traindata, y_traindata, x_testdata, standard = method)
total_logit['predict_prob_' + method]predict_y[:, 1]
from sklearn.metrics import confusion_matrix
#confusion = confusion_matrix(pre)

###########SVM部分
train_data_exper[train_data_exper['gsz'] == 0]['gsz'] = -1
x_traindatatrain_data_exper[variable_list]
y_traindatatrain_data_exper[['gsz']]
x_testdatatest_data_exper[variable_list]
total_SVMtest_data_exper[['股票编号']].copy()
method'scale'
predict_yget_prediction_SVM(x_traindata, y_traindata, x_testdata, standard = method)
total_SVM['predict_prob_' + method]predict_y[:, 1]

###合并
columns['股票编号', 'predict_prob_' + method]
totaltotal_logit[columns].merge(total_SVM[['股票编号', 'predict_prob_' + method]], on = ['股票编号'])

total['score_logit']total['predict_prob_' + method + '_x'].rank(ascendingFalse)
total['score_SVM']total['predict_prob_' + method + '_y'].rank(ascendingFalse)
total['score']total['score_logit'] + total['score_SVM']
###过滤今年前期已经送转过的股票
#q123_stockq123_sz_data['stock'].tolist()
#total_filtertotal.loc[total['stock'].isin(q123_stock) == False]
###过滤ST股票
#stock_listtotal_filter['stock'].tolist()
#st_datapd.DataFrame(get_extras('is_st', stock_list , start_date = '2017-10-25',  end_date = '2017-11-01',  df = True).iloc[-1, :])
#st_data.columns  = ['st_signal']
#st_listst_data[st_data['st_signal'] == True]
#total_filtertotal_filter[total_filter['stock'].isin(st_list) == False]
total.sort_values('score', inplaceTrue, ascendingTrue)
total.reset_index(drop = True, inplaceTrue)
total[:50]
sixth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 6]
fifth_year_data = train_data_complete[train_data_complete['年份（年末）'] == 5]
train_data_exper = pd.concat([fifth_year_data, sixth_year_data], axis = 0)
train_data_exper.dropna(inplace = True)
test_data_exper = train_data_complete[train_data_complete['年份（年末）'] == 7]
test_data_exper.dropna(inplace = True)
variable_list = list(a['importance'].sort_values(ascending = False).index[:7])
treeDecisionTreeClassifier(random_state = 0)
tree.fit(x_traindata, y_train)
print("Trainning set score:{:.3f}".format(tree.score(X_train, y_train)))
print("     Test set score:{:.3f}".format(tree.score(X_test, y_test)))
