import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


# onehot编码模型
def onehotmodel(data):
    num_col = data.select_dtypes(include=[np.number])
    non_num_col = data.select_dtypes(exclude=[np.number])
    onehotnum = pd.get_dummies(non_num_col)
    data = pd.concat([num_col, onehotnum], axis=1)
    return data


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_CUST_ID = test_data["CUST_ID"]
train_data_target = train_data["bad_good"]
train_data.drop(["bad_good"], axis=1, inplace=True)

dropcols = [
    "OPEN_ORG_NUM", "IDF_TYP_CD", "GENDER", "CUST_EUP_ACCT_FLAG", "CUST_AU_ACCT_FLAG",
    "CUST_DOLLER_FLAG", "CUST_INTERNATIONAL_GOLD_FLAG", "CUST_INTERNATIONAL_COMMON_FLAG",
    "CUST_INTERNATIONAL_SIL_FLAG", "CUST_INTERNATIONAL_DIAMOND_FLAG", "CUST_GOLD_COMMON_FLAG",
    "CUST_STAD_PLATINUM_FLAG", "CUST_LUXURY_PLATINUM_FLAG", "CUST_PLATINUM_FINANCIAL_FLAG",
    "CUST_DIAMOND_FLAG","CUST_INFINIT_FLAG", "CUST_BUSINESS_FLAG",
]
# 删除没有意义的列
train_data.drop(dropcols, axis=1, inplace=True)
test_data.drop(dropcols, axis=1, inplace=True)

# 删除train.csv中的重复行
train_data = train_data.drop_duplicates(keep="first")
# 删除NaN行
train_data.dropna(inplace=True)

# 对train.csv和test.csv数据进行Onehot编码
train_data = onehotmodel(train_data)
test_data = onehotmodel(test_data)

x_train, x_test, y_train, y_test = train_test_split(
    train_data, train_data_target, test_size=0.3
)

XGB = XGBClassifier(nthread=-1, learning_rate=0.3, max_depth=5, gamma=0,  subsample=0.9, colsample_bytree=0.5)
# nthred=-1表示使用所有的CPU核
# learning_rate:学习率
# max_depth:树的最大深度
# gamma:树的叶子节点上做进一步分区所需的最小损失函数下降值
# subsample:每棵树随机采样的比例
# colsample_bytree:每棵树随机采样的列数的占比


score = cross_val_score(XGB, train_data, train_data_target, cv=5).mean()
print("训练集上5折交叉验证分数:", score)

# 模型评价
model = XGB.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("数据的准确率: ", accuracy_score(y_test, y_pred))
print("数据的精确率: ", precision_score(y_test, y_pred))
print("数据的召回率: ", recall_score(y_test, y_pred))
print("数据的Macro-F1: ", f1_score(y_test, y_pred, average="macro"))

# 提交结果
test_pred = model.predict(test_data)
test_pred = pd.DataFrame(test_pred, columns=["bad_good"])
sub = pd.concat([test_CUST_ID, test_pred], axis=1)
sub.to_csv(r'D:\submission.csv')
