'''
This code shows the difference between the r2_score and the adjusted r2_score with the 2 datasets .
'''

from http.client import error
import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error


class TRAINING_CASE1:
    try:
        def __init__(self, location):
            self.df = pd.read_csv(location)
            self.X = self.df.iloc[:, 0]
            self.Y = self.df.iloc[:, 1]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=99)
            self.reg = LinearRegression()

    except Exception as e:
        error_type, error_message, error_line_no = sys.exc_info()
        print(error_type, error_message, error_line_no.tb_lineno)

    def r2_score_of_case1(self):
        try:
            self.x_train = self.x_train.values.reshape(-1,1)
            self.reg.fit(self.x_train,self.y_train)
            self.y_train_predict = self.reg.predict(self.x_train)
            print(f'the r2_score of train case1:{r2_score(self.y_train, self.y_train_predict)}')

        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def adjusted_r2_of_case1(self):
        try:
            self.c = 1-(1-r2_score(self.y_train,self.y_train_predict))*(len(self.x_train)-1) / (len(self.x_train)-self.df.shape[1]-1)
            print(f'adjust_r2_Score of tarin case1:{self.c}')

        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def r2_score_of_case1_test(self):
        try:
            self.x_test = self.x_test.values.reshape(-1,1)
            self.reg.fit(self.x_test,self.y_test)
            self.y_test_predict = self.reg.predict(self.x_test)
            print(f'the r2_score of case1 test:{r2_score(self.y_test, self.y_test_predict)}')

        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def adjusted_r2_of_case1_test(self):
        try:
            self.c1 = 1-(1-r2_score(self.y_test,self.y_test_predict))*(len(self.x_test)-1) / (len(self.x_test)-self.df.shape[1]-1)
            print(f'adujest_r2_Score of test  case1:{self.c1}')

        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

class TRAINING_CASE2:
    def __init__(self,location1):
        try:
            self.DF = pd.read_csv(location1)
            self.X1 = self.DF[['YearsExperience','Height']]
            self.Y1 = self.DF.iloc[:,1]
            self.x1_train,self.x1_test,self.y1_train,self.y1_test = train_test_split(self.X1,self.Y1,test_size=0.3,random_state=99)
            self.reg1=LinearRegression()
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def train_r2_case2(self):
        try:
            self.reg1.fit(self.x1_train,self.y1_train)
            self.y1_train_predict = self.reg1.predict(self.x1_train)
            print(f'The r2 score of taraining data of case2 : {r2_score(self.y1_train,self.y1_train_predict)}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def train_adj_case2(self):
        try:
            self.a = 1-(1-r2_score(self.y1_train,self.y1_train_predict)) * (len(self.x1_train)-1) / (len(self.x1_train)-self.DF.shape[1]-1)
            print(f'the adjusted r2 score of traning data of case2:{self.a}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def test_r2_case2(self):
        try:
            self.reg1.fit(self.x1_test,self.y1_test)
            self.y1_test_predict = self.reg1.predict(self.x1_test)
            print(f'the r2 score of testing data case2 : {r2_score(self.y1_test,self.y1_test_predict)}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)

    def test_adj_case2(self):
        try:
            self.a1 = 1-(1-r2_score(self.y1_test,self.y1_test_predict)) * (len(self.x1_test)-1) / (len(self.x1_test)-self.DF.shape[1]-1)
            print(f'the adj  r2 score of testing case 2: {self.a1}')
        except Exception as e:
            error_type, error_message, error_line_no = sys.exc_info()
            print(error_type, error_message, error_line_no.tb_lineno)


if __name__ == "__main__" :
    try:
        obj = TRAINING_CASE1('C:\\Users\\Bunny\\PycharmProjects\\pythonProject\\case_1.csv')
        obj.r2_score_of_case1()
        obj.adjusted_r2_of_case1()
        obj.r2_score_of_case1_test()
        obj.adjusted_r2_of_case1_test()
        obj1 = TRAINING_CASE2('C:\\Users\\Bunny\\PycharmProjects\\pythonProject\\case_2.csv')
        obj1.train_r2_case2()
        obj1.train_adj_case2()
        obj1.test_r2_case2()
        obj1.test_adj_case2()
    except Exception as e:
        error_type, error_message, error_line_no = sys.exc_info()
        print(error_type, error_message, error_line_no.tb_lineno)
