
import math
import scipy.signal
import torch

#정규화 및 전처리 계산
from sklearn.preprocessing import MinMaxScaler

#시각화및 저장,계산
import numpy as np
import pandas as pd
# 시각화및 저장,계산
import numpy as np
import pandas as pd
import scipy.signal
import torch
import matplotlib.pyplot as plt

# 정규화 및 전처리 계산
from sklearn.preprocessing import MinMaxScaler
import e_train as params
import psycopg2



#API및 데이터 불러오기
#기타
#크롤링



class ind_env:
    def __init__(self):
        self.MD=0
        pass


    def AccumN(self,data,period): # 시리즈 데이터의 기간만큼 누적합 한다
        data= pd.Series(data).rolling(period).sum()
        data= data.dropna().reset_index()[0]
        return data


    def period_equal(self,data): #데이터의 기간을 일치 시켜준다
        len_data=[] #길이저장
        res=[] #결과저장

        for data_dim in range(len(data)): # start_period 찾는다
            len_data.append(len(data[data_dim])) # 데이터 길이 저장
        start_period=min(len_data)

        for data_dim in range(len(data)): #기간 일치
            res_data=data[data_dim][-start_period:]
            res.append(res_data)

        return res #들어온 순서대로 출력



    def save_ind(self,data):
        #커스텀 지표 저장소
        pass







































