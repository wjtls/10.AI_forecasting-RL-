
import datetime
from datetime import datetime
import ccxt

# 시각화및 저장,계산
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import torch
# 정규화 및 전처리 계산
from sklearn.preprocessing import MinMaxScaler

# 외부 py 호출
import a_ind as ind_
import e_train as params
import time
import requests

# API및 데이터 불러오기
# 크롤링

np.random.seed(1)


def save_price_data(data,ind_data,data_name):
    data_name = data_name.replace('/', '_')
    data = [pd.Series(data[step],name='data').reset_index()['data'] for step in range(len(data))]

    data_ = pd.DataFrame(data[:-1])
    date = pd.DataFrame(data[-1])
    data_minute = pd.DataFrame([params.minute])

    date_name_ = str('save_date_'+data_name)
    data_name_ = str('save_price_'+data_name)
    minute_name = 'minute_data'

    date.to_csv(date_name_,index=False)
    data_.to_csv(data_name_,index=False)
    torch.save(ind_data,'save_ind_'+data_name)
    data_minute.to_csv(minute_name,index=False)

def load_price_data(data_name):  #불러온 데이터를 csv로 저장하고 동일한 날짜인경우 불러올때 csv를 호출함으로써 시간 절약
    res = 0
    data = 0
    data_name = data_name.replace('/', '_')
    ind_data_ = [0, 0, 0]

    try:
        csv_data=pd.read_csv('save_date_'+data_name).values
        ind_data = torch.load('save_ind_'+data_name)
        past_minute =pd.read_csv('minute_data').values[0][0] #과거 분봉

        if params.minute == past_minute and params.data_count[0] == csv_data[0][0][:16] and params.data_count[1] == csv_data[-1][0][:16]:
            res='csv'
            csv_data_ = pd.read_csv('save_price_'+data_name).values

            data = [pd.Series(csv_data_[step]) for step in range(len(csv_data_))]
            data.append(pd.Series(csv_data.reshape(-1))) # 날짜 추가
            ind_data_ = [pd.Series(ind_data[step]) for step in range(len(ind_data))]
        else :
            #API와 저장된 데이터의 불러올 날짜가 다르면 새로 API 호출
            print('불러온 데이터의 마지막 날짜:',csv_data[-1][0][:16],'         사용자가 설정한 날짜:',params.data_count[1])
            res='API'

    except:
        print('저장된 데이터 파일이 없습니다. 새로운 API 호출 실시')
        res= 'API'

    return res,data,ind_data_


class Env:
    def __init__(self):
        self.coin_or_stock = 0
        self.close_=0
        self.open=0
        self.vol_=0
        self.date_=0

    def stock_select(self, coin_or_stock):  # 종목을 정하는 함수
        # 현재는 비트코인으로 수동설정이며 향후 주식,코인을 통틀어 알고리즘에 의한 종목 선택을 할예정
        name_=0
        if coin_or_stock == 'future':
            name_=params.stock_name

        return name_


    def unix_date(self, date):  # 날짜를 유닉스로 변경
        epoch = datetime(1965, 1, 1)  # 유닉스 기준일
        t = datetime.strptime(date, '%Y-%m-%d')
        diff = t - epoch

        return (diff.days * 24 * 3600 + diff.seconds)


    def coin_total_time_Frame(self, data, minute,col_name):  # all data 분봉 출력( traning O  , real backtest x , real trading x)

        # 분봉출력: 실시간 시뮬레이터, 학습 = data count 만큼(또는 전체) 를 뽑아서 연산
        #       실전 = 일부만 뽑아서 전체를뽑은것과 같은 분봉이 나와야함

        price_data = data

        if len(data) % minute == 0:
            index_data = [step * minute for step in range(int(np.trunc(len(data) / minute)))]
        else:
            index_data = [step * minute for step in range(int(np.trunc(len(data) / minute)) + 1)]  # 인터벌

        res = pd.DataFrame(price_data).iloc[index_data]

        res = pd.DataFrame(res)
        res.columns= col_name


        return res

    def total_time_Frame(self, data, minute):  # all data 분봉 출력( traning O  , real backtest x , real trading x)

        # 분봉출력: 실시간 시뮬레이터, 학습 = data count 만큼(또는 전체) 를 뽑아서 연산
        #       실전 = 일부만 뽑아서 전체를뽑은것과 같은 분봉이 나와야함

        price_data = pd.Series(data)
        price_data.dropna(inplace=True)

        if len(data) % minute == 0:
            index_data = [step * minute for step in range(int(np.trunc(len(data) / minute)))]
        else:
            index_data = [step * minute for step in range(int(np.trunc(len(data) / minute)) + 1)]  # 인터벌

        res = price_data[index_data].reset_index()[0]

        return res


    def coin_data_create(self,
                         minute,
                    data_count,
                    real_or_train,
                    coin_or_stock,
                    point_value,
                    coin_symbol):

        print(coin_symbol,'코인 명')

        start_date=data_count[0]
        end_date=data_count[1]
        symbol=coin_symbol.replace('/','')
        data = []

        URL = 'https://api.binance.com/api/v3/klines'  # 바이낸스
        COLUMNS = ['Datetime', 'open', 'high', 'low', 'close', 'volume', 'Close_time', 'quote_av', 'trades',
                   'tb_base_av', 'tb_quote_av', 'ignore']


        start = int(time.mktime(datetime.strptime(start_date , '%Y-%m-%d %H:%M').timetuple())) * 1000
        end = int(time.mktime(datetime.strptime(end_date , '%Y-%m-%d %H:%M').timetuple())) * 1000
        params_ = {
            'symbol': symbol,
            'interval': '1m',
            'limit': 1000,
            'startTime': start,
            'endTime': end
        }

        print('datetime  1000개 마다 호출')

        while start < end:
            print(datetime.fromtimestamp(start // 1000))
            params_['startTime'] = start

            for step in range(20):
                try:
                    result = requests.get(URL, params=params_)
                    break
                except:
                    print('재접속 시도', step)
                    continue

            js = result.json()
            if not js:
                break
            data.extend(js)  # result에 저장
            start = js[-1][0] + 60000  # 다음 step으로
        # 전처리
        if not data:  # 해당 기간에 데이터가 없는 경우
            print('해당 기간에 일치하는 데이터가 없습니다.')
            return -1
        df = pd.DataFrame(data)
        df.columns = COLUMNS
        df['Datetime'] = df.apply(lambda x: datetime.fromtimestamp(x['Datetime'] // 1000), axis=1)
        df = df.drop(columns=['Close_time', 'ignore'])
        df['Symbol'] = symbol
        df.loc[:, 'open':'tb_quote_av'] = df.loc[:, 'open':'tb_quote_av'].astype(float)  # string to float
        df['trades'] = df['trades'].astype(int)

        data_set = pd.DataFrame(df).reset_index()
        data_set = self.coin_total_time_Frame(data_set.values.tolist(), minute, data_set.columns)

        open = pd.Series(data_set['open'])
        close = pd.Series(data_set['close'])
        high = pd.Series(data_set['high'])
        low = pd.Series(data_set['low'])
        vol = pd.Series(data_set['volume'])
        date = pd.Series(data_set['Datetime'])


        self.close_ = close
        self.open = open
        self.low = low
        self.high = high
        self.vol_ = vol
        self.date_ = date

        scaler = MinMaxScaler()  # 0-1사이로 정규화  평균0.5 분산1
        close_1 = scaler.fit_transform(self.close_.values.reshape(-1, 1))
        vol_1 = scaler.fit_transform(self.vol_.values.reshape(-1, 1))
        high_1 = scaler.fit_transform(self.high.values.reshape(-1, 1))
        open_1 = scaler.fit_transform(self.open.values.reshape(-1, 1))
        low_1 = scaler.fit_transform(self.low.values.reshape(-1, 1))

        close_ = self.close_  # 스케일링 이전 데이터
        vol_ = self.vol_
        open_ = self.open
        low_ = self.low
        high_ = self.high

        close_s = close_1.reshape(-1)  # 스케일링 데이터
        vol_s = vol_1.reshape(-1)
        low_s = low_1.reshape(-1)
        high_s = high_1.reshape(-1)
        open_s = open_1.reshape(-1)
        date = self.date_

        data=[close_, open_, high_, low_, vol_, close_s, open_s, high_s, low_s, vol_s, date]

        return data  #DB 인서트 형식




    def distribute_data(self, input_, ratio):  # 훈련,검증,테스트 셋으로 나눔 , ratio 인풋= 비율의 리스트
        # ratio= train과 val test 비율
        step_ratio, _ = divmod(len(input_), (ratio[0] + ratio[1] + ratio[2]))
        train_ratio = step_ratio * ratio[0]
        val_ratio = step_ratio * ratio[1]
        test_ratio = step_ratio * ratio[2]

        try: #데이터 처리 (텐서형태)
            train_data = input_[:train_ratio].view(-1, 1)
            val_data = input_[train_ratio:train_ratio + val_ratio].view(-1, 1)
            test_data = input_[train_ratio + val_ratio:].view(-1, 1)

        except: #날짜 처리( 텐서형태 아님)
            train_data = input_[:train_ratio].reshape(-1,1)
            val_data = input_[train_ratio:train_ratio + val_ratio].reshape(-1,1)
            test_data = input_[train_ratio + val_ratio:].reshape(-1,1)

        return train_data, val_data, test_data

    def data_len(self,data_set): #nan을 제외한 데이터 길이
        len_data=[]
        for dim in range(len(data_set)):
            len_data.append(len(pd.Series(data_set[dim]).dropna().reset_index().values))

        start_index= -min(len_data)

        return start_index

    def input_create(self, minute,  # minute:분봉
                     ratio,  # ratio는 데이터셋 비율을 리스트로 넣음
                     data_count,  # 데이터수
                     coin_or_stock,  # coin or stock
                     point_value, #포인트가치
                     short_ind_name, #숏에 들어갈 ind 이름
                     long_ind_name, #롱에 들어갈 ind 이름
                     data_
                     ):  # 인풋 전처리 함수

        # 롱 숏 파라미터가 다를경우 길이 다르게 계산한다
        ind = ind_.ind_env()
        close_, open_, high_, low_, vol_, close_scale, open_scale, high_scale, low_scale, vol_scale, date = data_
        # 본데이터

        data_count=999999999999
        close_=pd.Series(close_.tolist())
        open_=pd.Series(open_.tolist())
        high_ = pd.Series(high_.tolist())
        low_ = pd.Series(low_.tolist())
        vol_ = pd.Series(vol_.tolist())


        price_ = open_[-data_count:]  # 가격에 들어갈 값
        close_ = close_[-data_count:]
        open_ = open_[-data_count:]
        high_ = high_[-data_count:]
        low_ = low_[-data_count:]
        vol_ = vol_[-data_count:]
        date = date[-data_count:]

        # 정규화 데이터
        close_scale = close_scale[-data_count:]
        vol_scale = vol_scale[-data_count:]
        open_scale = open_scale[-data_count:]
        high_scale = high_scale[-data_count:]
        low_scale = low_scale[-data_count:]


        ###########################################지표###################

        stan_weight = 1

        # 어떤지표 호출해야 하는지 확인

        ind_list = []
        ori_ind_list = []
        for ind_name in long_ind_name + short_ind_name:

            if ind_name == 'CCI':
                x, ori = ind.CCI(open_,150)
                ind_list.append(x)
                ori_ind_list.append(ori)

            if ind_name == 'log_return':
                log_return, ori_log_return = ind.log_return(open_, 4)
                ind_list.append(log_return)
                ori_ind_list.append(ori_log_return)


            if ind_name == 'vol_log_return':
                vol_return,ori_vol_return= ind.log_return(vol_,4)
                ind_list.append(vol_return)
                ori_ind_list.append(ori_vol_return)

            if ind_name == 'log_slope':
                log_slope, ori_log_slope = ind.slope(pd.Series(log_return.reshape(-1)), 8)
                ind_list.append(log_slope)
                ori_ind_list.append(ori_log_slope)

            if ind_name == 'VR':
                VR, ori_VR = ind.VR(14, open_, vol_)
                ind_list.append(VR)
                ori_ind_list.append(ori_VR)

            if ind_name == 'VR_slope':
                VR_slope, ori_VR_slope = ind.slope(pd.Series(VR.reshape(-1)), 4)
                ind_list.append(VR_slope)
                ori_ind_list.append(ori_VR_slope)

            # CEO test ind
            if ind_name == 'NTS1m':
                NTS1m, ori_NTS1m = ind.slope_line(open_, round(120 * stan_weight))
                ind_list.append(NTS1m)
                ori_ind_list.append(ori_NTS1m)

            if ind_name == 'line':
                line, ori_line = ind.slope_line(open_, round(120 * stan_weight))  ##################CEO 테스트
                ind_list.append(line)
                ori_ind_list.append(ori_line)

            if ind_name == 'one':
                one, one2, ori_one, ori_one2 = ind.one_side(open_, round(18 * stan_weight), round(52 * stan_weight),
                                                            round(104 * stan_weight))  #########CEO 테스트
                ind_list.append(one)
                ori_ind_list.append(ori_one)

            if ind_name == 'slope':
                slope, ori_slope = ind.slope(open_, round(40 * stan_weight))  ##########CEO test
                slope, ori_slope = ind.log_return(slope, 4)
                ind_list.append(slope)
                ori_ind_list.append(ori_slope)

            if ind_name == 'LR1m':
                LR1m, ori_LR1m = ind.LR1m(open_, round(40 * stan_weight), round(60 * stan_weight),
                                          round(90 * stan_weight), round(120 * stan_weight), round(140 * stan_weight),
                                          round(170 * stan_weight), round(200 * stan_weight), round(220 * stan_weight),
                                          2)  ####CEO test
                ind_list.append(LR1m)
                ori_ind_list.append(ori_LR1m)

            if ind_name == 'LR_CCI':
                LR_CCI, ori_LR_CCI = ind.LR_CCI(open_, round(40 * stan_weight), round(60 * stan_weight),
                                                round(90 * stan_weight), round(120 * stan_weight),
                                                round(140 * stan_weight), round(170 * stan_weight),
                                                round(200 * stan_weight), round(220 * stan_weight), 2)  ####CEO test
                ind_list.append(LR_CCI.flatten())
                ori_ind_list.append(ori_LR_CCI.flatten())

            if ind_name == 's_LR_CCI':
                s_LR_CCI, ori_s_LR_CCI = ind.s_LR_CCI(open_, round(25 * stan_weight), 1)
                ind_list.append(s_LR_CCI.flatten())
                ori_ind_list.append(ori_LR_CCI.flatten())


            if ind_name =='LRS_CCI_1m':
                LRS_CCI,ori_LRS_CCI= ind.LRS_CCI(open_,16,32,54,72,108,144,180,210,18)
                ind_list.append(LRS_CCI.flatten())
                ori_ind_list.append(ori_LRS_CCI.flatten())



            if ind_name =='s_LRS_CCI_1m':
                s_LRS_CCI,s_ori_LRS_CCI=ind.s_LRS_CCI(open_,16,18)
                ind_list.append(s_LRS_CCI.flatten())
                ori_ind_list.append(s_ori_LRS_CCI.flatten())


            if ind_name == 'LR_CCI_1m':
                LR_CCI, ori_LR_CCI = ind.LR_CCI(open_, 120, 180,
                                                240, 360,
                                                480, 720,
                                                1080, 1400, 60)  ####CEO test
                ind_list.append(LR_CCI.flatten())
                ori_ind_list.append(ori_LR_CCI.flatten())


            if ind_name == 's_LR_CCI_1m':
                s_LR_CCI,s_ori_LR_CCI= ind.s_LR_CCI(open_,60,1)
                ind_list.append(s_LR_CCI.flatten())
                ori_ind_list.append(s_ori_LR_CCI.flatten())


            if ind_name == 'LR_CCI_3m':
                LR_CCI, ori_LR_CCI = ind.LR_CCI(open_, 600, 1000,
                                                1400, 1800,
                                                2100, 2400,
                                                2700, 3000, 36)  ####CEO test
                ind_list.append(LR_CCI.flatten())
                ori_ind_list.append(ori_LR_CCI.flatten())


            if ind_name == 's_LR_CCI_3m':
                s_LR_CCI, s_ori_LR_CCI = ind.s_LR_CCI(open_, 60, 1)
                ind_list.append(s_LR_CCI.flatten())
                ori_ind_list.append(s_ori_LR_CCI.flatten())



            if ind_name =='LR_NQSNP_TRI_1m':
                import z_TRI as TRI_ind
                TRI_env= TRI_ind.TRI_env()
                LR_TRI,ori_LR_TRI = TRI_env.NQSNP_TRI()

                ind_list.append(LR_TRI)
                ori_ind_list.append(ori_LR_TRI)



            if ind_name == 'span':
                span1, span2, minslope, ori_span1, ori_span2, ori_minslope = ind.Min_span(open_, high_, low_, 1,
                                                                                          100)  #####Min test
                ind_list.append(span1)
                ori_ind_list.append(ori_span1)

            if ind_name == 'LR_short':
                LR_short, ori_LR_short = ind.LR_short(open_, round(35 * stan_weight), round(45 * stan_weight),
                                                      round(56 * stan_weight), round(74 * stan_weight),
                                                      round(90 * stan_weight), round(140 * stan_weight),
                                                      round(152 * stan_weight), round(280 * stan_weight),
                                                      round(4 * stan_weight))  # CEO 테스트
                ind_list.append(LR_short)
                ori_ind_list.append(ori_LR_short)

            if ind_name == 'LR_long':
                LR_long, ori_LR_long = ind.LR_Long(open_, round(35 * stan_weight), round(45 * stan_weight),
                                                   round(56 * stan_weight), round(74 * stan_weight),
                                                   round(90 * stan_weight), round(140 * stan_weight),
                                                   round(152 * stan_weight), round(280 * stan_weight),
                                                   round(4 * stan_weight))
                ind_list.append(LR_long)
                ori_ind_list.append(ori_LR_lon)

            if ind_name == 'wma':
                wma, ori_wma = ind.WMA(open_, 10)
                ind_list.append(wma)
                ori_ind_list.append(ori_wma)

            if ind_name == 'LRLR':
                LR8_short, WLR, LRLR, ori_LR8_short, ori_WLR, ori_LRLR = ind.LRLRA(open_, round(35 * stan_weight),
                                                                                   round(45 * stan_weight),
                                                                                   round(56 * stan_weight),
                                                                                   round(74 * stan_weight),
                                                                                   round(90 * stan_weight),
                                                                                   round(140 * stan_weight),
                                                                                   round(152 * stan_weight),
                                                                                   round(280 * stan_weight),
                                                                                   round(5 * stan_weight),
                                                                                   round(93 * stan_weight),
                                                                                   round(4 * stan_weight))
                LRLR, ori_LRLR = ind.log_return(LRLR, 2)
                ind_list.append(LRLR)
                ori_ind_list.append(ori_LRLR)

            if ind_name == 's_LRLR':
                s_LRLR, ori_s_LRLR = ind.s_LRLRA(open_, round(25 * stan_weight), 1)
                s_LRLR, ori_s_LRLR = ind.log_return(s_LRLR, 4)
                ind_list.append(s_LRLR)
                ori_ind_list.append(ori_s_LRLR)

            if ind_name == 'cross_wma':
                cross_wma, ori_cross_wma = ind.cross_wma_price(open_, round(25 * stan_weight))  # wma와 가격의 크로스업 다운 고려
                ind_list.append(cross_wma)
                ori_ind_list.append(ori_cross_wma)

            if ind_name == 'cross_wma_one':
                cross_wma_one, cross_wma_one2, ori_wma_one, ori_wma_one2 = ind.cross_wma_one(open_,
                                                                                             round(10 * stan_weight),
                                                                                             round(20 * stan_weight),
                                                                                             round(40 * stan_weight),
                                                                                             round(80 * stan_weight))
                ind_list.append(cross_wma_one)
                ori_ind_list.append(ori_wma_one)

            if ind_name== 'CCI_exercise':
                CCI_e,ori_CCI_e = ind.CCI_exercise(open_,30,20)

                ind_list.append(CCI_e)
                ori_ind_list.append(ori_CCI_e)

            if ind_name=='CCI_trend':
                CCI_t,ori_CCI_t = ind.CCI_trend(open_,5,20) #CCI_period, log_period
                ind_list.append(CCI_t)
                ori_ind_list.append(ori_CCI_t)


            if ind_name=='CCI_trend2':
                CCI_t,ori_CCI_t = ind.CCI_trend(open_,5,100) #CCI_period, log_period
                ind_list.append(CCI_t)
                ori_ind_list.append(ori_CCI_t)

            if ind_name == 'NNCO_up_S':
                NCO_up, ori_up = ind.NNCO_up(open_, 10, 4)  # 지지 저항 Long
                ind_list.append(NCO_up)
                ori_ind_list.append(ori_up)

            if ind_name == 'NNCO_down_S':
                NCO_down, ori_down = ind.NNCO_down(open_, 10, 4)
                ind_list.append(NCO_down)
                ori_ind_list.append(ori_down)

            if ind_name == 'NNCO_up_L':
                NCO_up, ori_up = ind.NNCO_up(open_, 10,4)  # 지지 저항 Long
                ind_list.append(NCO_up)
                ori_ind_list.append(ori_up)

            if ind_name == 'NNCO_down_L':
                NCO_down, ori_down = ind.NNCO_down(open_, 10, 4)
                ind_list.append(NCO_down)
                ori_ind_list.append(ori_down)

            if ind_name == 'LRC3m':
                res, res2, sign_short, sign_short2 = ind.LRC3m(open_, 80, 60, 600, 1000, 1400, 1800, 2300, 2800,
                                                               3300, 3800, 225, 450, 900, 1350, 1800, 2250, 2700,
                                                               675, 3, 120)
                ind_list.append(res)
                ori_ind_list.append(sign_short)

            if ind_name == 'LRC7m':
                res, res2, res3, sign_short, sign_short2, sign_short3 = ind.LRC7m(open_, 30, 18, 12, 200, 400, 800,
                                                                                  1200, 1600, 2000, 2400, 3000, 200,
                                                                                  300, 400, 500, 600, 700, 800,
                                                                                  1000, 50, 100, 150, 200, 300, 400,
                                                                                  75, 125, 6, 60, 240)
                ind_list.append(res)
                ori_ind_list.append(sign_short)

            if ind_name == 'LRC3m':
                res, res2, sign_short, sign_short2 = ind.LRC3m(open_, 80, 60, 600, 1000, 1400, 1800, 2300, 2800,
                                                               3300,
                                                               3800, 225, 450, 900, 1350, 1800, 2250, 2700, 675, 3,
                                                               120)
                ind_list.append(res)
                ori_ind_list.append(sign_short)

            if ind_name == 'LRC7m':
                res, res2, res3, sign_short, sign_short2, sign_short3 = ind.LRC7m(open_, 30, 18, 12, 200, 400, 800,
                                                                                  1200, 1600, 2000, 2400, 3000, 200,
                                                                                  300, 400, 500, 600, 700, 800,
                                                                                  1000,
                                                                                  50, 100, 150, 200, 300, 400, 75,
                                                                                  125,
                                                                                  6, 60, 240)
                ind_list.append(res)
                ori_ind_list.append(sign_short)

            if ind_name == 'LALA':
                scale_res, ori_res = ind.LALA(open_, 120, 200, 280, 360, 420, 480, 540, 600, 36)
                ind_list.append(scale_res)
                ori_ind_list.append(ori_res)

            if ind_name == 'LALA35':
                # scale_res, ori_res= ind.LALA35(close_,open_,high_,low_,600, 1000,1400,1800,2100,2400,2700,3000,3) #35기준
                scale_res, ori_res = ind.LALA35(open_, open_, high_, low_, 1200, 2000, 2800, 3600, 4200, 4800,
                                                5400,
                                                6000, 6)  # 15분 기준
                ind_list.append(scale_res)
                ori_ind_list.append(ori_res)

            if ind_name == 'LRC15m':
                scale_res, ori_res = ind.LRC15m(open_, open_, high_, low_, 15, 100, 200, 400, 600, 800, 1000, 1200,
                                                1500, 2)
                ind_list.append(scale_res)
                ori_ind_list.append(ori_res)

            if ind_name == 'LRC_SBSignal_15m_2':
                scale_res, ori_res = ind.LRC_SBSignal_15m_2(open_, open_, high_, low_, 15, 100, 200, 400, 600, 800,
                                                            1000, 1200, 1500, 2)
                ind_list.append(scale_res)
                ori_ind_list.append(ori_res)

            if ind_name == 'tanos':
                scale_res, ori_res = ind.tanos(open_, 20)
                ind_list.append(scale_res)
                ori_ind_list.append(ori_res)

            if ind_name == 'tanos_span1':
                scale_선행스팬1, scale_선행스팬2, 선행스팬1, 선행스팬2 = ind.tanos_span(open_, 9, 26, 52)
                ind_list.append(scale_선행스팬1)
                ori_ind_list.append(선행스팬1)

            if ind_name == 'tanos_span2':
                scale_선행스팬1, scale_선행스팬2, 선행스팬1, 선행스팬2 = ind.tanos_span(open_, 9, 26, 52)
                ind_list.append(scale_선행스팬2)
                ori_ind_list.append(선행스팬2)

            if ind_name == 'choco7m_CCIv':
                ind_list.append(CCIv)
                ori_ind_list.append(CCIv)

            if ind_name == 'choco7m_CCIv2':
                ind_list.append(CCIv2)
                ori_ind_list.append(CCIv2)

            if ind_name == 'choco7m_선행스팬1':
                ind_list.append(s_선행스팬1)
                ori_ind_list.append(선행스팬1)

            if ind_name == 'choco7m_선행스팬2':
                ind_list.append(s_선행스팬2)
                ori_ind_list.append(선행스팬2)

            if ind_name == 'choco7m_vgap':
                ind_list.append(s_vgap)
                ori_ind_list.append(vgap)

            if ind_name == 'choco7m_vsiggap':
                ind_list.append(s_vsiggap)
                ori_ind_list.append(vsiggap)

            if ind_name == 'choco7m_value99':
                ind_list.append(s_value99)
                ori_ind_list.append(value99)

            if ind_name == 'LALA3m_LRLA':
                ind_list.append(s_CCI_LRLA)
                ori_ind_list.append(CCI_LRLA)

            if ind_name == 'LALA3m_q8':
                ind_list.append(s_q8_SigshortA)
                ori_ind_list.append(q8_SigshortA)

            if ind_name == 'LALA3m_선행스팬1':
                ind_list.append(LALA3m_선행스팬1)
                ori_ind_list.append(LALA3m_선행스팬1)

            if ind_name == 'LALA3m_선행스팬2':
                ind_list.append(LALA3m_선행스팬2)
                ori_ind_list.append(LALA3m_선행스팬2)



            if ind_name=='SBS_signal_Sigshort':
                vM1=6
                cci_period1 = 20
                cci_period2 = 20
                cci_period3 = 30
                cci_period4 = 40
                cci_period5 = 90
                cci_period6 = 130
                cci_period7 = 200
                cci_period8 = 270
                acc_period1= 2

                s_Sigshort,ori_Sigshort=ind.SBS_signal_Sigshort(open_,open_,high_,low_ , vM1, cci_period1,cci_period2, cci_period3, cci_period4
                                                                ,cci_period5, cci_period6, cci_period7, cci_period8,acc_period1)


                ind_list.append(s_Sigshort)
                ori_ind_list.append(ori_Sigshort)


            if ind_name=='SBS_signal_NTRI':
                vM1=6
                CCI_period = 6
                CCI_period2 = 12
                CCI_period3 = 24
                CCI_period4 = 48
                CCI_period5 = 96
                CCI_period6 = 135
                CCI_period7 = 200
                CCI_period8 = 270
                NTRIP1 = 3

                s_NTRI,NTRI = ind.SBS_signal_NTRI(open_,high_,low_,open_,CCI_period, CCI_period2, CCI_period3, CCI_period4,CCI_period5,CCI_period6,CCI_period7, CCI_period8,vM1,NTRIP1)




                ind_list.append(s_NTRI)
                ori_ind_list.append(NTRI)



            if ind_name == 'SBS_signal_NTGI':
                vM1 = 6
                CCI_period = 60
                CCI_period2 = 120
                CCI_period3 = 240
                CCI_period4 = 480
                CCI_period5 = 960
                CCI_period6 = 1350
                CCI_period7 = 2000
                CCI_period8 = 2700
                NTRIP1 = 3

                s_NTGI, NTGI = ind.SBS_signal_NTGI(open_,high_,low_,CCI_period, CCI_period2, CCI_period3, CCI_period4, CCI_period5, CCI_period6, CCI_period7, CCI_period8,vM1, NTRIP1)


                ind_list.append(s_NTGI)
                ori_ind_list.append(NTGI)

            if ind_name == 'SBS_signal_xSB':
                vM1 = 15
                cci_period1 = 100
                cci_period2 = 200
                cci_period3 = 400
                cci_period4 = 600
                cci_period5 = 800
                cci_period6 = 1000
                cci_period7 = 1200
                cci_period8 = 1500
                acc_period1 = 2

                s_xSB,xSB =  ind.SBS_signal_xSB(open_, open_, high_, low_, vM1, cci_period1, cci_period2, cci_period3, cci_period4,cci_period5, cci_period6, cci_period7, cci_period8, acc_period1)  # LRC 15분과 같은거(이름만 다름)


                ind_list.append(s_xSB)
                ori_ind_list.append(xSB)

            if ind_name == 'SBS_signal_tanos':
                s_NTGI, NTGI = ind.tanos(open_,20)

                ind_list.append(s_NTGI)
                ori_ind_list.append(NTGI)


            if ind_name == 'SBS_signal_span1':
                전환선=9
                기준선=26
                스팬2=52
                scale_선행스팬1 , scale_선행스팬2 , 선행스팬1, 선행스팬2 = ind.SBS_span(open_,high_,low_,전환선,기준선,스팬2)
                ind_list.append(scale_선행스팬1)
                ori_ind_list.append(선행스팬1)

            if ind_name == 'SBS_signal_span2':
                전환선=9
                기준선=26
                스팬2=52
                scale_선행스팬1 , scale_선행스팬2 , 선행스팬1, 선행스팬2 = ind.SBS_span(open_,high_,low_,전환선,기준선,스팬2)
                ind_list.append(scale_선행스팬2)
                ori_ind_list.append(선행스팬2)





        # 지표 dict
        ind_dict = {}
        ori_ind_dict = {}
        for idx, ind_name in enumerate(long_ind_name + short_ind_name):
            ind_dict[ind_name] = ind_list[idx]
            ori_ind_dict[ind_name] = ori_ind_list[idx]




        data_name=[name for name in short_ind_name]
        data_set=[ind_dict[name] for name in data_name]
        short_start_period = self.data_len(data_set)

        data_name = [name for name in long_ind_name]
        data_set = [ind_dict[name] for name in data_name]
        long_start_period = self.data_len(data_set)

        if np.abs(short_start_period) > np.abs(data_count):  # data카운트보다 period가 더 길면 data count 만큼만 출력
            short_start_period = -data_count
        else:
            short_start_period = short_start_period

        if np.abs(long_start_period) > np.abs(data_count):  # data count 만큼 출력
            long_start_period = -data_count
        else:
            long_start_period = long_start_period



        long_price_ = torch.Tensor(np.array(price_[long_start_period:]))
        short_price_= torch.Tensor(np.array(price_[short_start_period:]))
        long_date_ = date.iloc[long_start_period:].astype('string').values
        short_date_= date.iloc[short_start_period:].astype('string').values

        long_train_price_,long_val_price_,long_test_price_= self.distribute_data(long_price_,ratio)
        long_train_date, long_val_date, long_test_date = self.distribute_data(long_date_, ratio)
        short_train_price_,short_val_price_,short_test_price_= self.distribute_data(short_price_,ratio)
        short_train_date, short_val_date,short_test_date = self.distribute_data(short_date_, ratio)

        print('시작 날짜 :', long_date_[0], '마지막 날짜 :', long_date_[-1])

        long_data_set=[ind_dict[name] for name in long_ind_name] #롱 지표 데이터 모음
        long_ori_data_set= [ori_ind_dict[name][long_start_period:] for name in long_ind_name]

        short_data_set=[ind_dict[name] for name in short_ind_name] #숏 지표 데이터 모음
        short_ori_data_set = [ori_ind_dict[name][short_start_period:] for name in short_ind_name]

        #인덱스 맞추고 tensor로
        long_to_torch=[torch.clamp(torch.Tensor(long_data[long_start_period:]),-1,1) for long_data in long_data_set]
        short_to_torch=[torch.clamp(torch.Tensor(short_data[short_start_period:]),-1,1) for short_data in short_data_set]

        # train,val,test set 으로 나눈다
        long_distribute_data=[self.distribute_data(data,ratio) for data in long_to_torch] #train_price_, val_price_, test_price_ = self.distribute_data(price_, ratio)
        short_distribute_data= [self.distribute_data(data,ratio) for data in short_to_torch]

        #long ind data
        long_train_data = []
        long_val_data = []
        long_test_data = []

        #short ind data
        short_train_data=[]
        short_val_data=[]
        short_test_data=[]

        #long_data
        for long_data in long_distribute_data:
            long_train_data.append(long_data[0]) #data 는 train val test로 나뉘어있음
            long_val_data.append(long_data[1])
            long_test_data.append(long_data[2])

        long_trading_data = [data for data in long_to_torch]  # 실전 트레이딩시 데이터
        long_ori_price = [long_train_price_, long_val_price_, long_test_price_]
        long_date_data=[long_train_date,long_val_date,long_test_date]
        long_date_total=long_date_.tolist()

        #short data
        for short_data in short_distribute_data:
            short_train_data.append(short_data[0])
            short_val_data.append(short_data[1])
            short_test_data.append(short_data[2])

        short_trading_data = [data for data in short_to_torch]  # 실전 트레이딩시 데이터
        short_ori_price = [short_train_price_, short_val_price_, short_test_price_]
        short_date_data = [short_train_date, short_val_date, short_test_date]
        short_date_total = short_date_.tolist()


        #plot
        total_start_period=max(long_start_period,short_start_period) #플랏용 index. 음수이므로 max 취해야 작은값이 나옴

        if params.plot_print==True: #플랏이 트루일때 지표와 가격출력
            fig, ax = plt.subplots(3, 1, figsize=(10, 9))
            ax[0].plot(torch.cat(long_ori_price)[total_start_period:])

            for d in range(len(long_trading_data)):
                ax[1].set_ylabel('long input data')
                label = params.long_ind_name[d] if params.long_ind_name else f"Data {d}"
                ax[1].plot(long_trading_data[d][total_start_period:], label=label)

            for d in range(len(short_trading_data)):
                ax[2].set_ylabel('short input data')
                label2 = params.short_ind_name[d] if params.short_ind_name else f"Data {d}"
                ax[2].plot(short_trading_data[d][total_start_period:], label=label2)

            ax[1].legend(loc='upper left')
            ax[2].legend(loc='upper left')
            plt.show()

        long_input= [long_train_data, long_val_data, long_test_data, long_ori_price, long_trading_data , long_date_data, long_date_total]
        short_input= [short_train_data, short_val_data, short_test_data, short_ori_price, short_trading_data , short_date_data, short_date_total]

        ori_ind_data=[long_ori_data_set,short_ori_data_set]

        return long_input,short_input,ori_ind_data  # 스케일링된 data들과 기존 종가(close) 데이터가 나옴






    # --------------------------------------- RL에이전트 -------------------------------------

    def LSTM_observation(self, input_, window, input_dim):
        combined = torch.stack(input_, dim=-1)
        # 윈도우 사이즈
        window_size = window

        # 변환할 시계열 데이터의 길이
        n = combined.shape[0]

        # 변환된 시계열 데이터를 저장할 리스트를 생성
        sequences = []

        # 시계열 데이터를 window_size 만큼씩 잘라서 리스트에 저장
        for i in range(n - window_size + 1):
            sequence = combined[i:i + window_size]
            sequences.append(sequence)

        # 리스트를 tensor로 변환합니다.
        data = torch.stack(sequences)

        # batch_size, window_size, dim 순서로 tensor를 재배열
        data = data.permute(0, 1, 2)
        return data


    def CNN_observation(self, input_, dim):  # CNN인풋 데이터로 만든다
        # 배치 크기 × dim × 높이(height) × 너비(widht,window size)의 크기의 텐서를 선언
        data = input_.view(1, dim, 1, -1)

        return data



    def Liq(self, model, action, step):  # 청산 함수

        if step > 0:
            if action==0 and model.long_aver_price !=0:  # 숏포지션인데 롱잔량 있는 경우 (롱청산)
                #롱 청산
                model.cash += (model.long_unit * model.deposit) + (model.price - model.long_aver_price) * model.long_unit  # 롱 전량매도
                #롱 초기화
                model.long_unit=0
                model.long_aver_price=0
                model.long_price=[]
                liq=True

            elif action == 2 and model.short_aver_price !=0:  # 롱 포지션이고 숏 잔량 있는경우
                if model.short_aver_price == 0:  # 숏포지션에서 산게 없는경우
                    model.cash += (model.short_unit* model.deposit)

                else:
                    model.cash += (model.short_unit * model.deposit) + (model.short_aver_price-model.price)*model.short_unit

                # 숏 포지션 변수 초기화
                model.short_unit = 0
                model.short_price = []  # 매수했던 가격*계약수
                model.short_aver_price = 0
                liq=True

            else:
                liq = False

        

        else:
            liq = False

        return liq

    # 롱포지션 PV= (롱포지션 계약수 * 증거금약 15000달러) + (나스닥 달러가격 - 평단가) * 롱포지션 계약수
    # 숏포지션 PV= (숏포지션 계약수 * 증거금약 15000달러) + (나스닥 달러가격 - 평단가) * 숏포지션 계약수

    # 롱 PV 달러기준 = (계약건수 * 거래증거금) + ((거래종가-거래평단가)*포인트가치)*계약건수
    # model.cash += (unit[2] * model.deposit) + (model.price-model.long_aver_price)*unit[2]*포인트 가치(데이터 맨처음 이미 곱해짐)
    # model.PV= (model.long_unit * model.deposit) + (model.price-model.long_aver_price)*model.long_unit

    # 숏 PV 달러기준 = (계약건수 * 거래증거금) + ((거래평단가-거래종가)*포인트가치)*계약건수
    # model.cash += (unit[0] * model.deposit) + (model.short_aver_price-model.price)*unit[0]*포인트 가치(데이터 맨처음 이미 곱해짐)
    # model.PV= (model.short_unit * model.deposit) + (model.short_aver_price-model.price)*model.short_unit





    def long_discrete_step(self, action, unit, step,model):  # 에이전트의 액션이 discrete할때 보상을 결정하는 함수, 롱온리
        # 액션0= 청산 1=관망  2=롱
        action = action.item()

        # 롱 청산
        if action == 0:
            if model.long_aver_price == 0 and model.long_unit == 0:  # 숏포지션에서 산게 없는경우
                model.cash += (model.long_unit * model.price)/params.leverage  # 레버리지 적용하여 현금 반환
                model.long_unit = 0
                unit[0] = 0
                model.long_price = []
                model.long_aver_price = 0

            elif model.long_unit != 0:  # long 포지션 보유중인경우
                if model.long_unit < unit[0]:
                    # 가진계약 부족한데 팔계약은 있는경우 전량 청산
                    unit[0] = model.long_unit
                    model.cash += ((model.long_aver_price *(unit[0]/params.leverage)+( model.price-model.long_aver_price )* (unit[0])))  # 레버리지 적용하여 현금 반환
                    model.long_unit -= unit[0]
                    model.long_price = []
                    model.long_aver_price = 0

                else:  # 가진 계약 충분한 경우
                    model.long_unit -= unit[0]
                    model.cash += ((model.long_aver_price *(unit[0]/params.leverage)+( model.price -model.long_aver_price )* (unit[0])))   # 레버리지 적용하여 현금 반환

                    model.long_price.append(-model.long_aver_price * unit[0])
                    if model.long_unit == 0:
                        model.long_aver_price = 0
                    else:
                        model.long_aver_price = np.sum(model.long_price) / model.long_unit


        elif action == 2:

            if params.train_stock_or_future == 'future':
                # 레버리지를 적용하여 거래 가능한 금액 계산
                if model.cash >= unit[2] * model.price and unit[2] > 0:
                    model.cash -= model.price * unit[2]   # 실제 사용하는 현금은 레버리지로 나눔
                    model.long_price.append((model.price + model.slip) * unit[2]* params.leverage)
                    model.long_unit += unit[2]* params.leverage

                    if model.long_unit == 0:
                        model.long_aver_price = 0

                    else:
                        model.long_aver_price = np.sum(model.long_price) / model.long_unit

                elif model.cash < model.price:  # 유닛대비 현금부족
                    unit[2] = float(model.cash) / (model.price + model.slip)
                    model.cash -= unit[2] * (model.price + model.slip)  # 실제 사용하는 현금은 레버리지로 나눔
                    model.long_price.append((model.price + model.slip) * unit[2]* params.leverage)
                    model.long_unit += unit[2]* params.leverage

                    if model.long_unit == 0:
                        model.long_aver_price = 0

                    else:
                        model.long_aver_price = np.sum(model.long_price) / model.long_unit


                else:  # 1주도 못사는경우
                    unit[2] = 0


        else:  # 관망일경우
            model.cash = model.cash
            model.stock = model.stock
            model.long_unit = model.long_unit

        # Rs=다음스테이트에서 팔았을때 리워드
        # Rh="홀딩했을때
        # Rb="매수했을때


        cost = (unit[0] * (model.price - model.slip)) * params.coin_future_cost + ((unit[2] * (model.price + model.slip)) * params.coin_future_cost)  # 업비트는 매매시 0.05%
        # 롱 PV계산
        model.cash -= cost  # 수수료를 지불
        # 보유계약가치 + 순손익 + 현금 PV업데이트
        model.PV = (model.long_unit/params.leverage * model.long_aver_price) + ((model.price-model.long_aver_price) * model.long_unit) + model.cash  # 보유 가치 + 손익 + 현금
        model.PV_list.append(model.PV)

        max_price= max(model.price_data)

        if model.PV <= 0:
            model.PV = 0
            model.cash = 0
            model.stock = 0
            model.short_unit = 0
            model.long_unit = 0


        if model.PV <= 1:  # 청산인경우
            model.PV = -(max_price*params.leverage)
            model.cash = model.cash

        #######################리워드 계산
        reward1 = torch.log(model.PV_list[step + 1] / model.PV_list[step])
        reward = reward1 * params.long_reward_bonus
        reward_weight = 2


        if model.PV_list[step + 1] < params.cash * (3 / 4) and reward1 < 0:  # PV가 초기원금의 설정 기준 미만인경우 손실시 리워드가 마이너스 인경우만 더큰 -보상
            reward = reward * reward_weight

        model.action_data.append(action)
        model.reward_data.append(reward)
        model.step_data.append(step)

        if params.traj_print == True:
            print(model.Agent_num,'에이전트 넘버', action, '액션', model.PV, 'PV', model.price, '가격', model.cash, 'cash', model.long_unit, 'long_stock',
                  model.short_unit, '숏유닛',
                  unit, '유닛', unit, '리워드계산시유닛',
                  model.long_aver_price, '롱평단',
                  model.short_aver_price, '숏평단', reward,'리워드')


        return action, reward, step












    def short_discrete_step(self, action, unit, step,model):
        # 액션0= 숏 1관망 2 청산
        action = action.item()
        # 숏 청산
        if action == 0:
            if model.short_aver_price == 0 and model.short_unit == 0:  # 숏포지션에서 산게 없는경우
                model.cash += ((model.short_aver_price * (model.short_unit / params.leverage) + (
                            model.short_aver_price - model.price) * (model.short_unit)))  # 레버리지 적용(매도시에는 원래의 레버리지x 갯수로 매도)   # 전량 청산
                model.short_unit = 0
                unit[0] = 0
                model.short_price = []
                model.short_aver_price = 0

            if model.short_unit != 0:  # 숏 포지션 보유중인경우

                if np.round(float(model.short_unit),2) < np.round(float(unit[0]),2):
                    # 가진계약 부족한데 팔계약이 더많은 경우 전량 청산
                    unit[0] = model.short_unit
                    model.cash += ((model.short_aver_price *(unit[0]/params.leverage)+(model.short_aver_price - model.price)* (unit[0])))

                    model.short_unit -= unit[0]
                    model.short_price = []
                    model.short_aver_price = 0

                else :  # 가진 계약 충분한 경우
                    model.short_unit -= unit[0]
                    model.cash += ((model.short_aver_price *(unit[0]/params.leverage)+(model.short_aver_price - model.price)* (unit[0])))
                    model.short_price.append(-model.short_aver_price * unit[0])

                    if model.short_unit == 0:
                        model.short_aver_price = 0
                    else:
                        model.short_aver_price = np.sum(model.short_price) / model.short_unit



        # 숏 포지션 매수
        if action == 2:
            # 레버리지를 적용하여 거래 가능한 금액 계산
            if model.cash >= unit[2] * model.price and unit[2] > 0:
                model.cash -= unit[2] * model.price
                model.short_unit += unit[2] * params.leverage
                model.short_price.append((model.price - model.slip) * unit[2]* params.leverage)

                if model.short_unit == 0:
                    pass
                else:
                    model.short_aver_price = np.sum(model.short_price) / model.short_unit  # 숏 매수 평단가

            elif model.cash < model.price and model.cash > 1:  # 유닛대비 현금부족(1달러 이상있을때)

                unit[2] = float(model.cash) / (model.price - model.slip)  # 몫과 나머지
                model.cash -= unit[2] * (model.price - model.slip)
                model.short_price.append((model.price - model.slip) * unit[2]* params.leverage)
                model.short_unit += unit[2] * params.leverage

                if model.short_unit == 0:
                    pass
                else:
                    model.short_aver_price = np.sum(model.short_price) / model.short_unit  # 숏 매수 평단가

            else:
                unit[2] = 0


        else:  # 관망일경우
            model.cash = model.cash


        cost = (unit[0] * (model.price - model.slip)) * params.coin_future_cost + ((unit[2] * (model.price + model.slip)) * params.coin_future_cost)  # 업비트는 매매시 0.05%
        model.cash -= cost  # 수수료를 지불
        model.PV = (model.short_unit/params.leverage * model.short_aver_price)+ ((model.short_aver_price-model.price)*model.short_unit) + model.cash   #처음샀던 가치 + 손익 + 현금

        model.PV_list.append(model.PV)
        max_price= max(model.price_data)

        if model.PV <= 0:
            model.PV = 0
            model.cash = 0
            model.stock = 0
            model.short_unit=0
            model.long_unit=0

        if model.PV <= 1: # 청산인경우
            model.PV = -(max_price*params.leverage)
            model.cash = 0

        #######################리워드 계산
        reward1 = torch.log(model.PV_list[step + 1] / model.PV_list[step])
        reward = reward1 * params.short_reward_bonus
        reward_weight = 2

        if model.PV_list[step + 1] < params.cash * (3 / 4) and reward1 < 0:  # PV가 초기원금의 설정 기준 미만인경우 손실시 리워드가 마이너스 인경우만 더큰 -보상
            reward = reward * reward_weight

        model.action_data.append(action)
        model.reward_data.append(reward)
        model.step_data.append(step)

        if params.traj_print==True:
            print(model.Agent_num,'에이전트 넘버',action, '액션', model.PV, 'PV', model.price, '가격', model.cash, 'cash', model.short_unit, 'short_stock',
                  model.short_unit, '숏유닛',
                  unit, '유닛', model.long_aver_price, '롱평단',
                  model.short_aver_price, '숏평단', reward,'리워드')


        return action, reward, step



    def SC_discrete_step(self, action, unit, step, model): #stock, coin으로 진행 ( 위의 롱 숏은 선물 기준으로 함 )
        # 액션0= 매도 1=관망  2=매수
        # 참고 Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy 논문

        action = action.item()
        if action == 0:  # 매도일경우
            if model.stock >= unit[0]:  # 주식의 갯수가 팔려는 갯수이상일때  매도
                model.stock -= unit[0]
                model.cash += (model.price - model.slip) * unit[0]

            elif model.stock > 0:
                # 가진주식수 부족한데 팔주식은 있는경우
                model.cash += model.stock * (model.price - model.slip)  # 전량매도
                unit[0] = model.stock
                model.stock = model.stock - unit[0]

            else:
                unit[0] = 0

        elif action == 2:  # 매수일경우

            if model.cash >= unit[2] * (model.price + model.slip):  # 가진현금이 사려고하는 가격보다 많아서 살수있을때
                model.stock += unit[2]
                model.cash -= (model.price + model.slip) * unit[2]


            elif model.cash < (model.price + model.slip):  # 유닛대비 현금부족 1주도 못사는경우

                if params.coin_or_stock == 'coin':
                    quotient = model.cash / (model.price + model.slip + (model.price * params.stock_cost))
                    remainder = 0

                    model.stock += quotient + remainder
                    model.cash -= (quotient + remainder) * (model.price + model.slip)
                    unit[2] = quotient + remainder
                else:
                    quotient, remainder = divmod(float(model.cash), float(
                        model.price + model.slip + (model.price * params.stock_cost)))  # 몫과 나머지

                    model.stock += quotient
                    model.cash -= quotient * (model.price + model.slip)
                    unit[2] = quotient



            else:  # 돈부족해서 못사는경우 최대로 가진돈을 매수
                unit[2] = 0

        else:  # 관망일경우
            model.cash = model.cash
            model.stock = model.stock

        cost = (unit[0] * (model.price - model.slip)) * model.cost + ((unit[2] * (model.price + model.slip)) * model.cost)  # 업비트는 매매시 0.05%



        # PV 계산
        reward = 0

        model.cash -= cost  # 매도시 수수료를 지불
        model.PV = (model.stock * model.price + model.cash)  # PV업데이트
        model.PV_list.append(model.PV) # PV 저장

        # 리워드 계산
        reward=0

        if model.PV <= 0:
            model.PV = 0
            model.cash = 0
            model.stock = 0
            model.short_unit = 0
            model.long_unit = 0
            reward = torch.Tensor([-1000]) * bonus


        model.action_data.append(action)
        model.reward_data.append(reward)
        model.step_data.append(step)

        if params.traj_print == True:
            print(model.Agent_num,'에이전트 넘버',action, 'action', model.price, 'price', reward, '리워드', model.PV, 'PV',
                  model.stock, 'stock', model.cash, 'cash', unit, 'unit', cost, 'cost', step, 'step')

        return action, reward, step

