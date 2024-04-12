import pandas as pd

import e_train as params
import a_Env as env_
import c_PPO_Agent as PPO_Agent
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import b_network as NET
import torch.multiprocessing as multiprocessing

import random
import torch
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

env=env_.Env()




class back_testing:
    def __init__(self,train_val_test):
        # 시뮬 data
        self.is_back_testing=True
        self.window=0

        # long short 2개 생성
        self.PV_data = {'long':[], 'short':[]}
        self.action_data = {'long':[], 'short':[]}
        self.buy_data = {'long':[], 'short':[]}  # 매수한 가격
        self.sell_data = {'long':[], 'short':[]}  # 매도한 가격
        self.buy_date = {'long':[], 'short':[]}
        self.sell_date = {'long':[], 'short':[]}
        self.price_data={'long':[],'short':[]} #가격 데이터
        self.date_data={'long':[], 'short':[]} # 날짜 데이터
        self.scale_input={'long':[],'short':[]}

        self.train_val_test = train_val_test

        self.Global_policy_net={}
        self.Global_value_net={}
        self.agent_data={}


        # 데이터 호출
        env = env_.Env()

        is_API, data_ , ind_data = env_.load_price_data(params.API_data_name)  # csv를 불러올지, api를 불러올지 선택

        long_input_ = ind_data[0]
        short_input_= ind_data[1]
        ori_ind_data = ind_data[2]

        if is_API == 'API':
            data_ = env.coin_data_create(params.minute, params.data_count, params.real_or_train, params.coin_or_stock,
                                         params.point_value, params.API_data_name)  # 학습시 뽑은 history 데이터

            long_input_, short_input_, ori_ind_data = env.input_create(params.minute, params.ratio, params.data_count,
                                                                       params.coin_or_stock, params.point_value,
                                                                       params.short_ind_name, params.long_ind_name,
                                                                       data_)  # ind에서 높은 값을 뽑음

            ind_data = [long_input_, short_input_, ori_ind_data]
            env_.save_price_data(data_, ind_data, params.API_data_name)


        long_train_data,long_val_data,long_test_data,long_ori_close,long_total_input,long_date_data,long_total_date= long_input_
        short_train_data,short_val_data,short_test_data,short_ori_close,short_total_input,short_date_data,short_total_date= short_input_


        # APPO는 전체 넣음, PPO는 데이터셋 나눠서 넣음

        if train_val_test == 'train':
            self.long_price_data = long_ori_close[0]
            self.long_scale_input = long_train_data
            self.long_date_data = long_date_data[0]

            self.short_price_data = short_ori_close[0]
            self.short_scale_input = short_train_data
            self.short_date_data = short_date_data[0]


        elif train_val_test == 'val':
            self.long_price_data = long_ori_close[1]
            self.long_scale_input = long_val_data
            self.long_date_data = long_date_data[1]

            self.short_price_data = short_ori_close[1]
            self.short_scale_input = short_val_data
            self.short_date_data = short_date_data[1]


        elif train_val_test == 'test':
            self.long_price_data = long_ori_close[2]
            self.long_scale_input = long_test_data
            self.long_date_data = long_date_data[2]

            self.short_price_data = short_ori_close[2]
            self.short_scale_input = short_test_data
            self.short_date_data = short_date_data[2]


        elif train_val_test == 'total':
            self.long_price_data = torch.cat([long_ori_close[0], long_ori_close[1], long_ori_close[2]])
            self.long_scale_input = long_total_input
            self.long_date_data = long_total_date

            self.short_price_data = torch.cat([short_ori_close[0], short_ori_close[1], short_ori_close[2]])
            self.short_scale_input = short_total_input
            self.short_date_data = short_total_date



        # 에이전트 호출

        for short_or_long in params.short_or_long_data:
            if short_or_long=='long':
                input_dim=len(params.long_ind_name)
                self.window = params.window[short_or_long]
            if short_or_long=='short':
                input_dim=len(params.short_ind_name)
                self.window = params.window[short_or_long]

            # global net
            Global_actor = NET.Global_actor
            Global_critic = NET.Global_critic
            self.Global_policy_net[short_or_long]=Global_actor(params.device, self.window,input_dim, short_or_long, params.Neural_net, params.bidirectional_)  #dict 에 글로벌넷 저장
            self.Global_value_net[short_or_long]=Global_critic(params.device, self.window,input_dim, short_or_long, params.Neural_net, params.bidirectional_)  #Global_policy, Global_value ,


            #숏 or 롱 포지션 따라 인풋 정의
            if short_or_long=='short':
                input_=self.short_scale_input
                self.input_dim=params.input_dim['short']
                ori_close=self.short_price_data
                date_data=self.short_date_data

            else:
                input_=self.long_scale_input
                self.input_dim = params.input_dim['long']
                ori_close=self.long_price_data
                date_data=self.long_date_data

            if params.train_stock_or_future == 'future':
                cost_ = params.cost  # 선물 cost
            else:
                cost_ = params.stock_cost  # 주식 cost

            agent_num = 0  # 글로벌 에이전트 넘버=0
            #에이전트 정의
            self.agent_data[short_or_long] = PPO_Agent.PPO(self.window,  # LSTM 윈도우 사이즈
                                                      params.cash,  # 초기 보유현금
                                                      cost_,  # 수수료 %
                                                      params.device,  # 디바이스 cpu or gpu
                                                      params.k_epoch,  # K번 반복
                                                      input_,  # 인풋 데이터
                                                      ori_close,  # 주가 데이터
                                                      date_data,  # 날짜 데이터
                                                      self.input_dim,  # feature 수
                                                      agent_num,
                                                      params.coin_or_stock,
                                                      params.deposit,
                                                      params.backtest_slippage,
                                                      short_or_long,  # 숏인지 롱인지
                                                      self.Global_policy_net[short_or_long],  # 글로벌넷
                                                      self.Global_value_net[short_or_long],  # 글로벌넷
                                                      self.is_back_testing
                                                      )



    def reset(self):
        self.PV_data = {'long':[], 'short':[]}
        self.action_data = {'long':[], 'short':[]}
        self.buy_data = {'long':[], 'short':[]}  # 매수한 가격
        self.sell_data = {'long':[], 'short':[]}  # 매도한 가격
        self.buy_date={'long':[], 'short':[]}
        self.sell_date= {'long':[], 'short':[]}



    def back_test(self,is_short_or_long,long_res,short_res):  # 백테스팅
        #시뮬레이션

        if params.multi_or_thread=='multi':
            self.reset() #시뮬 데이터 리셋
        else: #스레드 학습인경우 data dict 초기화 x (롱과 숏 모두 모아서 계산해야함)
            pass

        self.agent=self.agent_data[is_short_or_long]

        self.agent.reset()  # 리셋 (리셋때 back testing=False 된다)
        self.agent.back_testing = True
        self.is_back_testing=True

        self.agent.scale_input = self.agent.scale_input # 인풋 데이터
        self.agent.price_data = self.agent.price_data  # 종가 데이터

        # 데이터 가공
        self.agent.LSTM_input = self.agent.LSTM_observation(self.agent.scale_input, self.agent.window,
                                                            self.agent.dim)  # LSTM 데이터
        self.agent.LSTM_input_size = self.agent.LSTM_input.size()[2]


        if is_short_or_long == 'short':  # 숏인경우 숏가중치 저장
            policy_net = self.Global_policy_net[is_short_or_long]
            self.decide_action = self.agent.short_decide_action
            self.discrete_step = env.short_discrete_step
            self.Global_lr = params.short_Global_actor_net_lr

        elif is_short_or_long == 'long':  # 롱인경우 롱가중치 저장
            policy_net = self.Global_policy_net[is_short_or_long]
            self.decide_action = self.agent.long_decide_action
            self.discrete_step = env.long_discrete_step
            self.Global_lr = params.long_Global_actor_net_lr

        if params.train_stock_or_future != 'future':
            self.decide_action = self.agent.SC_decide_action
            self.discrete_step = self.agent.SC_discrete_step
            print('주식처럼 백테스트')

        ##저장된 가중치 load
        policy_net.load()

        value_net = self.Global_value_net['long']
        value_net.load()
        value = value_net(self.agent.LSTM_input.to(self.agent.device)).to(self.agent.device)
        mean_value = torch.mean(value)
        value = torch.cumsum(value-mean_value, dim=0)
        value = value.detach().to('cpu').numpy()

        plt.title('predict value')
        plt.plot(value)
        plt.show()

        # back testing

        for step in range(len(self.agent.price_data)):
            with torch.no_grad():
                prob_ = policy_net(self.agent.LSTM_input.to(self.agent.device)).to(self.agent.device)
                policy = F.softmax(prob_,dim=1)  # policy
            self.agent.price = self.agent.price_data[step]  # 현재 주가업데이트

            if params.train_stock_or_future == 'future':  # 선물처럼 학습시킬경우
                action, unit = self.decide_action(policy[step],params.deposit)  # 액션 선택
            else:
                action, unit = self.decide_action(policy[step])

            action, reward, step_ = self.discrete_step(action, unit, step, self.agent)  # PV및 cash, stock 업데이트

            if step == 10:
                max_values = policy.max(dim=0).values  # 각 열에서 가장 큰 값들을 찾습니다.
                max_policy = max_values.tolist()  # policy 값으로 변환하여 출력합니다.
                min_values = policy.min(dim=0).values  # 각 열에서 가장 큰 값들을 찾습니다.
                min_policy = min_values.tolist()  # policy 값으로 변환하여 출력합니다.


            if action == 0: #매도
                self.action_data[is_short_or_long].append(0)
                if unit[0] !=0 : #매매 유닛이 0이 아닌경우
                    self.sell_data[is_short_or_long].append(self.agent.price_data[step])
                    self.sell_date[is_short_or_long].append(step)

            elif action == 1: #관망
                self.action_data[is_short_or_long].append(1)

            else: #매수
                self.action_data[is_short_or_long].append(2)
                if unit[2] != 0:  # 매매 유닛이 0이 아닌경우
                    self.buy_data[is_short_or_long].append(self.agent.price_data[step])
                    self.buy_date[is_short_or_long].append(step)

            # 데이터 저장
            self.PV_data[is_short_or_long].append(self.agent.PV)

            date_data_set={'long': self.long_date_data,
                           'short': self.short_date_data}

            if step % 50 == 0:  #실시간 출력값
                print(step + 1, '/', len(self.agent.price_data), '테스팅중..',  is_short_or_long + '_agent PV :', float(self.PV_data[is_short_or_long][-1]) )
                print(policy[step],'policy',action,'액션',unit,'유닛',self.agent.stock,'보유주식수')

        market_first = self.agent.price_data[0]
        market_last = self.agent.price_data[-1]

        # 결과
        if params.multi_or_thread=='multi': #멀티프로세싱인경우 Queue에 저장
            if is_short_or_long=='long':
                long_res.put([self])
            else:
                short_res.put([self])
        else:
            if is_short_or_long=='long': #스레드인 경우 리스트 저장
                long_res.append(self)
            else:
                short_res.append(self)

        self.date_data[is_short_or_long]=self.agent.date_data
        self.price_data[is_short_or_long]= self.agent.price_data
        self.scale_input[is_short_or_long]= self.agent.scale_input
        #print(len(self.PV_data),len(self.action_data),len(self.buy_data),len(self.buy_date),len(self.sell_date),len(self.sell_date),'aksfnkasnfklsnkf')

        print((((market_last / market_first) - 1) * 100).item(), ':Market ratio of long return')
        print(float(((self.PV_data[is_short_or_long][-1] / self.PV_data[is_short_or_long][0]) - 1) * 100), '% :'+is_short_or_long+'_agent PV return')
        if params.coin_or_stock=='future': #선물인경우
            print(float((((self.PV_data[is_short_or_long][-1]-self.agent.init_cash) / self.agent.deposit)) * 100),'% :' + is_short_or_long + '_agent 증거금 대비 PV return')


        return long_res,short_res




    def res_plot(self, res_data):  # 백테스팅 이후 실행
        ########갯수 잘못됨
        ind_diff={}

        if params.train_stock_or_future == 'future': #학습방식이 선물인경우
            try:
                if len(res_data['long'].PV_data['long']) > len(res_data['short'].PV_data['short']):  # 롱데이터수 더길면 (롱의 지표변수가 더짧다)
                    ind_diff['long'] = np.abs(len(res_data['long'].PV_data['long']) - len(res_data['short'].PV_data['short']))  # 그래프 계산시 얼마나 빼야할지
                    ind_diff['short'] = 0

                    # 인덱스 갯수 줄이기(롱이 많은경우)
                    res_data['long'].PV_data['long']= res_data['long'].PV_data['long'][ind_diff['long']:]

                    res_data['long'].buy_date['long']=res_data['long'].buy_date['long'] - ind_diff['long']  # 인덱스 조절
                    res_data['long'].sell_date['long'] = res_data['long'].sell_date['long'] - ind_diff['long']  # 인덱스 조절

                    res_data['long'].buy_data['long']=res_data['long'].buy_data['long'][len(res_data['long'].buy_date['long'][res_data['long'].buy_date['long']<0]):]
                    res_data['long'].sell_data['long'] = res_data['long'].sell_data['long'][len(res_data['long'].sell_date['long'][res_data['long'].sell_date['long'] < 0]):]

                    res_data['long'].buy_date['long']=res_data['long'].buy_date['long'][res_data['long'].buy_date['long'] >= 0]
                    res_data['long'].sell_date['long'] = res_data['long'].sell_date['long'][res_data['long'].sell_date['long'] >= 0]

                    res_data['long'].agent.price_data= self.price_data['long'][ind_diff['long']:]
                    res_data['long'].agent.date_data=self.date_data['long'][ind_diff['long']:]

                    long_ind_data=[]
                    for dim in range(len(self.scale_input['long'])):
                        long_ind_data.append(self.scale_input['long'][dim][ind_diff['long']:])
                    res_data['long'].agent.scale_input = long_ind_data


                elif len(res_data['long'].PV_data['long']) < len(res_data['short'].PV_data['short']):  # 숏이20 더길면 숏에 20이 들어감
                    ind_diff['short'] = np.abs(len(res_data['long'].PV_data['long']) - len(res_data['short'].PV_data['short']))
                    ind_diff['long'] = 0

                    res_data['short'].PV_data['short'] = res_data['short'].PV_data['short'][ind_diff['short']:]
                    res_data['short'].buy_date['short'] = res_data['short'].buy_date['short'] - ind_diff['short']  # 인덱스 조절
                    res_data['short'].sell_date['short']= res_data['short'].sell_date['short'] - ind_diff['short']

                    res_data['short'].buy_data['short']=res_data['short'].buy_data['short'][len(res_data['short'].buy_date['short'][res_data['short'].buy_date['short']<0]):]
                    res_data['short'].sell_data['short']=res_data['short'].sell_data['short'][len(res_data['short'].sell_date['short'][res_data['short'].sell_date['short']<0]):]

                    res_data['short'].buy_date['short'] = res_data['short'].buy_date['short'][res_data['short'].buy_date['short'] >= 0]
                    res_data['short'].sell_date['short'] = res_data['short'].sell_date['short'][res_data['short'].sell_date['short'] >= 0]

                    res_data['short'].agent.price_data= self.price_data['short'][ind_diff['short']:]
                    res_data['short'].agent.date_data= self.date_data['short'][ind_diff['short']:]

                    res_data['short'].agent.scale_input = self.scale_input['short'][ind_diff['short']:]

                    short_ind_data=[]
                    for dim in range(len(self.scale_input['short'])):
                        short_ind_data.append(self.scale_input['short'][dim][ind_diff['short']:])
                    res_data['short'].agent.scale_input = short_ind_data



                else: #같으면
                    ind_diff['short']=0
                    ind_diff['long']=0
            except:
                pass



        if __name__ == '__main__':
            if self.is_back_testing == True:  # 백테스팅일경우 출력
                fig, ax = plt.subplots(4, 1, figsize=(10, 9))
                total_dim = len(params.short_or_long_data)

                for dim in range(total_dim):
                    if params.train_stock_or_future == 'future': #학습시 선물방식일때
                        is_short_or_long = params.short_or_long_data[dim]
                    else:
                        is_short_or_long = 'long'

                    #앞에 self 붙으면 class안에 중복된 이름있기 때문에 dict이 소실됨
                    agent = res_data[is_short_or_long].agent
                    PV_data = res_data[is_short_or_long].PV_data[is_short_or_long]
                    buy_data = res_data[is_short_or_long].buy_data[is_short_or_long]
                    buy_date = res_data[is_short_or_long].buy_date[is_short_or_long]
                    sell_data = res_data[is_short_or_long].sell_data[is_short_or_long]
                    sell_date = res_data[is_short_or_long].sell_date[is_short_or_long]
                    price_data= res_data[is_short_or_long].agent.price_data.view(-1)

                    if dim == 0:  # 처음에만 출력
                        if len(params.short_or_long_data) > 1 :  #헷지모드 on( 롱숏 둘다 백테스트 하며 각각 PV연산하여 합산)
                            long_agent = res_data['long'].agent
                            long_PV_data = res_data['long'].PV_data['long']

                            short_agent = res_data['short'].agent
                            short_PV_data = res_data['short'].PV_data['short']

                            long_data_date = long_agent.date_data
                            short_data_date = short_agent.date_data

                            # 길이 일치
                            PV_data_set = [long_PV_data, short_PV_data]
                            date_set = [long_data_date, short_data_date]

                            less_data = PV_data_set[np.argmin([len(long_PV_data), len(short_PV_data)])]  # 갯수가 더 적은 데이터
                            more_data = PV_data_set[np.argmax([len(long_PV_data), len(short_PV_data)])]

                            more_date = date_set[np.argmax([len(long_data_date), len(short_data_date)])]  # 갯수 더 많은 날짜 데이터

                            len_diff = np.abs(len(long_PV_data) - len(short_PV_data))  # 차이

                            less_data = torch.cat([torch.zeros(len_diff), torch.Tensor(less_data).view(-1)]).view(-1)  # 적었던 PV데이터 길이 일치
                            if len_diff ==0: #차이가 없는경우
                                more_data=long_PV_data
                                less_data=short_PV_data
                            res_PV = torch.Tensor(more_data).view(-1) + torch.Tensor(less_data).view(-1)  # PV 합
                            res_date = more_date

                            ax[0].set_ylabel('NI of short and long Agent')
                            ax[0].plot(res_date, (res_PV - (long_agent.init_cash + short_agent.init_cash)))

                            series_PV=pd.Series(res_PV-(long_agent.init_cash+short_agent.init_cash))
                            series_date=pd.Series(res_date)

                            day_count = int(round(1440 / params.minute))
                            week_count = int(round(7200 / params.minute))
                            month_count = int(round(31000 / params.minute))



                            ################daily

                            # 가장 빠른 00시 데이터 출력

                            day_series_date = pd.to_datetime(series_date)  # 날짜 형식으로 변환
                            dates = day_series_date.dt.date.unique()

                            # 각 날짜별로 00시 데이터 추출 및 중복 제거
                            one_day_before_00_data = []
                            for date in dates:
                                one_day_before_00_data.append(day_series_date[(day_series_date.dt.date == date) & (
                                            day_series_date.dt.hour == 0)].min())

                            # 중복 제거
                            one_day_before_00_data = pd.Series(one_day_before_00_data).drop_duplicates()

                            # 하루 전 가장 빠른 00시 데이터의 인덱스들을 리스트에 저장
                            one_day_before_indices = []
                            for data in one_day_before_00_data:
                                indices = day_series_date[
                                    (day_series_date.dt.date == data.date()) & (day_series_date.dt.hour == 0)].index
                                if len(indices) > 0:
                                    one_day_before_indices.append(indices[0])

                            print(series_date[one_day_before_indices])
                            print(one_day_before_indices,'하루전 인덱스')





                            series_date2 = pd.to_datetime(series_date)  # 날짜 형식으로 변환

                            week_date_idx = []
                            prev_date = None

                            for idx, date in enumerate(series_date2):
                                if prev_date is None or (prev_date - date).days >= 7:
                                    week_date_idx.append(idx)
                                    prev_date = date

                            print(week_date_idx)
                            print(series_date[week_date_idx],'week')






                            ###########month
                            month_date_idx = []
                            prev_month = None

                            for idx, date_str in enumerate(series_date):
                                date = pd.to_datetime(date_str)
                                if prev_month is None or prev_month != date.month:
                                    month_date_idx.append(idx)
                                    prev_month = date.month

                            print(series_date[month_date_idx])
                            print(month_date_idx,'한달전 인덱스')

                            daily_date = series_date[one_day_before_indices].reset_index()[0]  # 1일
                            weekly_date = series_date[week_date_idx].reset_index()[0]  # 약 1주일
                            month_date = series_date[month_date_idx].reset_index()[0] # 한달

                            daily_NI = pd.Series([series_PV[one_day_before_indices[step]] - series_PV[one_day_before_indices[step+1]] for step in range(len(one_day_before_indices)-1)])
                            weekly_NI = pd.Series([series_PV[week_date_idx[step]] - series_PV[week_date_idx[step+1]] for step in range(len(week_date_idx)-1)])
                            monthly_NI = pd.Series([series_PV[month_date_idx[step]] - series_PV[month_date_idx[step+1]] for step in range(len(month_date_idx)-1)])

                            daily_csv = pd.concat([daily_date, daily_NI], axis=1)
                            daily_csv.columns = ['date', 'daily NI']
                            weekly_csv = pd.concat([weekly_date, weekly_NI], axis=1)
                            weekly_csv.columns = ['date', 'weekly NI']
                            monthly_csv = pd.concat([month_date, monthly_NI], axis=1)
                            monthly_csv.columns = ['date', 'monthly NI']

                            daily_csv.to_csv('z_daily NI_backtest result')
                            weekly_csv.to_csv('z_weekly NI_backtest result')
                            monthly_csv.to_csv('z_monthly NI_backtest result')

                            print(daily_csv)
                            print(monthly_csv)
                            print('APPO_LS total NI :',res_PV[-1] - (long_agent.init_cash + short_agent.init_cash))

                        elif len(params.short_or_long_data) == 1:  # short or long 인경우
                            ax[0].set_ylabel('NI_of_' + is_short_or_long + '_Agent')
                            ax[0].plot(np.array(agent.date_data).reshape(-1), (torch.Tensor(PV_data) - agent.init_cash))

                        ax[3].set_ylabel('long input')
                        label_1 = params.long_ind_name[0]
                        ax[3].plot(np.array(agent.date_data).reshape(-1), np.array(agent.scale_input[0][agent.window-1:]),
                                   label=label_1)

                        for d in range(len(agent.scale_input)):
                            label_ = params.long_ind_name[d] if params.long_ind_name else f"Data {d}"
                            if d > 0:  # 두번째 부터 출력(첫번째는 이전에 출력했음)
                                ax[3].plot(np.array(agent.scale_input[d][agent.window-1:]), label=label_)
                        ax[3].legend(loc='upper left')


                    if is_short_or_long == 'long':  # 롱인경우
                        ax[1].set_ylabel(is_short_or_long + '_AI')
                        ax[1].scatter(buy_date, buy_data, marker='v', color='red')
                        ax[1].scatter(sell_date, sell_data, marker='v', color='blue')
                        ax[1].plot(price_data)


                    if is_short_or_long == 'short':  # 숏인경우
                        ax[2].set_ylabel(is_short_or_long + '_AI')
                        ax[2].scatter(buy_date, buy_data, marker='v', color='blue')
                        ax[2].scatter(sell_date, sell_data, marker='v', color='red')
                        ax[2].plot(price_data)

                plt.show()


    def mul_back_test(self):  # 병렬 백테스트( 숏 롱 )
        process_list = []
        res_data={}
        long_res=multiprocessing.Queue()  #결과 저장
        short_res=multiprocessing.Queue()

        print('병렬 백테스트 시작.', '코어수:', multiprocessing.cpu_count())




        if params.backtest_hedge_on_off=='on':
            if params.multi_or_thread=='multi': #멀티 프로세싱인경우 (메모리 많이잡아먹음)
                if __name__ == '__main__':
                    for is_short_or_long in params.short_or_long_data:  # func 목록 모두 저장
                        proc = multiprocessing.Process(target=self.back_test,args=([is_short_or_long, long_res,short_res]))
                        process_list.append(proc)
                    [process.start() for process in process_list]

                    for step in range(len(params.short_or_long_data)): #숏, 롱 데이터 큐에 저장
                        is_short_or_long = params.short_or_long_data[step]
                        if is_short_or_long == 'long':
                            res_ = long_res.get()[0]
                        if is_short_or_long == 'short':
                            res_ = short_res.get()[0]
                        res_data[is_short_or_long] = res_

                    [process.join() for process in process_list] #종료

            else: #멀티프로세싱 아닌경우
                long_res_ = []
                short_res_ = []
                for is_short_or_long in params.short_or_long_data: #스레드 시뮬레이션 ( 메모리 낮은경우 대비)
                    long_res,short_res=self.back_test(is_short_or_long,long_res_,short_res_)
                    try:
                        if is_short_or_long == 'long':
                            res_ = long_res[0]
                        if is_short_or_long == 'short':
                            res_ = short_res[0]
                        res_data[is_short_or_long] = res_
                    except:
                        res_data=0

        return res_data



from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as TK

if __name__=='__main__':
    # 결과 출력

    root = TK.Tk() #백테스트 결과 창띄우기

    # npy 저장 불러오기
    csv_agent_name= 'z_result.csvlong_2'
    df = pd.read_csv(csv_agent_name)
    cumul_epi_reward = df['self.epi_reward_data'].tolist()
    cumul_reward = df['self.cumul_reward_data'].tolist()
    cumul_policy_loss = df['self.policy_loss_data'].tolist()
    cumul_value_loss = df['self.value_loss_data'].tolist()
    cumul_PV = df['self.cumul_PV_data'].tolist()


    fig, ax = plt.subplots(5, 1, figsize=(10, 9))

    ax[0].set_ylabel('PV data traj')
    ax[0].plot(cumul_PV)

    ax[1].set_ylabel('reward')
    ax[1].plot(cumul_reward)

    ax[2].set_ylabel('policy loss')
    ax[2].plot(cumul_policy_loss)

    ax[3].set_ylabel('value loss')
    ax[3].plot(cumul_value_loss)


    ###########창으로 출력
    new_window = TK.Toplevel(root)  # 새로운 Toplevel 윈도우 생성
    new_window.title(f'창 ')

    canvas = FigureCanvasTkAgg(fig, master=new_window)  # 새 윈도우를 마스터로 설정
    canvas.draw()
    canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)

    # Navigation toolbar 추가
    toolbar = NavigationToolbar2Tk(canvas, new_window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=TK.TOP, fill=TK.BOTH, expand=1)


    bk = back_testing(params.back_train_val_test)
    res_data=bk.mul_back_test()
    bk.res_plot(res_data)

    plt.show()
    TK.mainloop() #창 유지 플랏


#추가해야될 기능 ,슬리피지, 월간 일간 주간 ,손익계산, 승률 , 거래횟수 , MDD, 롱수익 , 숏 수익


#로컬이 학습하고 옮긴다음 검증?
#CCI Neo 가지고 학습하는데