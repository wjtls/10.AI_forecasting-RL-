import c_PPO_Agent as PA
import numpy as np
import torch
import b_network as NET
import e_train as et

PPO=PA.PPO

import random
import torch
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class APPO():
    def __init__(self,
                 window,  # LSTM 윈도우 사이즈
                 cash,  # 초기 보유현금
                 cost,  # 수수료 %
                 device,  # 디바이스 cpu or gpu
                 k_epoch,  # K번 반복
                 long_input_,  # 인풋 데이터 리스트 모음
                 short_input_,
                 train_val_test,  # 데이터셋 이름
                 input_dim,  # feature 수
                 coin_or_stock,
                 Agent_num,  # 에이전트당 학습할 데이터 수
                 deposit,
                 slippage,
                 short_or_long_data
                 ):
        self.Global_policy_net={}
        self.Global_value_net={}
        self.short_or_long_data=short_or_long_data

        short_train_data, short_val_data, short_test_data, short_ori_close, short_total_input_, short_date_data, short_total_date = short_input_
        short_input_ = [short_train_data, short_val_data, short_test_data]

        long_train_data, long_val_data, long_test_data, long_ori_close, long_total_input_, long_date_data, long_total_date = long_input_
        long_input_ = [long_train_data, long_val_data, long_test_data]


        if train_val_test == 'train':
            self.long_price_data = long_ori_close[0]
            self.long_scale_input= long_input_[0]
            self.long_date_data= long_date_data[0]

            self.short_price_data = short_ori_close[0]
            self.short_scale_input = short_input_[0]
            self.short_date_data = short_date_data[0]



        elif train_val_test == 'val':
            self.long_price_data = long_ori_close[1]
            self.long_scale_input = long_input_[1]
            self.long_date_data = long_date_data[1]

            self.short_price_data = short_ori_close[1]
            self.short_scale_input = short_input_[1]
            self.short_date_data = short_date_data[1]

        elif train_val_test == 'test':
            self.long_price_data = long_ori_close[2]
            self.long_scale_input = long_input_[2]
            self.long_date_data = long_date_data[2]

            self.short_price_data = short_ori_close[2]
            self.short_scale_input = short_input_[2]
            self.short_date_data = short_date_data[2]

        elif train_val_test == 'total':
            self.long_price_data = torch.cat([long_ori_close[0], long_ori_close[1], long_ori_close[2]])
            self.long_scale_input=long_total_input_
            self.long_date_data=long_total_date

            self.short_price_data = torch.cat([short_ori_close[0], short_ori_close[1], short_ori_close[2]])
            self.short_scale_input = short_total_input_
            self.short_date_data = short_total_date

        self.window = window

        for short_long in short_or_long_data:
            self.window=self.window[short_long]

        self.dim_data = input_dim
        self.k_epoch=k_epoch
        self.device=device
        self.cost=cost
        self.cash=cash
        self.deposit=deposit
        self.slippage=slippage
        self.coin_or_stock=coin_or_stock

        # 파라미터
        self.Agent_num= Agent_num
        self.Agent_data_num = 0  # 각 에이전트당 총 학습 or 시뮬레이션 할 데이터수
        self.Agent_name = []
        self.Agent_input_data = {}
        self.Agent_infor = {}
        self.train_val_test = train_val_test

    def reset(self):
        self.Agent_name = []
        self.Agent_input_data = {}
        self.Agent_infor = {}

    def define_Agent_num(self,short_or_long):  #각 에이전트 넘버 및 글로벌 네트워크,데이터 처리

            if short_or_long=='short':
                self.price_data=self.short_price_data
                self.scale_input=self.short_scale_input
                self.date_data= self.short_date_data
                self.dim= self.dim_data[short_or_long]

            elif short_or_long=='long':
                self.price_data= self.long_price_data
                self.scale_input= self.long_scale_input
                self.date_data= self.long_date_data
                self.dim = self.dim_data[short_or_long]

            idx = 0
            #Agent_num_len, div = divmod(len(self.price_data), self.Agent_data_num)
            Agent_num_len = self.Agent_num
            self.Agent_data_num=round(len(self.price_data)/Agent_num_len)

            for Agent_num in range(Agent_num_len):  # Agent 별로 인풋 데이터 저장
                    self.Agent_name.append(short_or_long + '_'+str(Agent_num + 1))
                    # 데이터 기간 수정
                    input_data_ = []
                    if idx==0:  # 처음인경우 윈도우사이즈만큼 제외하고 데이터 넣음
                        for dim_ in range(self.dim):
                            if Agent_num == int(Agent_num_len) - 1:  # 마지막 인덱스인경우 남은 데이터 전체
                                input_data_.append(self.scale_input[dim_][idx:])
                                price_data = self.price_data[idx:]
                                date_data = self.date_data[idx:]
                            else:
                                input_data_.append(self.scale_input[dim_][idx:self.Agent_data_num + idx+self.window])
                                price_data = self.price_data[idx:self.Agent_data_num + idx+ self.window]
                                date_data = self.date_data[idx:self.Agent_data_num + idx+ self.window]
                        idx+=self.window # 처음 데이터가 윈도우만큼 사라지는거 방지하여 윈도우만큼 더뽑고 idx에 사이즈 더한다


                    else:   #그다음부터 윈도우사이즈 + agent당 데이터 크기 만큼 넣음(윈도우 사이즈로 인한 데이터 누락 없엔다)
                        for dim_ in range(self.dim):
                            if Agent_num == int(Agent_num_len) - 1:  # 마지막 인덱스인경우 남은 데이터 전체
                                input_data_.append(self.scale_input[dim_][idx-self.window+1:])
                            else:
                                input_data_.append(self.scale_input[dim_][idx-self.window+1:self.Agent_data_num + idx])

                            if Agent_num == int(Agent_num_len) - 1:  # 마지막 인덱스인경우 남은 가격데이터 전체
                                price_data = self.price_data[idx-self.window+1:]
                                date_data = self.date_data[idx-self.window+1:]
                            else:
                                price_data = self.price_data[idx-self.window+1:self.Agent_data_num + idx]
                                date_data = self.date_data[idx-self.window+1:self.Agent_data_num + idx]



                    self.Agent_input_data['Agent_' + short_or_long +'_'+ str(Agent_num + 1) +'_input'] = input_data_
                    self.Agent_input_data['Agent_' + short_or_long +'_'+ str(Agent_num + 1) +'_price'] = price_data
                    self.Agent_input_data['Agent_' + short_or_long +'_'+ str(Agent_num + 1) +'_date'] = np.array(date_data)

                    idx += self.Agent_data_num # 에이전트에 넣을 데이터 길이


    def distribute_Agent(self):  #에이전트 불러오기
        self.reset()
        input_dim=0
        Global_lr=0


        for short_or_long in self.short_or_long_data:  #
            self.define_Agent_num(short_or_long)
            if short_or_long=='long':
                input_dim=len(et.long_ind_name)
                Global_lr=et.long_Global_actor_net_lr
                self.window= et.window[short_or_long]
            if short_or_long=='short':
                input_dim=len(et.short_ind_name)
                Global_lr=et.short_Global_actor_net_lr
                self.window = et.window[short_or_long]

            # global net
            '''''
            if torch.cuda.device_count() > 1 and self.device == 'cuda':
                Global_actor = nn.DataParallel(NET.Global_actor).module
                Global_critic = nn.DataParallel(NET.Global_critic).module
            '''''
            Global_actor = NET.Global_actor
            Global_critic = NET.Global_critic

            self.Global_policy_net[short_or_long] = Global_actor('cpu', self.window, input_dim, short_or_long, et.Neural_net, et.bidirectional_).share_memory()
            self.Global_value_net[short_or_long] = Global_critic('cpu', self.window, input_dim, short_or_long, et.Neural_net, et.bidirectional_).share_memory()

        num_cuda =et.num_cuda





        for step,Agent_num in enumerate(self.Agent_name):  # self.Agent_name= 문자열 데이터

            idx_=Agent_num.index('_')
            short_or_long=Agent_num[:idx_].replace('_','')

            if torch.cuda.device_count() > 1 and num_cuda>1 and et.device == 'cuda': # 멀티 gpu

                if et.device == 'cuda':
                    cuda_ = step % num_cuda
                    self.device = f"cuda:{cuda_}"



            self.Agent_infor['Agent_' + Agent_num] = PPO(self.window,  # LSTM 윈도우 사이즈
                                                         self.cash,  # 초기 보유현금
                                                         self.cost,  # 수수료 %
                                                         self.device,  # 디바이스 cpu or gpu
                                                         self.k_epoch,  # K번 반복
                                                         self.Agent_input_data['Agent_' + Agent_num + '_input'],
                                                         # 인풋 데이터
                                                         self.Agent_input_data['Agent_' + Agent_num + '_price'],
                                                         # 주가 데이터
                                                         self.Agent_input_data['Agent_' + Agent_num + '_date'],  # 날짜
                                                         self.dim_data[short_or_long],  # feature 수
                                                         Agent_num,
                                                         self.coin_or_stock,
                                                         self.deposit,
                                                         self.slippage,  # 슬리피지
                                                         short_or_long,  # 숏인지 롱인지
                                                         self.Global_policy_net[short_or_long],
                                                         # 글로벌넷
                                                         self.Global_value_net[short_or_long],
                                                         # 글로벌넷
                                                         False     # 백테스팅이면 True
                                                         )


    def res_params(self): # 에이전트 분배 결과
        self.distribute_Agent() #에이전트 dict , params

        return self.Agent_infor,self.Agent_name

