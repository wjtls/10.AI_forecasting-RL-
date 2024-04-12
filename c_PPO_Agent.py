#정규화 및 전처리 계산
import pandas as pd

import e_train as params
import copy

# 시각화및 저장,계산
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# 정규화 및 전처리 계산
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

####################################외부 py 호출
import a_Env as Env_
import b_network as NET
import pandas as pd

#######################################################################################################################################
Env=Env_.Env
PPO_actor=NET.PPO_actor
PPO_critic=NET.PPO_critic
Global_actor=NET.Global_actor
Global_critic=NET.Global_critic

from collections import defaultdict
import random
import torch
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PPO(nn.Module, Env):
    '''''
    -Additional skills 

    1. Value function clipping :implementation instead fits the value network with a PPO-like objective

    2. Reward scaling        
    3. Reward Clipping :The implementation also clips the rewards with in a preset range (usually [−5,5] or [−10,10])

    4. Observation Normalization 
    5. Observation Clipping:
    6. Hyperbolic tan activations :exploration 좀더 잘할수 있도록함
    7. Global Gradient Clipping 


    8. Adam learning rate annealing
    9. Orthogonal initialization and layer scaling

    '''''

    def __init__(self, window,  # LSTM 윈도우 사이즈
                 cash,  # 초기 보유현금
                 cost,  # 수수료 %
                 device,  # 디바이스 cpu or gpu
                 k_epoch,  # K번 반복
                 input_,  # 인풋 데이터
                 price_data,  # 주가 데이터
                 date_data, #날짜 데이터
                 input_dim,  # feature 수 (숏이나 롱의 dim 하나만 들어온다 ( APPO py 에선 dict으로 들어옴)
                 Agent_num, #에이전트 넘버
                 coin_or_stock,# 코인인지 주식인지
                 deposit, #증거금
                 slippage, #슬리피지
                 if_short_or_long,
                 Global_actor,
                 Global_critic,
                 is_back_testing
                 ):

        # 클래스 상속
        nn.Module.__init__(self)
        Env.__init__(self)
        self.is_back_testing=is_back_testing #백테스트면 True 반환
        self.short_or_long=if_short_or_long

        self.input = price_data
        self.scale_input = input_
        self.window = window
        self.dim = input_dim
        self.price_data = self.input[self.window - 1:]  # 종가 데이터
        self.LSTM_input = self.LSTM_observation(self.scale_input, self.window, self.dim)  # LSTM 데이터

        # 에이전트 변수
        self.Agent_num = Agent_num  # 몇번째 Agent 인가

        self.init_cash=cash
        self.init_cost=cost
        self.init_slip=slippage

        self.date_data=date_data[self.window-1:] # 백테스팅때 사용


        
        '''''
        확인함수
        pd.set_option('display.max_columns', None)  # 모든열 출력
        pd.set_option('display.max_row', None)  # 모든행 출력

        a=pd.DataFrame(self.price_data)
        b=pd.DataFrame(self.date_data)
        d=pd.DataFrame(self.scale_input[0][self.window-1:])
        c=pd.concat([a,b,d],axis=1)
        print(c)
        print(len(a),len(b),len(d))
        '''''

        self.cash = cash  # 가진 현금
        self.cost = cost  # 수수료 비용
        self.deposit=deposit # 증거금
        self.stock = 0  # 가진 주식수
        self.slip=slippage # 슬리피지
        self.PV = self.cash  # 현 포트폴리오 벨류 저장

        #롱포지션
        self.long_price=[]
        self.long_aver_price = 0
        self.long_unit = 0

        # 숏 포지션이 있을경우
        self.stock = 0
        self.short_unit = 0
        self.short_price = []  # 매수했던 가격*계약수
        self.short_aver_price = 0

        self.Global_actor=Global_actor
        self.Global_critic= Global_critic

        self.past_PV = self.cash  # 한스탭이전 포트폴리오 벨류 (초기는 현금)
        self.gamma = 0.99
        self.Cumulative_reward = 0  # 누적 리워드
        self.old_prob = 0  # old prob 저장
        self.total_loss = 0  # actor_loss+critic_loss
        self.back_testing = False  # 백테스팅 or 학습일경우 False
        self.coin_or_stock = coin_or_stock

        self.Advantage_hat = []
        self.Advantage = 0
        self.target_data = []

        #학습때 사용 데이터
        self.next_step = []
        self.action_data = []
        self.reward_data = []
        self.step_data = []
        self.PV_list=[self.cash]

        #최종 출력때 사용 데이터
        if self.is_back_testing==False:
            self.cumul_data=defaultdict(list)   #dict에 리스트들어가있는 형태

        #학습 파라미터
        self.epsilon = 0.2  # PPO의 입실론
        self.lambda_ = 0.98  # 람다값
        self.K_epoch = k_epoch
        self.idx = 0  # idx (policy 추출위함)
        self.device = device
        self.Global_lr=0

    def reset(self):
        self.cash = self.init_cash  # 가진 현금
        self.cost = self.init_cost  # 수수료 퍼센트
        self.PV = self.init_cash  # 포트폴리오 벨류 저장
        self.slip=self.init_slip
        self.past_PV = self.cash  # 이전 포트폴리오 벨류 (초기는 현금과같음))
        self.long_price = []
        self.deposit=self.deposit #증거금
        self.long_aver_price=0
        self.long_unit=0
        self.stock = 0  # 가진 주식수
        self.step_data = []
        self.gamma = 0.99
        self.Cumulative_reward = 0  # 누적 리워드
        self.back_testing = False

        self.old_prob = []
        self.next_step = []
        self.action_data = []
        self.reward_data = []
        self.step_data = []
        self.PV_list = [self.cash]
        self.idx = 0

        self.tur_value = 0
        self.tur_yt_data = []

        self.Advantage_hat = []
        self.Advantage = 0
        self.target_data = []

        # 숏 포지션이 있을경우
        self.stock = 0
        self.short_unit = 0
        self.short_price = []  # 매수했던 가격*계약수
        self.short_aver_price = 0



    def SC_decide_action(self, policy):  #discrete decide action  stock이나 코인일때

        policy1 = torch.clamp(policy, 0, 1)
        action_s = Categorical(policy1)
        action = action_s.sample()  # 매도시 최소 1주 매도
        policy1 = policy1.to('cpu')

        self.sell_policy=policy1[0]
        self.buy_poliy=policy1[2]

        max_trade_cash = self.init_cash  # 최대 트레이딩 가능 캐시
        limit_buy_cash = self.init_cash*1  # 스탭당 최대 매수가능 캐시
        min_buy_cash = 0  # self.init_cash*0.2 #매수시 최소 매수 캐시
        limit_sell_unit = 10000000  # 스탭당  최대 판매 유닛
        min_sell_unit = 1 #매도시 최소 판매 유닛


        매매단위설정=True  #1주당 설정금액만큼 매수(코인)
        매매시단위=self.init_cash*0.3      #1주당 금액만큼씩 매수,매도




        if self.back_testing == True:  # 백테스팅일때 최대 한개
            action = torch.argmax(policy1)

            max_trade_cash = self.init_cash  # 최대 트레이딩 가능 캐시

            limit_buy_cash = self.init_cash * 1  # 스탭당 최대 매수가능 캐시
            min_buy_cash = 0  # self.init_cash*0.2 #매수시 최소 매수 캐시

            limit_sell_unit = 10000000  # 스탭당 최대 판매 유닛
            min_sell_unit = 1  # self.init_cash*0.2 #매도시 최소 판매 유닛

            if 매매단위설정==True:
                매매시단위=self.init_cash*0.3



        if action == 0:  # 매도
            unit0 = policy1[0] * self.stock

            if params.train_stock_or_future == 'coin': # 코인인경우 (코인이지만 설정된 일정수량씩 매매)
                if 매매단위설정 == True:
                    unit0 = torch.round(unit0)

                if unit0 >= limit_sell_unit:
                    unit0 = torch.Tensor([limit_sell_unit]).to(params.device)

                if unit0 <= min_sell_unit: # 매도 너무 적은경우 최소 매도만큼
                    if unit0==0:
                        unit0=torch.Tensor([0]).to(params.device)
                    else:
                        unit0 = torch.Tensor([self.stock]).to(params.device)

                unit = [unit0.item(), 0, 0]

            else:
                unit0 = max((policy1[0] * self.stock), torch.Tensor([min_sell_unit]))
                unit0 = torch.round(unit0)
                unit = [unit0, 0, 0]

                if unit0 >= limit_sell_unit:
                    unit0 = torch.Tensor([limit_sell_unit]).to(params.device)

                if torch.Tensor([unit0]).to('cpu')*torch.Tensor([(self.price-self.slip)]).to('cpu') <= min_sell_cash:
                    unit0,_ = divmod(min_sell_cash,float( torch.Tensor([unit0]).to('cpu')* torch.Tensor([(self.price-self.slip)]).to('cpu')))


        elif action == 2:  # 매수
            limit_cash = limit_buy_cash #설정한 최대 매수가능 캐시
            limit_cash = min(limit_cash, self.cash)
            unit2 = (policy1[2] * limit_cash) / (self.price+self.slip)

            if params.coin_or_stock =='stock' and 매매단위설정==False:
                unit2 = torch.max(((policy1[2] * limit_cash) / (self.price+self.slip)), torch.Tensor([0]))
                unit2 = torch.round(unit2)

            if 매매단위설정==True:
                limit_cash = min(limit_cash,매매시단위)
                unit2 = torch.max(((policy1[2] * limit_cash) / (self.price+self.slip)), torch.Tensor([0]))
                unit2 = torch.round(unit2)

            limit_buy_unit = torch.round(limit_buy_cash/(self.price+self.slip))  #설정한 최대 매수 가능 유닛
            min_buy_unit = torch.round(min_buy_cash/(self.price+self.slip)) # 설정한 최소 매수 수량
            max_trade_unit = max(torch.round(max_trade_cash/(self.price+self.slip)),0) #최대 보유가능 수량



            if params.train_stock_or_future == 'coin':
                if unit2 >= limit_buy_unit:
                    unit2 = torch.Tensor([limit_buy_unit]).to(params.device)

                if torch.Tensor([unit2]).to(params.device) <= torch.Tensor([min_buy_unit]).to(params.device): # 매수 너무 적은경우 최소 매수량 만큼
                    if unit2==0:
                        unit2=torch.Tensor([0]).to(params.device)
                    else:
                        unit2 = torch.Tensor([min_buy_unit]).to(params.device)

                if torch.Tensor([self.stock]).to(params.device) + torch.Tensor([unit2]).to(params.device) >= torch.Tensor([max_trade_unit]).to(params.device):  # 앞으로 살 갯수 + 이미 가지고있는게 max이하여야함
                    unit2 = max_trade_unit - self.stock
                    unit2 = torch.max(torch.Tensor([unit2]),torch.Tensor([0]))

                if params.coin_or_stock == 'stock' or 매매단위설정 == True:
                    if unit2 < 1:  # 소수점단위의 주식수는 없으므로
                        unit2 = torch.Tensor([0])

                unit = [0, 0, unit2.item()]




        else:  # 관망
            unit = [0, 0, 0]
        return action, unit







    def long_decide_action(self, policy, deposit):  # 롱포지션 ,관망(청산) : 롱온리 액션
        policy1 = torch.clamp(policy, 0, 1)
        action_s = Categorical(policy1)
        action = action_s.sample()

        PV_reward= False # 맨처음 스탭에서 0.99폴리시로 리워드 한번먹고 계속 같은 행동하면 계약수가 모자라서 잃어도 리워드 감소가 없음 . 이를 방지하기 위해 이런경우 PV로 리워드 계산

        ori_unit=[0,0,0] # 리워드 계산시 사용되는 unit
        policy1=policy1.to('cpu')

        # 학습일때
        limit_long_unit = 30000  # 스탭당  최대 롱 계약
        limit_long_eq= 30000
        min_sell_cash = self.init_cash*0.2 #최소 청산 캐시
        max_trade_unit = 0.3

        if self.back_testing == True:  # 백테스팅중인 경우
            action = torch.argmax(policy1)
            limit_long_unit = 10000
            min_sell_cash = self.init_cash * 0.2  # 최소 청산 캐시
            limit_long_eq = 10000
            max_trade_unit = 0.3


        if action == 0:  # 롱 청산
            unit0 = policy1[0] * self.long_unit
            unit0 = float(unit0)

            ori_unit = [unit0, 0, 0]
            if unit0 >= limit_long_eq:
                unit0 = limit_long_eq

            if unit0*self.price <= min_sell_cash: # 매도액 너무 적은경우 최소 매도액만큼
                unit0 = min_sell_cash/(unit0*(self.price-self.slip))

            unit = [unit0, 0, 0]



        elif action == 2:  # 롱 포지션

            unit2= float(policy1[2] * self.cash)/ float(self.price+self.slip)
            unit2=float(unit2)

            if unit2 >= limit_long_unit: #유닛수가 리미트보다 크면 리미트로 제한
                unit2 = limit_long_unit

            if self.long_unit+unit2>=max_trade_unit: # 앞으로 살 갯수 + 이미 가지고있는게 max이하여야함
                unit2= max_trade_unit-self.long_unit

            unit = [0, 0, unit2]


        else:  # 관망
            unit = [0, 0, 0]

        return action, unit






    def short_decide_action(self, policy, deposit):  # 롱숏 포지션
        # PV 가 낮은데 리워드가 높은경우 : 매수시 리워드 받고나서 계속 가지고 있다가 잃는경우

        policy1 = torch.clamp(policy, 0, 1)
        action_s = Categorical(policy1)
        action = action_s.sample()

        ori_unit=[0,0,0] # 리워드 계산시 사용되는 unit
        policy1=policy1.to('cpu')

        # 학습일때
        limit_short_unit = 10000# 스탭당  최대 숏 계약
        limit_short_eq=10000
        min_sell_cash = self.init_cash*0.2 #최소 청산 캐시
        max_trade_unit=10000

        PV_reward = False

        if self.back_testing==True: # 백테스팅일때
            action = torch.argmax(policy1)
            limit_short_unit= limit_short_unit  #스탭당 최대 거래 계약수 제한
            limit_short_eq = limit_short_eq
            min_sell_cash = self.init_cash * 0.2  # 최소 청산 캐시(룰베이스에선 항상 전량청산)
            max_trade_unit = 10000  #최대 보유 유닛 제한


        if action == 0:  # 숏 청산
            unit0 = policy1[0] * self.short_unit
            unit0 = float(unit0)

            if unit0 >= limit_short_eq:
                unit0 = limit_short_eq

            if unit0 * self.price <= min_sell_cash:  # 매도액 너무 적은경우 최소 매도액만큼
                unit0 = min_sell_cash / (unit0 * (self.price+self.slip))

            unit=[unit0,0,0]



        elif action == 2:  # 숏 매수

            unit2= float(policy1[2] * self.cash)/float(self.price-self.slip)
            unit2 = float(unit2)

            if unit2 >= limit_short_unit:
                unit2 = limit_short_unit

            if self.short_unit + unit2 >= max_trade_unit:
                unit2 = max_trade_unit - self.short_unit

            unit = [0, 0, unit2]

        else:  #관망
            unit =[0,0,0]

        return action, unit


    def print_tensor_info(self,tensor, name):
        print(f"{name} shape: {tensor.shape}, device: {tensor.device}")

    def print_grad(self,model, name):
        print(f"{name} model gradients:")
        for param in model.parameters():
            if param.grad is not None:
                print(param.grad)
            else:
                print("None")


    def each_optimize(self, Global_policy_net, Global_value_net, policy_net, value_net,optimizer):
        prob = policy_net(self.LSTM_input.to(self.device))
        value = value_net(self.LSTM_input.to(self.device))
        log_policy = F.log_softmax(prob, dim=1)

        total_reward = torch.tensor(self.reward_data, dtype=self.LSTM_input.to(self.device).dtype).to(self.LSTM_input.to(self.device).device).view(-1, 1)  # 텐서 새로만들면 항상 device설정해줘야함
        total_step = self.step_data
        total_next = self.next_step
        # Advantage_hat 계산(#PPO 논문 12번식) GAE

        with torch.no_grad():
            # s시점에서 a를 취하고 다음시점에 얻는것이 리워드기 때문에 사용된 토탈 리워드는 기본적으로 R_(t+1) 이다.
            target = total_reward + self.gamma * value_net(self.LSTM_input.to(self.device))[total_next]
            delta = target.detach() - value_net(self.LSTM_input.to(self.device))[total_step]
            delta = torch.Tensor(delta.flip(dims=[0])).to(self.LSTM_input.to(self.device).device)  # 델타를 역순으로 저장한다

            self.Advantage_hat = []
            for delta_ in delta:
                self.Advantage = ((self.gamma * self.lambda_)) * self.Advantage + delta_
                self.Advantage_hat.append(self.Advantage)

        self.Advantage_hat.reverse()
        self.Advantage_hat = torch.tensor(self.Advantage_hat, dtype=torch.float).view(-1).to(self.LSTM_input.to(self.device).device)


        batch_size = params.batch_size
        n_samples = len(total_step)
        n_batches = n_samples // batch_size

        for batch_idx in range(n_batches):  # 미니배치 사용
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = (batch_idx + 1) * batch_size

            with torch.no_grad():
                batch_total_step = total_step[batch_start_idx:batch_end_idx]
                batch_old_log_prob = torch.tensor(self.old_prob[batch_start_idx:batch_end_idx], dtype=torch.float).to(self.device)
                batch_old_action = torch.tensor(self.action_data[batch_start_idx:batch_end_idx]).to(self.device).view(-1, 1)
                batch_target = target[batch_start_idx:batch_end_idx].detach()

            new_prob_ = log_policy[batch_total_step]
            new_prob = new_prob_.gather(1, batch_old_action)

            prob_ratio = torch.exp(new_prob - batch_old_log_prob.view(-1, 1)).view(-1)
            surr1 = prob_ratio * self.Advantage_hat[batch_total_step]
            surr2 = torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * self.Advantage_hat[batch_total_step]

            value_surr1 = F.mse_loss(value[batch_total_step], batch_target.detach())
            pre_step = torch.Tensor(batch_total_step) - 1
            pre_step = pre_step.tolist()

            value_surr2 = F.mse_loss(torch.clamp(value[batch_total_step], value[pre_step] - self.epsilon, value[pre_step] + self.epsilon),
                batch_target.detach())

            policy_loss = -torch.min(surr1, surr2).to(self.device)
            policy_loss = policy_loss.mean()
            value_loss = min(value_surr1, value_surr2)
            self.total_loss = policy_loss + value_loss

            optimizer.zero_grad()

            policy_loss.backward(retain_graph=True)
            value_loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(value_net.parameters(), params.value_grad_clip)
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), params.policy_grad_clip)
            self.share_grad(Global_policy_net, policy_net)  # Global update : policy 가중치를 Global로
            self.share_grad(Global_value_net, value_net)

            optimizer.step()

        if self.device != 'cpu':
            torch.cuda.empty_cache()


        return policy_loss,value_loss


    def share_grad(self,Global_net,local_net):
        for para,share_para in zip(Global_net.parameters(),local_net.parameters()):
            if share_para.grad is not None:
                if self.device=='cpu':
                    para._grad =share_para.grad.to(para.device)
                else:
                    para._grad = share_para.grad.detach().cpu().clone()
            else:
                pass



    def each_train(self, epoch, global_batch,global_steps,start_event):  # 글로벌 배치: 몇번 학습후 업데이트 할것인지
        #start_event.wait()  # 이벤트 설정까지 대기
        self.back_testing = False
        Global_policy_net = self.Global_actor  # each train 마다 갱신 x 외부에서 불러온 하나의 Global net임
        Global_value_net = self.Global_critic
        input_dim = 0
        print(self.Agent_num, 'Agnet nunm', len(self.scale_input[0]), 'input 갯수')

        input_dim = 0
        if self.short_or_long == 'short':  # 숏인경우 숏가중치 저장
            input_dim = len(params.short_ind_name)
            policy_net = copy.deepcopy(
                PPO_actor(self.device, self.window, input_dim, self.Agent_num, params.short_PPO_actor_net_lr,params.Neural_net,params.bidirectional_))
            value_net = copy.deepcopy(
                PPO_critic(self.device, self.window, input_dim, self.Agent_num, params.short_PPO_critic_net_lr,params.Neural_net,params.bidirectional_))
            self.decide_action = self.short_decide_action
            self.discrete_step = self.short_discrete_step
            self.Global_lr = params.short_Global_actor_net_lr


        elif self.short_or_long == 'long':  # 롱인경우 롱가중치 저장
            input_dim = len(params.long_ind_name)
            policy_net = copy.deepcopy(
                PPO_actor(self.device, self.window, input_dim, self.Agent_num, params.long_PPO_actor_net_lr,params.Neural_net,params.bidirectional_))
            value_net = copy.deepcopy(
                PPO_critic(self.device, self.window, input_dim, self.Agent_num, params.long_PPO_critic_net_lr,params.Neural_net , params.bidirectional_))
            self.decide_action = self.long_decide_action
            self.discrete_step = self.long_discrete_step
            self.Global_lr = params.long_Global_actor_net_lr


        if params.train_stock_or_future != 'future':
            self.decide_action = self.SC_decide_action
            self.discrete_step = self.SC_discrete_step


        # 옵티마이저
        optimizer = NET.Global_share_adam(Global_policy_net.parameters(), lr=self.Global_lr, betas=(0.92, 0.999), device='cpu')

        if params.num_cuda > 1 and params.device == 'cuda':  # 멀티 GPU인경우
            if Global_policy_net.device != policy_net.device:  # Global_policy_net, optimizer의 device 일치
                Global_policy_net = Global_policy_net.to('cpu')
                optimizer = NET.Global_share_adam(Global_policy_net.parameters(), lr=self.Global_lr,
                                                  betas=(0.92, 0.999), device='cpu')

            if Global_value_net.device != value_net.device:  # Global_value_net, optimizer의 device 일치
                Global_value_net = Global_value_net.to('cpu')
                Global_value_net.device = 'cpu'


        policy_net.optimizer=optimizer
        value_net.optimizer=optimizer
        Global_policy_net.optimizer=optimizer
        Global_value_net.optimizer=optimizer


        self.cumul_PV_data = []
        self.policy_loss_data=[]
        self.value_loss_data=[]
        self.cumul_reward_data = []
        self.epi_reward_data = []

        for epoch_step in range(epoch):  # 총 에포크 반복
            self.reset()

            ##############에피소드 생성
            for epi_step in range(len(self.price_data) - 1):  # 에피소드를 돈다(K=1인 반복횟수 에이전트. )
                with torch.no_grad():
                    prob_ = policy_net(self.LSTM_input.to(self.device)).to(self.device)

                policy = F.softmax(prob_,dim=1)
                log_prob = F.log_softmax(prob_,dim=1)
                self.price = self.price_data[[epi_step]]  # 현재 주가업데이트

                if params.train_stock_or_future=='future': #선물처럼 학습시킬경우
                    action, unit = self.decide_action(policy[epi_step],self.deposit)
                else:
                    action, unit = self.decide_action(policy[epi_step])

                # (액션 리워드 스탭 )각각저장 및 스탭실행
                action, reward, step_ = self.discrete_step(action, unit, epi_step, self)
                self.old_prob.append(F.log_softmax(prob_,dim=1)[epi_step][action])
                self.next_step.append(step_ + 1)
                self.Cumulative_reward += reward

                if epi_step%100==0:
                    print(self.Agent_num,'에이전트 넘버',epi_step,'/',len(self.price_data) - 1,'에피소드')


            for K_epoch in range(self.K_epoch):  # 논문에서 정의한 K 만큼 반복학습
                #############생성된 에피소드 학습
                policy_loss,value_loss = self.each_optimize(Global_policy_net,Global_value_net,policy_net, value_net,optimizer)
                policy_net.load_state_dict(Global_policy_net.to('cpu').state_dict())  # 글로벌을 로컬에 불러오기

            if torch.isnan(policy_loss) or torch.isnan(value_loss):
                print('policy 의 NAN 값으로 인한 정지')
                print(self.print_grad(policy_net, 'policy gradient'))
                break


            self.cumul_data['self.epi_reward_data'].append(float(reward))
            self.cumul_data['self.cumul_reward_data'].append(float(self.Cumulative_reward))
            self.cumul_data['self.policy_loss_data'].append(float(policy_loss.detach()))
            self.cumul_data['self.value_loss_data'].append(float(value_loss.detach()))
            self.cumul_data['self.cumul_PV_data'].append(float(self.PV))

            df = pd.DataFrame(self.cumul_data)
            df.to_csv('z_result.csv'+str(self.Agent_num), index=False)

            # np.save('z_result.npy'+str(self.Agent_num), self.cumul_data) # npy저장

            policy_net.save()  # save local net
            value_net.save()
            Global_value_net.save()  # 글로벌넷 저장
            Global_policy_net.save()


            print('_Agent_' + self.Agent_num + '_학습중', epoch_step + 1, '/', epoch, '진행', '리워드:', self.Cumulative_reward,
                  'PV:', f"{self.PV}")
            print(policy[:10], self.Agent_num, '폴리시')

            with torch.no_grad():
                global_prob_ = Global_policy_net(self.LSTM_input.to(Global_policy_net.device))

            global_policy = F.softmax(global_prob_, dim=1)
            print(global_policy[:10], self.Agent_num, '글로벌 폴리시')


        plt.plot(reward_data)
        plt.title('total reward')

        # old는 에피소드돌릴때 폴리시(실제했던확률)- 즉 액션할때 했던 폴리시
        # . new는 학습할때 새로뽑은 폴리시


#######################################################################################################################################




