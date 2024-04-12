# 네트워크 클래스들


#정규화 및 전처리 계산


#시각화및 저장,계산
import copy
import os
import numpy as np
# 시각화및 저장,계산
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import b_Quantum_LSTM_net as QLSTM_ # 퀀텀 LSTM
import a_Env as Env_

# 정규화 및 전처리 계산

#API및 데이터 불러오기

#크롤링

from torch.nn import TransformerEncoder, TransformerEncoderLayer

import random
import torch
import random
import torch

import torchquantum as tq


seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

Env=Env_.Env



#--------------------------------------------------------------------------------------
#공용
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # positional encoding을 위한 변수들을 초기화합니다.
        max_len = int(max_len)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # positional encoding 값을 계산합니다.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # positional encoding을 텐서로 변환하고 모델의 버퍼로 등록합니다.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력에 positional encoding 값을 더해줍니다.
        x = x + self.pe[:x.size(0), :]
        return x


class Global_share_adam(torch.optim.Adam):
    def __init__(self,params,lr,betas=(0.9,0.99),eps=1e-9,weight_decay=0,device='cpu'):
        torch.optim.Adam.__init__(self,params,lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step']=0
                state['exp_avg']=torch.zeros_like(p.data).to(device)
                state['exp_avg_sq']=torch.zeros_like(p.data).to(device)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()





#-------------------------------------------------------------------------------------------
#강화학습 + 양자 강화학습



class PPO_critic(tq.QuantumModule,nn.Module):
    def __init__(self, device, window, dim, Agent_number,lr_,Neural_net, bidirectional_):
        nn.Module.__init__(self)
        self.device = device
        self.Neural_net =Neural_net
        self.bidirectional_=bidirectional_

        #LSTM GRU
        self.dim = dim
        self.hidden_size = 64
        self.hidden_size_2 = 64
        self.hidden_size_3 = 32
        self.num_layers = 1
        self.Net=0


        #트랜스포머
        dropout=0.1
        self.t_num_layers = 1
        self.nhead = 2  # 어탠션 헤드 갯수(더 다양한관점)
        self.d_model = dim # 트랜스포머의 인풋 아웃풋 벡터 차원수(뉴런) # 주로 d_model을 nhead로 나눌수있는값 사용
        self.t_hidden_size_2 = 32

        if self.Neural_net=='LSTM':  #LSTM, GRU, Transformer
            self.Net = nn.Sequential(nn.LSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, bidirectional=self.bidirectional_))

            if self.bidirectional_ == True and self.Neural_net != 'Transformer':  # 양방향일때, 트랜스포머모델이 아닐때
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 1))

            else: #양방향 아닌경우
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size , self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 1))

        if self.Neural_net=='GRU':
            self.Net = nn.Sequential(nn.GRU(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                            batch_first=True, bidirectional=self.bidirectional_))

            if self.bidirectional_ == True and self.Neural_net != 'Transformer':  # 양방향일때, 트랜스포머모델이 아닐때
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 1))

        if self.Neural_net == 'Transformer':

            encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead,dropout=dropout)
            self.Net = TransformerEncoder(encoder_layer, self.t_num_layers)

            self.Linear = nn.Sequential(nn.Linear(self.d_model , self.t_hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(self.t_hidden_size_2, 1))

        if self.Neural_net == 'Quantum':
            self.n_wires = dim

            if self.bidirectional_==True:
                self.Net = nn.Sequential(QLSTM_.BiQLSTM(input_size=self.dim, hidden_size=self.hidden_size ,num_layers=self.num_layers,
                                 batch_first=True,device=self.device))
            else:
                self.Net = nn.Sequential(
                    QLSTM_.QLSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                   batch_first=True, device=self.device))

            self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size_3, 1))

            self.measure = tq.MeasureAll(tq.PauliZ)


        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0)
                m.weight.data.fill_(1.0)

        self.Net.apply(initialize_weights)
        self.Linear.apply(initialize_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_, eps=1e-10)
        self.check_point = os.path.join('future_PPO_critic_' + Agent_number )
        self.to(device)


    def forward(self, input_):

        if self.Neural_net == 'GRU' or self.Neural_net =='LSTM':
            if self.device=='cuda':
                self.flatten_parameters()  # flatten
            p, _ = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        if self.Neural_net == 'Transformer':

            p = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        if self.Neural_net == 'Quantum':
            p, _ = self.Net(input_)
            v = self.Linear(p[:, -1, :])
        return v

    def load(self):
        self.load_state_dict(torch.load(self.check_point))


    def save(self):
        torch.save(self.state_dict(), self.check_point)

    def flatten_parameters(self):
        for module in self.Net:
            if hasattr(module, 'flatten_parameters'):
                module.flatten_parameters()





class PPO_actor(tq.QuantumModule,nn.Module):
    def __init__(self, device, window,dim, Agent_number,lr_,Neural_net, bidirectional_):
        nn.Module.__init__(self)
        self.device = device
        self.Neural_net =Neural_net
        self.bidirectional_=bidirectional_

        #LSTM GRU
        self.dim = dim
        self.hidden_size = 64
        self.hidden_size_2 = 64
        self.hidden_size_3 = 32
        self.num_layers = 1
        self.Net=0

        #트랜스포머
        dropout = 0.1
        self.t_num_layers = 1
        self.nhead = 2  # 어탠션 헤드 갯수(더 다양한관점)
        self.d_model = dim  # 트랜스포머의 인풋 아웃풋 벡터 차원수(뉴런) # 주로 d_model을 nhead로 나눌수있는값 사용
        self.t_hidden_size_2 = 32

        if self.Neural_net=='LSTM':  #LSTM, GRU, Transformer
            self.Net = nn.Sequential(nn.LSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, bidirectional=self.bidirectional_))

            if self.bidirectional_ == True and self.Neural_net != 'Transformer':  # 양방향일때, 트랜스포머모델이 아닐때
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 3))

            else:
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 3))

        if self.Neural_net=='GRU':
            self.Net = nn.Sequential(nn.GRU(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                            batch_first=True, bidirectional=self.bidirectional_))

            if self.bidirectional_ == True and self.Neural_net != 'Transformer':  # 양방향일때, 트랜스포머모델이 아닐때
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, 3))

        if self.Neural_net == 'Transformer':
            encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead, dropout=dropout)
            self.Net = TransformerEncoder(encoder_layer, self.t_num_layers)

            self.Linear = nn.Sequential(nn.Linear(self.d_model, self.t_hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(self.t_hidden_size_2, 3))


        if self.Neural_net == 'Quantum':
            self.n_wires = dim

            if self.bidirectional_==True:
                self.Net = nn.Sequential(QLSTM_.BiQLSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                 batch_first=True,device=self.device))
            else:
                self.Net = nn.Sequential(
                    QLSTM_.QLSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                   batch_first=True, device=self.device))

            self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size_3, 3))

            self.measure = tq.MeasureAll(tq.PauliZ)




        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0)
                m.weight.data.fill_(1.0)

        self.Net.apply(initialize_weights)
        self.Linear.apply(initialize_weights)


        self.optimizer = optim.Adam(self.parameters(), lr=lr_, eps=1e-10)
        self.check_point = os.path.join('future_PPO2_actor_' + Agent_number)
        self.to(device)


    def forward(self, input_):
        if self.Neural_net == 'GRU' or self.Neural_net == 'LSTM':
            if self.device=='cuda':
                self.flatten_parameters()  # flatten

            p, _ = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        if self.Neural_net == 'Transformer':

            p = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        if self.Neural_net == 'Quantum':

            '''''
            # bsz = x.shape[0]
            # x = F.avg_pool2d(x, 6).view(bsz, 16)

            use_qiskit = False

            if use_qiskit:
                x = self.qiskit_processor.process_parameterized(self.q_device, self.encoder, self.q_layer, self.measure,
                                                                x)
            else:
                x,_=self.LSTM(input_)
                xsize = x[:,-1,:].size()[0]

                x_data = torch.zeros(xsize,self.hidden*2)  # LSTM 스텝 수에 맞게 출력 텐서 생성
                for i in range(xsize):
                    x_sample = x[i, -1, :].unsqueeze(0)
                    self.encoder(self.q_device, x_sample)
                    self.Net(self.q_device)
                    x_data[i, :] = self.measure(self.q_device)  # 각 샘플에 대해 277스탭의 폴리시 측정
                v = self.Linear(x_data.to(self.device))
            '''''
            p, _ = self.Net(input_)
            v = self.Linear(p[:, -1, :])
        return v

    def load(self):
        self.load_state_dict(torch.load(self.check_point))


    def save(self):
        torch.save(self.state_dict(), self.check_point)

    def flatten_parameters(self):
        for module in self.Net:
            if hasattr(module, 'flatten_parameters'):
                module.flatten_parameters()





























# -------------------------------------------------------------------------------------------------------------------------------------
#글로벌 넷


class Global_critic(tq.QuantumModule,nn.Module):
    def __init__(self, device, window,dim,save_name,Neural_net, bidirectional_):
        nn.Module.__init__(self)
        self.device = device
        self.Neural_net =Neural_net
        self.bidirectional_=bidirectional_

        #LSTM GRU
        self.dim = dim
        self.hidden_size = 64
        self.hidden_size_2 = 64
        self.hidden_size_3 = 32
        self.num_layers = 1
        self.Net=0

        #트랜스포머
        dropout = 0.1
        self.t_num_layers = 1
        self.nhead = 2  # 어탠션 헤드 갯수(더 다양한관점)
        self.d_model = dim  # 트랜스포머의 인풋 아웃풋 벡터 차원수(뉴런) # 주로 d_model을 nhead로 나눌수있는값 사용
        self.t_hidden_size_2 = 32


        if self.Neural_net=='LSTM':  #LSTM, GRU, Transformer
            self.Net = nn.Sequential(nn.LSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, bidirectional=self.bidirectional_))

            if self.bidirectional_ == True and self.Neural_net != 'Transformer':  # 양방향일때, 트랜스포머모델이 아닐때
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 1))

            else:
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 1))



        if self.Neural_net=='GRU':
            self.Net = nn.Sequential(nn.GRU(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                            batch_first=True, bidirectional=self.bidirectional_))

            if self.bidirectional_ == True and self.Neural_net != 'Transformer':  # 양방향일때, 트랜스포머모델이 아닐때
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, 1))



        if self.Neural_net == 'Transformer':
            encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead, dropout=dropout)
            self.Net = TransformerEncoder(encoder_layer, self.t_num_layers)

            self.Linear = nn.Sequential(nn.Linear(self.d_model, self.t_hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(self.t_hidden_size_2, 1))

        if self.Neural_net == 'Quantum':
            self.n_wires = dim

            if self.bidirectional_==True:
                self.Net = nn.Sequential(QLSTM_.BiQLSTM(input_size=self.dim, hidden_size=self.hidden_size ,num_layers=self.num_layers,
                                 batch_first=True,device=self.device))
            else:
                self.Net = nn.Sequential(
                    QLSTM_.QLSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                   batch_first=True, device=self.device))

            self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size_3, 1))

            self.measure = tq.MeasureAll(tq.PauliZ)


        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0)
                m.weight.data.fill_(1.0)

        self.Net.apply(initialize_weights)
        self.Linear.apply(initialize_weights)


        self.optimizer = optim.Adam(self.parameters(), lr=3e-5, eps=1e-10)
        self.check_point = os.path.join('future_Global_critic_'+save_name)
        self.to(device)


    def forward(self, input_):
        if self.Neural_net == 'GRU' or self.Neural_net == 'LSTM':
            if self.device=='cuda':
                self.flatten_parameters()  # flatten
            p, _ = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        if self.Neural_net == 'Transformer':

            p = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        if self.Neural_net == 'Quantum':
            p, _ = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        return v

    def load(self):
        self.load_state_dict(torch.load(self.check_point))


    def save(self):
        torch.save(self.state_dict(), self.check_point)


    def flatten_parameters(self):
        for module in self.Net:
            if hasattr(module, 'flatten_parameters'):
                module.flatten_parameters()


class Global_actor(tq.QuantumModule,nn.Module):
    def __init__(self, device, window,dim,save_name,Neural_net, bidirectional_):
        nn.Module.__init__(self)
        self.dim = dim
        self.device = device
        self.Neural_net =Neural_net
        self.bidirectional_=bidirectional_

        #LSTM GRU
        self.hidden_size = 64
        self.hidden_size_2 = 64
        self.hidden_size_3 = 32
        self.num_layers = 1
        self.Net=0

        #트랜스포머
        dropout = 0.1
        self.t_num_layers = 1
        self.nhead = 2  # 어탠션 헤드 갯수(더 다양한관점)
        self.d_model = dim  # 트랜스포머의 인풋 아웃풋 벡터 차원수(뉴런) # 주로 d_model을 nhead로 나눌수있는값 사용
        self.t_hidden_size_2 = 32



        if self.Neural_net=='LSTM':  #LSTM, GRU, Transformer
            self.Net = nn.Sequential(nn.LSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, bidirectional=self.bidirectional_ ))

            if self.bidirectional_ == True and self.Neural_net != 'Transformer':  # 양방향일때, 트랜스포머모델이 아닐때
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 3))

            else:
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_3, 3))

        if self.Neural_net=='GRU':
            self.Net = nn.Sequential(nn.GRU(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                            batch_first=True, bidirectional=self.bidirectional_))

            if self.bidirectional_ == True and self.Neural_net != 'Transformer':  # 양방향일때, 트랜스포머모델이 아닐때
                self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size_2, 3))

        if self.Neural_net == 'Transformer':
            encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead, dropout=dropout)
            self.Net = TransformerEncoder(encoder_layer, self.t_num_layers)

            self.Linear = nn.Sequential(nn.Linear(self.d_model, self.t_hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(self.t_hidden_size_2, 3))

        if self.Neural_net == 'Quantum':
            self.n_wires = dim

            if self.bidirectional_==True:
                self.Net = nn.Sequential(QLSTM_.BiQLSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                 batch_first=True,device=self.device))
            else:
                self.Net = nn.Sequential(
                    QLSTM_.QLSTM(input_size=self.dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                   batch_first=True, device=self.device))

            self.Linear = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size_2, self.hidden_size_3),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size_3, 3))

            self.measure = tq.MeasureAll(tq.PauliZ)


        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0)
                m.weight.data.fill_(1.0)

        self.Net.apply(initialize_weights)
        self.Linear.apply(initialize_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-5, eps=1e-10)
        self.check_point = os.path.join('future_Global_actor_'+save_name)
        self.to(device)


    def forward(self, input_):
        if self.Neural_net == 'GRU' or self.Neural_net == 'LSTM':
            if self.device=='cuda':
                self.flatten_parameters()  # flatten

            p, _ = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        if self.Neural_net == 'Transformer':

            p = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        if self.Neural_net == 'Quantum':
            p, _ = self.Net(input_)
            v = self.Linear(p[:, -1, :])

        return v


    def load(self):
        self.load_state_dict(torch.load(self.check_point))

    def save(self):
        torch.save(self.state_dict(), self.check_point)


    def flatten_parameters(self):
        for module in self.Net:
            if hasattr(module, 'flatten_parameters'):
                module.flatten_parameters()

# -------------------------------------------------------------



