
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Event
import d_APPO_agent as APPO_agent
import a_Env as env_
import time
import numpy as np
import random
import torch
# cnfduswkemfdl djeltj aksed  gksqhdxnwk   3qhdwl  tlrdml ra

#
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


trading_site = 'binance'   #binance , upbit
Binance_API_key ='api_key'
Binance_Secret_key ='key'
Slack_token = "token"  # slack 토큰

API_coin_name='ETH' #코인명
API_data_name='ETH/USDT' # 바이낸스 API에서 불러올 이름####################################################

coin_or_stock='coin' # 불러올 데이터
train_stock_or_future='future' # 학습방식

real_or_train='train' #코인 불러올때 train 이면 학습구간을 API에서 바로 불러옴(DB아님) , real이면 갯수만큼 API호출
API_time_diff=[+1,5,59] # API데이터 인서트시 사용 , API인경우 +1 * x시간 y분을 더하여 표준시간으로 맞춰줘야함 (API와 학습데이터는 CME시각, yfinance는 표준시각이다 )#########################
time_diff=[] #yfinance의 시간조절 파라미터(자동설정되므로 건들 x)



PV_reward=False
PV_reward_only=True

#기능 참 거짓 설정 파라미터
traj_print= False
data_insert= False #api 데이터 실시간으로 인서트 할건지
plot_print= True
insert_yf_data= True # 실시간 데이터 인서트 허용


#금융 파라미터 (선물 - 학습시 선물 방식)
slippage=0 #슬리피지 (달러)
backtest_slippage=5 #백테스트 슬리피지(달러)  #나스닥:5 ES:12.5 골드: 10

cash=1000 #달러
cost=2.5 #수수료 달러 (선물 계산시)
coin_future_cost=0.002 #수수료 달러 (선물 계산시)
stock_cost=0.001 # 설정값 계수0.001 = 0.1% (현물인경우)

leverage = 1
point_value= 1300 # 달러 가치  GC:100 ES:50 NQ:20 ES:50 FDX(Dax):25 Dow(YM):5
deposit=1000000  # 증거금


#실전 트레이딩
limit_unit=1 #최대 보유 리미트 유닛
#학습시 파라미터
data_count=['2020-01-01 00:00', '2024-03-05 23:41']   #호출할 데이터 갯수[날짜 , count]  학습시 날짜이전 count 만큼 호출 #####################################시 분초 쓰지않음
api_data_count=99999900 #실시간 트레이딩 할때 불러올 API data 갯수( 1000개정도가 한계)

Agent_num=4 #에이전트 갯수
batch_size=30
Global_batch=5
each_epoch=300000
minute=401   #데이터의 분봉 ex) 3으로 설정하면 3분마다   311 학습됨
ratio=[2,8,1]  #train,val,test set 비율 train과 val만 넣는다
train_val_test='total'

Neural_net = 'LSTM'  #'LSTM', Quantum
bidirectional_ = True  #퀀텀이면   양방향 퀀텀LSTM이 됨
device='cuda'  #cuda , cpu(Quantum은 cpu가 더빠름)


if device =='cuda':
    print(torch.cuda.is_available(),'쿠다 설정 확인')


long_ind_name=['NNCO_up_L','NNCO_down_L','CCI_trend','CCI_trend2','tanos']
short_ind_name=['NNCO_up_S','NNCO_down_S','CCI_trend','CCI_trend2','tanos']


input_dim={'short':len(short_ind_name),'long':len(long_ind_name)}  #Network input dim  숏과 롱의 dim을 dict로 넣는다
short_or_long_data=['long']###########################################


#learning rate4
long_Global_actor_net_lr=3e-5##################################
long_PPO_actor_net_lr=0
long_PPO_critic_net_lr=0

short_Global_actor_net_lr=3e-5###############################
short_PPO_actor_net_lr= 0
short_PPO_critic_net_lr= 0


#백테스팅시 기능 파라미터
multi_or_thread='thread' #백테스팅 멀티프로세싱으로 할건지 ('multi', 'thread')
backtest_hedge_on_off= 'on' #on: 백테스팅시 각각 트레이딩해서 PV합산할것(헷지모드) , off: 트레이딩 롱 or 숏 하나만(헷지 off모드)
back_train_val_test='total'
if_real_time='True' # 실시간 데이터도 합쳐서 불러옴


#APPO param
k_epoch= 20
window={'long':15, 'short':15}
num_cuda=1


reward_bonus= 100    # 주식처럼 학습시
long_reward_bonus=1000  # 선물처럼 학습시
short_reward_bonus=1000

policy_grad_clip=0.1
value_grad_clip=0.1

long_reward_clip=1000000
short_reward_clip=1000000





from torch.multiprocessing import Manager, Process, Queue






class start_Mul():  #멀티프로세싱 클래스
    def __init__(self):
        self.idx = 0

    def reset(self):
        self.idx = 0


    def func_start(self, func_dict, Agent_name_list, global_batch, each_epoch,global_steps,start_event):
        self.reset()
        proc_list=[]
        while True:
            if self.idx < len(Agent_name_list):
                name = Agent_name_list[self.idx]
                func_ = func_dict['Agent'+'_'+str(name)]
                proc = Process(target=func_.each_train, args=([each_epoch, global_batch,global_steps,start_event]))
                proc_list.append(proc)
            else:
                break
            self.idx += 1

        return proc_list


    # dict형태로 에이전트 이름받아서 함수 실행
    def start(self, func_dict, Agent_name_list, global_batch, each_epoch,global_steps):

        start_event = Event()
        print('병렬 학습 시작.','코어수:',multiprocessing.cpu_count())
        ori_close= long_input_[3]

        try: #total 이 아닌경우
            print('총 데이터수:',len(ori_close[0])+len(ori_close[1])+len(ori_close[2]))
            print('len of train data :' ,len(ori_close[0]))
            print('len of val data:', len(ori_close[1]))
            print('len of test data:', len(ori_close[2]))

        except: # total 인경우
            print('총 데이터수:',len(ori_close))

        if __name__ == '__main__':
            start_time = time.perf_counter()
            start_time5 = time.process_time()

            # net 호출
            process_list = self.func_start(func_dict, Agent_name_list,Global_batch,each_epoch,global_steps,start_event)
            [process_.start() for process_ in process_list]
            start_event.set()
            [process_.join() for process_ in process_list]

            end_time = time.perf_counter()
            end_time5 = time.process_time()

            print(end_time, '시각')
            print(end_time5 - start_time5, '프로세스 걸린 시간')

# 실시간 데이터 학습하는거 추가해야함
# > 학습구간인 price_ai의 데이터와 전진에 해당하는 price_ai2 데이터를 합쳐서 학습 하는 기능 추가
############################################################################

if __name__ == '__main__':  #py에서 멀티프로세싱할때 또 실행되지않도록
    #데이터 호출
    if device=='cuda':
        if device == 'cuda':
            ctx = multiprocessing.get_context('spawn')
        else:
            ctx = multiprocessing

        try:
            multiprocessing.set_start_method('spawn')
            ctx = multiprocessing.get_context('spawn')
        except:
            pass

    env = env_.Env()
    is_API, data_ , ind_data = env_.load_price_data(API_data_name)  # csv를 불러올지, api를 불러올지 선택

    long_input_ = ind_data[0]
    short_input_= ind_data[1]
    ori_ind_data = ind_data[2]

    if is_API == 'API':
        data_ = env.coin_data_create(minute, data_count, real_or_train, coin_or_stock,
                                     point_value, API_data_name)  # 학습시 뽑은 history 데이터

        long_input_, short_input_, ori_ind_data = env.input_create(minute, ratio, data_count,
                                                                   coin_or_stock, point_value,
                                                                   short_ind_name, long_ind_name,
                                                                   data_)  # ind에서 높은 값을 뽑음

        ind_data = [long_input_, short_input_, ori_ind_data]
        env_.save_price_data(data_, ind_data, API_data_name)

    if train_stock_or_future=='future':
        cost_=cost# 선물 cost
    else:
        cost_=stock_cost # 주식 cost

    #APPO agent 호출
    appo=APPO_agent.APPO(window,  # LSTM 윈도우 사이즈
                        cash,  # 초기 보유현금
                        cost_,  # 수수료 %
                        device,  # 디바이스 cpu or gpu
                        k_epoch,  # K번 반복
                        long_input_,  # 인풋 데이터 리스트 모음
                        short_input_,
                        train_val_test,  # 데이터셋 이름
                        input_dim,  # feature 수
                        coin_or_stock,
                        Agent_num,  # 에이전트갯수
                        deposit,
                        slippage,
                        short_or_long_data)

    #멀티 프로세싱 파라미터
    Agent_infor,Agent_name=appo.res_params() #에이전트 분배 결과


    # 글로벌 네트워크 Queue
    try:
        global_net_queue = ctx.Queue()
    except:
        global_net_queue = Queue()

    global_steps = multiprocessing.Value('L', 0)  # 공유된 글로벌 스텝 카운터 생성
    #멀티 프로세싱 학습 시작
    mul=start_Mul()
    mul.start(Agent_infor, Agent_name, Global_batch, each_epoch, global_steps)

    # Queue에서 업데이트 된 네트워크를 받아와서 저장

