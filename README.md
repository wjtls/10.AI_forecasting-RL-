# AI_predictor-trading-
RL 인공지능으로 예측 및 최적행동 도출 

<br/>
<br/>
-동적이고 데이터간의 상관성이 높은 환경에서 각종 시계열 예측 딥러닝 모델 LSTM CNN 등, 정상성을 가정하는 ARIMA 모델 및 LGBM 부스팅 방식들은 이와 같은 환경에서 오버피팅,편향 및 높은 분산을 보이는 등 낮은 성능을 보인다.<br/><br/>
-이는 데이터 수를 늘리고 피쳐 수를 줄이는방식, Drop out , 데이터 노이즈 감소 방식 등으로 일부 해결이 가능하지만 여전히 lagging이 발생하고 분포가 달라지거나 휩소가 발생하면 예측값이 크게 벗어나는등 예측에 큰 리스크가 있다.<br/><br/>
-또한 기존의 싱글에이전트 강화학습 방식도 하이퍼 파라미터에 매우 민감하여 실전에서 낮은 퍼포먼스를 보인다<br/><br/>
-따라서 이러한 환경에서 적응시키고자 Asynchronous Method 및 입력데이터 커스텀, 각종 분산과 샘플효율성 증가 방법을 통해 AI 예측 및 행동 최적화 프로젝트 진행한다.
<br/><br/><br/><br/>

## 1. 요약
![image](https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/6d1c7ae3-f81b-4b4a-b03f-7b492b4042bc)
- 커스텀 지표 또는 사용자가 지정한 데이터를 참고하여 예측이 필요한 환경에서 다수 에이전트가 비동기적으로 수요 또는 가격등을 학습하고 예측값을 도출하며 최적의사결정까지 진행한다.
<br/><br/><br/><br/>


## 2. 사용 알고리즘 구조
![image](https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/617dd02e-4db8-4138-aced-a32f53a123c7)
- 비평/예측 모델 Critic 뉴럴네트워크와 행동네트워크 Actor 뉴럴네트워크가 강화학습을 통해 학습한다.
- actor network는 action을 선택하고 critic network는 Q(s,a) 를 평가 한다. 이는 학습 분산을 줄여준다.(안정성)
- 크리틱 네트워크로 GAE를 추정하므로 얼마나 좋은 행동가치인지 뿐만아니라 얼마나 더 좋아질수 있는지도 고려한다.(예측)
<br/><br/><br/><br/>


## 3. 오버피팅 문제 완화 노력

   - 가장 중요하게 여긴 부분은 데이터 이다. Raw 데이터를 예측 논리에 맞게 잘 커스텀하여 데이터의 노이즈를 줄이고 repaint 현상을 막는다.  (ex. 예측시 분포에서 잘나타나지 않는 특이 변수를 설명하기위한 입력데이터 가공하는 등)
     
   - 데이터 차원이 늘어날수록 차원의 저주를 피하기 위해 주성분 분석을 통해 상관계수가 높은 데이터는 제거하여 사용
   
   - 리플레이 버퍼를 사용함으로써 샘플효율성을 증대시킨다
   
   - ![image](https://user-images.githubusercontent.com/60399060/146135720-9f131c45-c616-4383-bf87-f9235cf7f55f.png)
   - new policy가 old policy 와 크게 다르지않도록 Clipping 하기 때문에 논문에서 안정성이 높고 빠르다는 결과를 보인다. <br/>

   - ![image](https://user-images.githubusercontent.com/60399060/146135945-5e1bd0e9-8ef7-49c2-9d41-b2ae8ebb9f25.png)
   - GAE Advantage를 사용하여 Advantage를 잘추산한다. 이로인해 분산을 더 적절하게 감소 시킬수 있다
     
   - 신뢰 지역(Trust region) 에서 GAE를 구하고 r세타를 연산하는 덕에 buffer를 사용할수 있고 next batch에서 좋지않은 policy를 뽑을 경우 재사용하지 않는다
     
   - ![image](https://user-images.githubusercontent.com/60399060/146136194-aa3647e1-29a8-45f4-a21c-6d38884ab353.png)
   - 새로운 정책이 기존 정책에서 너무 멀리 바뀌는 것을 피하기 위해 대리 목표를 활용하여 min을 취함으로 샘플 효율성 문제를 해결한다.
     
   - 정책 업데이트를 정규화하고 교육 데이터를 재사용할 수 있기 때문에 대리 목표는 핵심 기능이다. 따라서 on policy 알고리즘 이지만 on policy 의 수렴성과 대리목표(Surrogate loss) 사용으로 off policy의 장점인 샘플 효율성을 가지게 된다.


<br/><br/><br/><br/>

## 4. 결과/ 여러 데이터 예측
   ![image](https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/04756254-e8de-4270-84ac-ea70a6426fa4)

   - 예측 해야할 데이터 (2020년 ~ 2024년 가격데이터, 5379개)
   - window 사이즈 
   
   <p float="left">
     <img src="https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/b6a7297a-7052-48b1-808a-a2d45530e458" width="400" />
     <img src="https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/6f476551-b9a0-4c1b-920d-a8409cfdcc39" width="400" /> 
     <img src="https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/2b59963b-01fb-4cb7-8d32-14e773316def" width="400" /> 

   </p>
   - LSTM(왼쪽) 과 GRU(두번째) 랜덤포레스트(아래 농산물가격)예측 예시 <br/>
   - 이들은 loss 감소를위해 전 시점의 가격을 도출하거나 편향이 있으므로 실제 예측을 하면 예측성능 저하 발생<br/><br/><br/>
   
   <p float="left">
     <img src="https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/d9974cd8-d623-45a4-9ebb-58ef5898d101" width="400" height='300'/>
     <img src="https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/bf9e5935-ffef-4bbb-b45c-b14641a7a7e7" width="400" height='300' /> 
   </p>

   - 현모델의 예측<br/>
   - MDD : 14%
   - PV return : 320% (수량 제한)
     
   - 왼쪽 사진 가장 위(AI 성과)는 AI가 예측한 수치를 토대로 최적행동을 하고, 가격을 잘예측하면 수치가 상승/ 예측에 실패하면 하락하게 된다. 학습구간/ 테스트셋 구간에서 잘예측했다
     
   - 왼쪽 사진 가장아래(AI 예측 수치화)는 AI가 예측한 수치를 나타낸 그래프이다. LSTM, GRU, ARIMA 등의 예측 모델에서 학습구간에서 과대적합 되기때문에 테스트셋 또는 실예측에서 심각한 성능저하를 일으킨다.
     이와 달리 데이터의 노이즈 까지 과도하게 적합시키지 않고 분산 및 표준편차가 적은 예측수치를 보이지만 여전히 오버피팅의 위험은 존재한다
     
   - 학습구간에서 최대 높이에 도달할때쯤 하락을 예측하기 시작했으며 마찬가지로 테스트구간인 최근시점 (2024.2)에서 ETH 가격 또한 추세 하락을 예측하여 실제로 향후 2달간 하락을 하는중이다.
     
   - 예측 그래프는 critic net의 출력을 그래프화 한 수치로, 이의 로그리턴을 추출, 최적 의사결정을 도출했을때 (파란색 : 하락예측 ,빨간색: 상승예측 표시) 변동구간에서 복잡성 또한 고려했음을 알수있다.
     
   - 예측은 실시간으로 계속 진행되며 매스탭마다 향후 미래가치를 고려하여 다음 스탭을 예측한다 4년 학습 후 2달간의 테스트셋 예측 데이터이므로 향후 전진분석이 더 필요하다
     

   ![image](https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/e8817954-a482-4b01-bc73-5d6ae379b0f8)

   - 4개의 에이전트중 첫번째 로컬 에이전트의 PV, reward, Critic loss, actor loss
   - 800 스탭이후부터 각 loss가 수렴하기 시작하고 PV 및 reward도 수렴을 시작한다
     <br/><br/><br/>
     
   # 여러 데이터 테스트셋 예측 (2년 구간)
   ![image](https://github.com/wjtls/10.AI_forecasting-RL-/assets/60399060/9b27cb4f-4f4a-4099-817d-159cb9041d68)
   ![image](https://github.com/wjtls/10.AI_forecasting-RL-/assets/60399060/da8cbea0-aae7-46d9-a8d2-57979db675f4)
   






<br/><br/><br/><br/>

## 5. 겪었던 문제와 개선
   - critic loss 및 actor loss 가 수렴하지 않는 현상<br/>
    = 하이퍼 파라미터 설정의 문제 확인<br/>
    = 뉴럴넷의 복잡성 확인<br/>
    = 보상함수 설정 확인<br/>
    = 가중치 소실 확인 <br/>
    = 연산중 inf 또는 None값 확인/ device 설정문제 확인  <br/>

   - 보상함수를 전체 가치로 설정하지 않고 일부 예측의 가치로 설정하여 예측 성과가 떨어지는 문제<br/>
    = 이를 전체 가치를 활용하는 보상함수로 재설정하여 문제를 해결<br/>
     
   - 비동기 학습 및 리플레이버퍼, 그래디언트 클리핑, 분산감소 방법론등을 사용했지만 동적인 환경에서 여전히 오버피팅의 위험이 존재.<br/>
    = 팀원과 함께 Raw 데이터를 전처리하여 그대로 넣지않고 새롭게 가공하여 노이즈 감소 및 논리를 가진 데이터 입력하여 완화<br/>
     
   - 10분 간격의 데이터를 학습하고 Test set 또는 실시간 데이터를 읽을때 시작 시점에 따라 성능 차이 발생 <br/>
    = 실제 학습했던 데이터 텀과 유사하게 읽어들이기 위해 학습시 마지막 기간값을 토대로 시간을 계산하여 실시간 데이터를 읽어들임<br/>

     


<br/><br/><br/><br/>
## 6. 업로드 파일 참고 사항
   - f_backtest.py 실행시 저장된 데이터를 사용한 백테스트 가능
   - 현 업로드 파일에서는 API key 및 지표.py, train.py, forward test 비활성화
     
