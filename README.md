# AI_predictor-trading-
인공지능으로 예측 및 최적행동 도출


## 1. 요약
![image](https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/6d1c7ae3-f81b-4b4a-b03f-7b492b4042bc)
- 커스텀 지표 또는 사용자가 지정한 데이터를 참고하여 예측이 필요한 환경에서 다수 에이전트가 비동기적으로 수요 또는 가격등을 학습하고 예측값을 도출하며 최적의사결정까지 진행한다.
<br/><br/><br/><br/>


## 2. 사용 알고리즘 구조
![image](https://github.com/wjtls/10.AI_predictor-trading-/assets/60399060/617dd02e-4db8-4138-aced-a32f53a123c7)
   
<br/><br/><br/><br/>



## 3. 오버피팅 문제 완화 노력

   - 
   - 리플레이 버퍼

   
   - ![image](https://user-images.githubusercontent.com/60399060/146135720-9f131c45-c616-4383-bf87-f9235cf7f55f.png)
   - new policy가 old policy 와 크게 다르지않도록 Clipping 하기 때문에 논문에서 안정성이 높고 빠르다는 결과를 보인다. <br/>
   
   - ![image](https://user-images.githubusercontent.com/60399060/146135945-5e1bd0e9-8ef7-49c2-9d41-b2ae8ebb9f25.png)
   - GAE Advantage를 사용하여 Advantage를 잘추산한다. 이로인해 분산을 더 적절하게 감소 시킬수 있다.
   - 신뢰 지역(Trust region) 에서 GAE를 구하고 r세타를 연산하는 덕에 buffer를 사용할수 있고 next batch에서 좋지않은 policy를 뽑을 경우 재사용하지 않는다
 
   - ![image](https://user-images.githubusercontent.com/60399060/146136194-aa3647e1-29a8-45f4-a21c-6d38884ab353.png)
   - 새로운 정책이 기존 정책에서 너무 멀리 바뀌는 것을 피하기 위해 대리 목표를 활용하여 min을 취함으로 샘플 효율성 문제를 해결한다.
   - 정책 업데이트를 정규화하고 교육 데이터를 재사용할 수 있기 때문에 대리 목표는 핵심 기능이다. 따라서 on policy 알고리즘 이지만 on policy 의 수렴성과 대리목표(Surrogate loss) 사용으로 off policy의 장점인 샘플 효율성을 가지게 된다.




## 4. 결과





## 5. 개선방안





## 6. 업로드 파일 참고 사항
   - f_backtest.py 실행시 저장된 데이터를 사용한 백테스트 및 결과확인 가능
   - 학습 코드 및 실전 사용 코드 업로드 하지만 사용은 차단 (현 업로드 파일에서는 API key 및 지표 비활성화)
     
