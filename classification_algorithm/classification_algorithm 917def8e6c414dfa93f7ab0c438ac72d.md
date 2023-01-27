# classification_algorithm

1. Perceptron
2. Adaline
3. standardization
4. SGD

# Perceptron Algorithm

**프랑크 로젠블라트가 MCP(간소화된 뇌의 뉴런 개념)를 기반으로 제안한 인공 신경망**

![Untitled](classification_algorithm%20917def8e6c414dfa93f7ab0c438ac72d/Untitled.png)

### 과정

퍼셉트론이 샘플 **x**를 입력받아 가중치 **w**를 연결합니다. x와 w를 점곱 형태로 연산을 한 값을 임계 함수(단위 계단 함수)로 전달되어 -1 또는 +1의 이진 출력을 만듭니다.  여기서 임계함수를 활성함수라고 생각할 수 있습니다. 만약 퍼셉트론이 클래스 레이블을 정확하게 예측한 경우 가중치의 업데이트는 일어나지 않습니다. 

### Model

```python
class Perceptron(object):
    
    # param : 학습률, 훈련 데이터셋 반복 횟수, 가중치 초기화를 위한 난수 생성 시드
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
        
        # 데이터 학습 메소드
        # param : X.shape = [n_samples, n_features], y.shape = [n_samples]
    def fit(self, X, y):
        # seed 기반 난수 생성
        rgen = np.random.RandomState(self.random_state)
        
        # np.random.normal -> 평균:0, 표준편차:0.01, 사이즈:1 + X.shape[1]인 무작위 가중치 생성
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        # epoch마다 잘못 분류된 횟수 기록
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    # 최종 입력 계산
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    # 임계 함수
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, -1)
```

### 퍼셉트론으로 붓꽃 데이터셋 분류하기

![Untitled](classification_algorithm%20917def8e6c414dfa93f7ab0c438ac72d/Untitled%201.png)

위 산점도와 같이 꽃받침 길이와 꽃잎 길이에 따라 분포된 데이터로 퍼셉트론 훈련을 해보겠습니다. 

![Untitled](classification_algorithm%20917def8e6c414dfa93f7ab0c438ac72d/Untitled%202.png)

n_iter = 10, eta = 0.1로 훈련한 결과입니다. 위 그래프에서 알 수 있는 것은, 6번째 epoch이후 오차가 수렴하였고 훈련 샘플을 완벽하게 분류한 것을 알 수 있습니다.

마지막으로 2차원 데이터셋의 결정 경계를 시각화해 보겠습니다.

![Untitled](classification_algorithm%20917def8e6c414dfa93f7ab0c438ac72d/Untitled%203.png)

퍼셉트론이 두 개의 붓꽃 클래스를 완벽하게 분류하였지만 단점이 존재합니다. 위의 그래프를 예시로 보면, 1개의 직선으로 완벽하게 두 클래스가 구분될 경우에만 수렴한다는 단점입니다. 만약 선형 결정 경계로 구분 완벽하게 구분되지 않는다면 학습은 멈추지 않을 것이며, 최대 에포크를 지정해야 할 것입니다.

# Adaline Algorithm

**퍼셉트론의 향상된 버전이며 퍼셉트론과 달리 가중치 업데이트에 선형 활성화 함수 사용**

![Untitled](classification_algorithm%20917def8e6c414dfa93f7ab0c438ac72d/Untitled%204.png)

진짜 클래스 레이블과 예측 클래스 레이블을 비교하는 퍼셉트론과 달리, 진짜 클레스 레이블과 활성화 함수의 출력 값을 비교하여 모델의 오차를 계산하고 가중치를 업데이트 합니다. 비용 함수로써 오차제곱합(아래 수식)을 사용합니다. 

![Untitled](classification_algorithm%20917def8e6c414dfa93f7ab0c438ac72d/Untitled%205.png)

### Model

```python
class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])        
        self.cost_ = []
        
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # 오차 제곱합 구현, 2.0으로 나눈 것은 그래디언트 간소화 목적
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
```

### 학습률 조정을 통하여 아달린의 에포크 횟수 대비 비용 함수의 값 보기

![Untitled](classification_algorithm%20917def8e6c414dfa93f7ab0c438ac72d/Untitled%206.png)

왼쪽 그래프는 비용함수를 최소화하지 못하고 오차는 에포크마다 점점 더 커지는 것을 알 수있는데 이것을 통하여 전역 최솟값을 지나쳤다는 것을 알 수 있습니다.

오른쪽 그래프는 비용이 감소하고 있지만 학습률이 너무 작기 때문에 전역 최솟값에 수렴하기 위해서는 많은 에포크가 필요할 것입니다.

# standardization

특성 스케일 방법으로, 데이터에 표준 정규 분포 성질을 부여하여 경사 하강법이 더 빠르게 수렴하도록 하기 위한 방법

- 각 특성의 평균을 0에 맞추고 표준 편차를 1로 만듭니다

```python
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

```

# SGD(Stochastic Gradient Descent)

optimizer의 한 종류로써 가장 기본적인 방법이다. 

![Untitled](classification_algorithm%20917def8e6c414dfa93f7ab0c438ac72d/Untitled%207.png)

모든 샘플에 대하여 누적된 오차합을 기반으로 가중치를 업데이트하는 BGD와 달리, 미니 배치를 나누어 각 훈련 샘플마다 가중치를 업데이트 하기 때문에 가중치가 더 자주 업데이트되어 수렴 속도가 훨씬 빠릅니다. 또, BGD에 비해 local minima에 빠질 가능성이 낮습니다.

**tip**

훈련 샘플 순서를 무작위하게 주입하고 순환되지 않도록 에포크마다 각 배치를 섞어야 합니다.

참고:참고:[https://heeya-stupidbutstudying.tistory.com/entry/ML-신경망에서의-Optimizer-역할과-종류](https://heeya-stupidbutstudying.tistory.com/entry/ML-%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%97%90%EC%84%9C%EC%9D%98-Optimizer-%EC%97%AD%ED%95%A0%EA%B3%BC-%EC%A2%85%EB%A5%98)