# classification_model_with_sklearn

1. Perceptron with sklearn
2. Logistic Regression
3. SVM
4. KNN

# Perceptron with sklearn

**data load**

```python
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('class label : ', np.unique(y))
# output : class label :  [0 1 2]
```

사이킷런에서 제공하는 붓꽃 데이터셋을 가져왔습니다. y의 클래스 레이블을 보면 정수로 인코딩되어 있는 것을 알 수 있는데, 클래스 레이블을 정수로 인코딩하는 것은 대부분의 머신러닝 라이브러리들의 공통적인 관례입니다.

**data split for train, test**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
```

150개의 데이터를 split하는 과정입니다. test_size = 0.3 으로 설정하게 되면 30%(45)의 데이터는 X_test에 들어가고 70%(105)의 데이터는 X_train에 들어갑니다. stratify=y를 통해 나누어진 데이터의 클래스 레이블의 비율을 동일하게 맞춰줍니다.

**standardization**

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# train
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
```

사이킷런 내장 모듈로 데이터 표준화를 진행하고, 표준화된 데이터를 기반으로 퍼셉트론 모델을 훈련합니다.

**visualization**

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled.png)

그래프와 같이 선형 결정 경계로서 최선의 분류를 했다고 생각해볼 수 있습니다.

# Logistic Regression

아달린과 달리 활성화 함수를 시그모이드 함수를 사용합니다.

**오즈비(odds ratio)**

특정 이벤트가 발생할 확률로 P / (1-P) 로 나타낼 수 있습니다. 여기서 P는 양성 샘플로 예측하려는 대상을 의미합니다. 

**logistic sigmoid function**

오즈비에 로그 함수를 취한 후 역수를 취한 것으로 활성화 함수의 한 종류입니다.

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%201.png)

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%202.png)

**장점**

그래프처럼 0~1 사이의 연속적인 실수를 반환합니다. 또한 매끄러운 특성으로 gradient exploding이 발생하지 않습니다.

**단점**

- 중앙값이 0.5이므로, 출력의 가중치의 합이 계속해서 커집니다.
- gradient vanishing

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%203.png)

로지스틱 회귀 알고리즘의 궁극적인 목표는 클래스 레이블을 예측하기 위하는 것이지만, 위의 과정에서 임계함수를 통과하기 전 시그모이드 함수의 출력 값(확률)만 보는 것이 유용한 사례도 있습니다. 예를 들어 비가 올지말지와 같은 이진 예측뿐만 아니라 비가 올 확률을 예측하는 경우에 로지스틱 회귀를 사용할 수 있습니다. 또한 환자가 질병을 가질 확률을 예측하는데에도 사용할  수 있어서 로지스틱 회귀가 의학 분야에 널리 사용됩니다.

**Model**

```python
class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
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
            
            # 오차 제곱합 대신 로지스틱 비용을 계산합니다.
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
```

**Regularization**

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%204.png)

**overfitting**

과대적합은 훈련 데이터를 외우다싶이 학습을 하여 테스트 데이터에서는 잘 동작하지 않는 현상을 말합니다. 따라서 모델 파라미터가 너무 많아 복잡한 모델이 형성됩니다. 높은 분산을 가집니다.

해결방법

1. **훈련 데이터 양을 늘립니다**. 데이터가 적을 수록 데이터의 특정 패턴이나 노이즈를 암기하는 식으로 모델이 형성됩니다. 따라서 데이터를 늘림으로써 데이터의 일반적인 패턴을 학습하게 됩니다.
2. **모델의 복잡도를 줄입니다.** 
3. **가중치 규제.** L1 규제나 L2 규제를 적용합니다.
4. **Dropout. 랜덤으로 신경망의 뉴런을 끕니다.**

# SVM

Support Vector machine. svm의 최적화 대상은 마진을 최대화하는 것입니다. 마진(Margin)이란 결정경계와 서포트 벡터 사이의 거리를 말합니다.

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%205.png)

실선은 결정경계이고 검은 테두리의 점 3개가 서포트 벡터입니다. 실선과 서포트벡터를 기준으로 그은 점선의 거리(마진)가 최대가 될때, 최적의 결정 경계가 됩니다. 

위 그래프에서 2개의 특성으로 데이터를 분류했는데, n개의 특성에서 n+1의 서포트 벡터가 존재한다는 것을 알 수 있습니다.

**SVM의 장점**

대부분의 머신러닝 지도 학습 알고리즘에서는 학습 데이터를 모두 사용합니다. 하지만 svm에서 우리는 데이터의 결정경계를 세우고 분류를 하기 위하여 서포트 벡터만 골라내면 되었습니다. 따라서 서포트 벡터만 잘 골라낸다면 나머지 데이터는 무시해도 되기 때문에 학습 속도가 매우 빠를 것이라고 예측해 볼 수 있습니다.

**hard margin, soft margin**

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%206.png)

이상치를 허용하지 않고 서포트벡터를 설정한 경우에 첫번째 그림과 같이 마진이 작아지고 기준이 까다로운 결정경계가 세워지는데 이것을 하드마진이라고 합니다. 오버피팅 문제가 있을 수 있습니다.

반대로 이상치 허용범위를 널널하게 잡아주고 서포트벡터를 잡으니 마진은 커지지만 언더피팅 문제가 있을 수 있습니다.

**code**

```python
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
```

SVC를 불러올 때 kernel은 결정경계의 형태를 지정해 줄 수 있습니다. C의 조정으로 이상치의 허용범위를 조절하여 하드마진과 소프트마진의 원인이 될 수 있습니다.

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%207.png)

fit()으로 학습을 진행하면서 결정경계를 세우게 됩니다. 붓꽃 데이터셋의 꽃 분류 문제에 svm을 사용해 그래프로 그려본 것입니다. 2개의 결정경계가 잘 세워진 것을 볼 수 있습니다.

## 커널 SVM을 사용하여 비선형 문제 풀기

**Idea**

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%208.png)

SVM에서 커널을 사용하면 비선형 문제를 풀 수 있습니다. 첫번째 그림만 보면 절대 선형 초평면으로 구분할 수 없을 것 같습니다. 이렇게 선형적으로 구분할 수 없는 데이터를 비선형적으로 구분하는 커널의 기본 아이디어는 다음과 같습니다. 매핑 함수를 사용하여 원본 특성의 비선형 조합을 선형적으로 구분되는 **고차원 공간에 투영**하는 것입니다. 방법을 알아 보겠습니다.

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%209.png)

위 수식대로 2차원 데이터를 3차원 데이터로 변환하여 줍니다. 예를 들어 (1, 2)데이터를 (1, 2, 5) 3차원 데이터로 변환해줍니다. 2번째 그림처럼 3차원 공간에 투영된 데이터에 대하여 초평면을 세웁니다. 이후 원본 특성 공간으로 되돌리면 3번째 그림과 같이 비선형 결정경계가 됩니다.

**문제점**

이러한 방식으로 비선형 문제를 풀 수 있습니다. 하지만 데이터를 고차원 공간에 투영하는 이러한 매핑 방식의 문제는 계산 비용이 매우 비싸다는 것입니다. 점곱 계산 방식을 사용하는데, 이 방식의 높은 비용을 절감하기 위하여 커널 함수를 사용합니다. 커널 함수 중 가장 많이 사용되는 커널이 **방사 기저 함수(RBF)**입니다.

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%2010.png)

**code**

```python
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
```

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%2011.png)

kernel을 rbf로 지정해주고 gamma를 설정해줍니다. gamma는 변수 C와 비슷한 역할로 얼마나 유연하게 경계를 그을 것인가를 설정하는 변수입니다. gamma값을 크게하면 서포트 벡터의 영향이나 범위가 줄게됩니다. 따라서 gamma값을 크게하면 overfitting, 작게하면 underfitting을 초대할 수 있습니다. 

# KNN(K-Nearest Neighbor)

매우 간단한 알고리즘입니다. 작동원리는 간단합니다.

1. k값과 거리 측정 기준을 선택합니다.
2. 샘플에서 k개의 최근접 이웃을 찾습니다.
3. 다수결 투표를 통해 클래스 레이블을 할당합니다.

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%2012.png)

**모수 모델(parametric model)**

모델 파라미터, 가중치나 편향을 조정하기 위하여 데이터를 사용하고 나면 더 이상 데이터가 필요없어지게 되어 저장할 필요가 없습니다. 따라서 메모리 낭비가 발생하지 않습니다. 모수 모델에는 퍼셉트론, 로지스틱 회귀, 선형 SVM이 있습니다

**비모수 모델(nonparametric model)**

비모수 모델은 가설 함수를 구하려는 모델이 아닙니다. 단지 데이터를 학습에 사용하지 않고 매번 새로운 데이터를 분류할 때 마다 저장해두었던 데이터를 사용합니다. K-NN알고리즘이 이에 해당합니다.

**Code**

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)
```

![Untitled](classification_model_with_sklearn%20667ae3d876304a70b787057a358dc24c/Untitled%2013.png)

변수 p로 거리 측정 기준을 정합니다. 1 - 맨해튼 거리, 2 - 유클리디안 거리. n_neighbors로 K 값을 설정합니다.