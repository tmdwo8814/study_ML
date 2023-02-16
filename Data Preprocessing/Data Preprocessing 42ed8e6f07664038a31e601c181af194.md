# Data Preprocessing

1. handling missing value
2. handling categorical data
3. split train test
4. Data scaling
5. what is L1 L2?
6. sequential feature selection
7. Use feature importance in RandomForest

# 1. handling missing value

데이터의 품질과 양은 머신러닝 모델을 학습하는데 매우 중요한 요소입니다. 따라서 모델을 학습하기 위해 필요한 양질의 데이터 얻기 위해서, 결측치를 다루는 일은 필수적입니다. 이제부터 어떻게 결측치를 식별하고 처리하는지 알아보겠습니다.

**결측치 식별**

```python
import pandas as pd
from io import StringIO

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

# 결과
	A	  B	  C	 D
0	1.0	2.0	3.0	4.0
1	5.0	6.0	NaN	8.0
2	10.0	11.0	12.0	NaN
```

테스트를 위하여 결측치를 포함한 데이터를 만들어 보겠습니다. StringIO 함수는 문자열을 파일처럼 다루기 위해 사용합니다. 결측치는 일반적으로 예약된 문자열 NaN으로 표시됩니다. 

```python
df.isnull().sum(axis = 1)

# 결과
0    0
1    1
2    1
dtype: int64
```

가장 기본적인 결측치 확인 방법입니다. isnull() 메소드는 결측치를 불리언 타입으로 반환해줍니다. 이후 sum메소드로 축을 정하여 한눈에 파악하기 편하게 결측치의 개수를 얻을 수 있습니다.

**결측치 제거**

```python
df.dropna(axis = 1)
```

dropna 메소드로 결측치를 포함한 행과 열을 제거 할 수있습니다. 다음은 pandas 공식 문서에서 제공한 사용법입니다.

**DataFrame.dropna(***, *axis=0*, *how=_NoDefault.no_default*, *thresh=_NoDefault.no_default*, *subset=None*, *inplace=False*)**

**결측치 대체**

```python
from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
```

위 코드는 사이킷런의 Simpleimputer 클래스를 사용하여결측치를 열 기준으로 평균값으로 대체하는 코드입니다. 따라서 1행의 결측치는 7.5로 대체되고, 2행의 결측치는 6으로 대체됩니다

앞에서 소개한 제거, 대체 이외에도 모델기반대치, 보간법 등 여러가지 결측치처리 방법들이 있습니다.

# 2. categorical data

범주형 데이터는 수치로 계산하지 못하는, 혈액형, 이름 등을 말합니다. 주로 문자열로 되어 있어서 머신러닝 모델이 넣지 못하기 때문에, 수치형 데이터로 변환하는 **인코딩 과정**이 필요합니다.

범주형 데이터를 처리할 때 먼저 생각해야 될 것은 **순서의 유무**입니다. 예를들어 티셔츠의 사이즈는 XL > L > M 으로 순서가 있습니다. 하지만 티셔츠의 색상, red, green, blue는 순서가 없습니다. 이처럼 순서의 유무를 따져보고 데이터를 처리해야합니다.

### 순서 특성 매핑

```python
import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']

# 결과
color	size	price	classlabel
0	green	M	10.1	class2
1	red	L	13.5	class1
2	blue	XL	15.3	class2
```

테스트를 위한 데이터를 만들었습니다. 이번 테스트에서 클래스 레이블의 순서는 없다고 가정합니다.

**map()을 활용하여 size특성 매핑 하기**

```python
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)

# 결과
color	size	price	classlabel
0	green	1	10.1	class2
1	red	2	13.5	class1
2	blue	3	15.3	class2
```

매핑할 특성을 딕셔너리 형태로 만들어주고 map메소드로 전달합니다.

**클래스 레이블 인코딩**

클래스 레이블을 정수로 인코딩하는 것은 머신러닝 라이브러리들의 관례이므로 인코딩을 해주어 전달해줍시다. 앞전에 언급하였듯이 우리가 사용할 클래스 레이블에는 순서가 없다는 것을 기억해둡시다.

```python
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)

# 결과
color	size	price	classlabel
0	green	1	10.1	1
1	red	2	13.5	0
2	blue	3	15.3	1
```

**사이킷런의 LabelEncoder 클래스로 인코딩하기**

```python
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

# 결과
array([1, 0, 1])
```

### 순서가 없는 특성에 원-핫 인코딩 적용

역시 앞서 알아본 LabelEncoder 클래스를 이용하여 color특성을 인코딩할 수 있습니다. 

blue = 0

green = 1

red = 2 로 인코딩 될겁니다. 

이렇게 인코딩된 배열을 모델에 주입하게 되면 우리가 범주형 데이터를 다룰 때 가장 흔히 저지르는 실수 중 하나가 됩니다. 색상 특성은 순서가 필요없지만, 순서 특성이 생기게 되면서 관계성이 생기게 됩니다. 따라서 관계성 문제를 해결하기 위한 방법이 원-핫 인코딩입니다. 이 방법은 순서 없는 특성에 새로운 dummy를 만드는 것입니다. 쉽게 말해 해당하는 특성을 제외한 모든 값을 0으로 치환한 변수를 만드는 것 입니다.

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled.png)

ex) “A dog is chasing a boy on the playground”

- 정수인코딩
    
    “A” : 0, “dog”: 1, “is”: 2, “chasing”: 3, “boy”: 4, “on”: 5, “the”: 6, “playground”: 7
    
- 원-핫 인코딩
    
    “A” : [1, 0, 0, 0, 0, 0, 0, 0], “dog”: [0, 1, 0, 0, 0, 0, 0, 0]
    “is”: [0, 0, 1, 0, 0, 0, 0, 0], “chasing”: [0, 0, 0, 1, 0, 0, 0, 0]
    “boy”: [0, 0, 0, 0, 1, 0, 0, 0], “on”: [0, 0, 0, 0, 0, 1, 0, 0]
    “the”: [0, 0, 0, 0, 0, 0, 1, 0], “playground”: [0, 0, 0, 0, 0, 0, 0, 1]
    

정수인코딩은 순서를 내포할 수 있고, 원-핫 인코딩은 순서를 무마한다고 생각할 수 있습니다.

**사이킷런의 OneHotEncoder 클래스로 원-핫 인코딩 구현**

```python
from sklearn.preprocessing import OneHotEncoder

X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

# 결과
array([[0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]])
```

일반적으로 데이터를 원-핫 인코딩으로 dummy를 만드는 가장 편리한 방법은 pandas의 **get_dummies 메소드**를 이용하는 것 입니다.

**pandas.get_dummies(*data*, *prefix=None*, *prefix_sep='_'*, *dummy_na=False*, *columns=None*, *sparse=False*, *drop_first=False*, *dtype=None*)**

```python
pd.get_dummies(df[['price', 'color', 'size']])

# 결과
price	size	color_blue	color_green	color_red
0	10.1	1	0	1	0
1	13.5	2	0	0	1
2	15.3	3	1	0	0
```

get_dummies 의 매개변수를 통해 여러 설정들을 해줄 수 있습니다. 컬럼을 지정해주지 않으면 모든 범주형 데이터에 대하여 dummy를 만듭니다. 코드 위에 판다스 공식 문서에서 제공하는 문서를 보면 drop_first 매개변수가 있습니다. 이 기능은 더미의 맨 앞 열을 삭제시키는 기능입니다, 이 기능을 왜 있을까요? 이어서 알아보겠습니다.

### 다중 공선성(Multicollinearity)

**다중 공선성**이란 회귀 분석에서 사용된 일부 변수가 다른 변수와 상관도가 높아 데이터 분석시 문제가 발생하는 것입니다. 우리는 회귀 분석을 하면서 모든 독립변수들 각각이 종속변수를 정확하게 설명해주길 바랍니다. 만약 독립변수 중 상관도가 높은 변수가 있다면 문제가 발생합니다. 예를 들어 알아보겠습니다. 

생활 습관이 외적 건강미에 미치는 영향을 알아보기 위한 회귀 문제입니다.

건강미를 종속변수 Y로 놓고

이를 설명하는 독립변수 a를 일섭취 음주량, 독립변수 b를 혈중 알코올 농도 라고 가정하겠습니다.

a와 b가 굉장한 상관관계를 가지고 있습니다. 각각의 증가량이 비례하기 때문이죠. 이것이 왜 문제가 될까 생각이 들었습니다. 예를들면 한명의 학생에게 두명의 선생님이 동시에 가르친다면 전달력에 약할 것처럼 상관관계가 높은 a와 b또한 Y에 전달되는 영향이 희미해질 것입니다. 

따라서 get_dummies 의 drop_first를 이용하여 임의로 특성 하나를 삭제하여 혹시 모를 다중 공선성 문제를 예방한다고 생각합니다.

# 3. split train test

모델을 학습하고 테스트하는 과정에서 우리는 데이터가 필요합니다. 따라서 현재 보유하고 있는 데이터를 학습용 데이터와 테스트용 데이터로 나누어야 하는데, 사이킷런에서 이미 이 기능을 제공하고 있습니다. 코드를 통해 알아보겠습니다.

UCI 머신러닝 저장소에서 Wine 데이터셋 받기

```python
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

df_wine.head()
```

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%201.png)

위 형태의 데이터를 불러왔습니다.

train_test_split 메소드를 이용하여 데이터를 분리하겠습니다.

```python
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
x_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)
```

X에는 클래스 레이블을 제외한 데이터를 담고, y에는 클래스 레이블을 담았습니다.

train_test_split 메소드를 이용하여 데이터를 나누었습니다. test_size 매개변수를 통해 전체 데이터에서 30%를 테스트 데이터에 할당함을 설정합니다. stratify = y로 지정하면서, 나눠진 테스트 데이터와 훈련 데이터의 클래스 레이블의 비율을 맞추어 줍니다.

실전에서 주로 사용하는 테스트셋와 훈련셋의 비율은 6:4 ~ 8:2 정도입니다. 주로 데이터셋의 크기에 따라 비율이 정해진다고 보면 됩니다. 대용량 데이터셋의 경우 9:1 또는 99:1 까지도 갈 수 있습니다.

# 4. Data scaling

머신러닝 모델을 학습할 때 데이터 전처리의 중요성은 앞서 많이 언급하였듯이, 모델의 성능을 좌지우지할 만큼 중요한 과정입니다. 이 과정 중 한가지 방법, 스케일링을 소개합니다. 

데이터에 3가지 특성이 있다고 가정해봅니다. 

x1 은 0부터 1까지의 값을 가집니다

x2 는 0부터 10000000까지의 값을 가지고

x3 는 10000 부터 100000000의 값을 가집니다.

각 변수끼리 상관 정도가 적고 모두 중요한 특성일 때, x2와 x3에 비하여 x1의 영향도가 상대적으로 적을 것입니다. 따라서 변수들의 스케일을 맞추는 과정이 필요할 것입니다. 스케일링의 방법을 알아보겠습니다.

**정규화**

특성 스케일을 [0, 1] 범위에 맞추는 것입니다.

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%202.png)

 구현

```python
# 수식을 이용한 구현
a = np.array([0, 1, 2, 3, 4])
b = (a - a.min()) / (a.max() - a.min())

# 사이킷런의 MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
```

**표준화**

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%203.png)

특성들의 평균을 0, 분산을 1로 만들어 정규분포와 같은 특징을 만드는 것입니다.

표준화는 이상치 정보가 유지되기 때문에 최대-최소 정규화에 비해 이상치에 덜 민감합니다.

구현

```python
# 수식을 이용한 구현
a = np.array([1, 2, 3, 4, 5])
b = (a - ex.mean()) / a.std()

# sklearn의 StandardScaler
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
```

# 5. what is L1 L2?

overfitting을 막기 위한 방법에는 여러가지 방법이 있었는데요, 그 중 한가지인 **가중치 규제**에 대하여 알아보겠습니다.

L1, L2의 regularization을 알아보기 위하여 그 개념을 하나씩 적립해보는 방식으로 이어가겠습니다.

### L1 norm, L2 norm

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%204.png)

L1 norm은 두 벡터의 원소간 차를 절대값을 취해 합한 것입니다.

L2 norm은 두 벡터간 원소간 차를 제곱하여 루트를 씌운것으로 일반적인 두 점 사이의 거리를 구할 때 사용하는 것입니다.

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%205.png)

위 그림으로 예시를 들면, L2 norm으로 구한 길을 초록색 선, 하나만 존재하죠.

나머지 길들은 L1 norm으로 구한 길로, 여러가지 방법이 존재하지만 각각의 거리는 모두 같습니다.

### L1 loss, L2 loss

loss도 위에서 알아본 기본 개념과 똑같이 작용합니다

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%206.png)

L1 loss : 실제값에서 예측값 사이의 오차의 절대값을 구하고, 그 값들의 합을 구합니다. L2 loss에 비해 이상치의 영향을 덜 받아 robust한 특징을 갖습니다.

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%207.png)

L2 loss : 오차 제곱합과 똑같은 형태를 하고 있습니다. 또한 L2 norm을 구하는 것과 달리 마지막에 루트를 취해주지 않습니다. 오차를 제곱하기 때문에 L1 loss에 비하여 outlier의 영향을 많이 받습니다. 따라서 outlier를 신경쓰기 위해선 L2 loss가 더 적합하다고 볼 수 있습니다.

### L1 regularization, L2 regularization

마지막으로 규제에 대하여 알아볼것입니다. 이미 알고있는 규제의 종류에는 drop out, early stopping등이 있습니다. 모델을 학습하기 위해 cost를 감소시키는 방향으로만 가게 되면, 특정 가중치가 너무 커질 수가 있습니다. 특정 가중치만 커지게 되면, 특정 특성에 대하여 의존하는 상황이 발생되면서 일반화  성능이 감소됩니다. 오버피팅이 발생된다고 할 수 있죠. 따라서 L1, L2 regularization으로 가중치 규제를 해주는 것입니다.

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%208.png)

L2 규제의 기하학적 특성에 대하여 알아보고 가겠습니다.

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%209.png)

우리의 목표는 비용 함수를 최소화 하는 가중치의 조합을 찾는 것입니다. 타원의 중심 포인트가 될 것입니다. 규제 파라미터인 람다를 조절하여 규제의 강도를 조절할 수 있습니다. 람다를 크게 하면 회색 구가 작아지면서 0에 수렴할 것입니다. 동시에 가중치는 0에 가까워지고 오버피팅을 초래합니다.

L1 규제를 알아보겠습니다.

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%2010.png)

L1 방식의 규제 범위는 절대값의 합이므로 마름모 형태의 범위가 생성됩니다. 그림을 보시면 w1이 0일 때 비용함수의 등고선이 마름모와 만나고 있습니다. L1 규제의 등고선은 날카롭기 때문에 비용 함수와 만나는 지점은 축에 가깝게 위치할 가능성이 높습니다. 따라서 희소성을 나타낸다고 볼 수 있습니다.

**L1 규제가 희소성을 나타냄**의 의미는 크기가 작은 가중치들을 0으로 수렴시켜 지워버리는 것입니다. 따라서 중요한 가중치가 남게 되어 특성 수가 줄게 되고 모델은 더욱 sparse 해집니다. **특성 선택의 도구 역할**을 한다고 볼 수있습니다.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=1)
lr.fit(X_train_std, y_train)
print('훈련 정확도:', lr.score(X_train_std, y_train))
print('테스트 정확도:', lr.score(X_test_std, y_test))

# 결과
훈련 정확도: 1.0
테스트 정확도: 1.0
```

사이킷런에서 L1 규제를 지원하는 모델을 penalty = ‘l1’으로 지정하여 희소한 모델을 만들 수 있습니다.

C 값은 람다의 역수로써 C 값을 높이면 가중치 규제가 약해져 희소한 모델을 만들기 힘들고, 낮추면 희소한 모델을 만듭니다.

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%2011.png)

# 6. sequential feature selection

모델 복잡도를 줄이고 과적합을 피하기 위하여, 앞에서 알아본 L1, L2 규제와 또 다른 방법으로 특성 선택을 통한 차원 축소가 있습니다. 차원 축소는 규제가 없는 모델에서 유용합니다!

차원 축소에는 **특성 선택**과 **특성 추출**, 두가지 방법이 있습니다.

특성 선택을 알아보겠습니다.

**순차 특성 선택 알고리즘**은 초기 차원의 특성 공간보다 더 작은 차원의 특성 공간으로 축소하는 것입니다.

또한, 주어진 문제에 대한 가장 관련이 높은 특성을 자동으로 선택하는 것이 목적입니다.

순차 특성 선택 알고리즘 중 가장 일반적인 **순차 후진 선택, Sequential Backward Selection, SBS** 알고리즘이 있습니다. SBS 알고리즘으로 과대적합 문제를 해결할 수 있습니다.

SBS 알고리즘의 idea!

1. 목표 특성 개수를 설정
2. 특성을 제거하기 전과 후의 모델 성능 차이를 알아보기 위한 기준 함수의 값을 최대화 하는 특성 선택
3. 기준 함수의 값을 최대화 한 특성 제거
4. 전체 특성 개수가 목표 특성 개수가 되면 종료, 혹은 2 단계로 돌아감

**SBS 알고리즘 구현하기**

```python
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
    # k_features : 목표 특성 개수
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    
    # dim : 현재 특성 개수
    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]
    
    # 정확도 점수 반환 함수
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
```

# 7. Use feature importance in RandomForest

지금까지 유용한 특성을 선택하는 방법으로 규제, SBS 알고리즘을 알아보았습니다.

3번째 방법으로, 앙상블 기법인 랜덤 포레스트를 사용합니다.

랜덤 포레스트를 사용하면 앙상블에 참여한 모든 결정 트리에서 계산한 평균적인 불순도 감소로 특성 중요도를 측정할 수 있습니다.

사이킷런에서 제공하는 RandomForestClassifier 로 모델을 세우고, feature_importances_ 속성으로 특성 중요도 속성을 알 수 있습니다.

와인 데이터셋으로 특성 중요도 판별 해보겠습니다.

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from Data import *

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()

plt.show()
```

![Untitled](Data%20Preprocessing%2042ed8e6f07664038a31e601c181af194/Untitled%2012.png)

모델을 생성하면서 n_estimators 매개변수로 트리의 개수를 500개로 지정합니다.

모델을 훈련을 한 후, feature_importances_ 속성을 통하여 indices 매개변수에 특성 중요도를 역순으로 넘파이 배열의 형태로 넣어줍니다.

그래프를 보면, Proline, Flavanoids, Color intensity 순으로 판별력이 좋은 특성을 알 수 있습니다.