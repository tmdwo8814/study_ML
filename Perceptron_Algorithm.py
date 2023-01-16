# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 퍼셉트론 알고리즘
#
# - 이진분류(지도학습) 모델을 학습하기 위한 알고리즘

import numpy as np


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
        rgen = np.ramdom.RandomState(self.random_state)
        
        # 학습된 가중치
        # np.random.normal -> 평균:0, 표준편차:0.01, 사이즈:1 + X.shape[1]인 가중치 생성
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
    
    # return class label: 1 or -1
    def predict(self, X):
        return np.where(self.net_input(x)>=0.0, 1, -1)

rgen = np.random.RandomState(1)
type(rgen)

a = [1, 2, 3] 
b = [4, 5]
sum([i*j for i, j in zip(a, b)])
