# RNN

[[바람돌이의 빅데이터] : 네이버 블로그](https://blog.naver.com/winddori2002/221974391796)

## Recurrent Neural Network(순환 신경망)

시퀀스데이터(순서를 가진 데이터) 처리에 강점을 가진 신경망

- 자연어처리에 사용/cnn은 컴퓨터 비전,이미지
- 단어의 순서를 고려할 수 있어, 순서 변화에 따른 의미 변화를 찾아낼 수 있음

![Untitled](RNN%/Untitled.png)

![Untitled](RNN%/Untitled%201.png)

### hidden state

앞선 신경망들은 은닉층에서 활성화 함수를 지난값은 출력층으로만 향함. 

But, RNN은 은닉층의 메모리 셀에서 나온 값을 다시 자신의 입력으로 사용하는 재귀적, 순환적 특성을 가지고있음

—>출력값을 다시 입력값으로 보낼 때, 임시 저장장치가 필요함

—>이것이 hidden state

[RNN - code](https://www.notion.so/RNN-code-df7b374bf6ff4bd78cb62898b4dd8bb7)
