# 강화학습(Reinforcement Learning) 기본 설계 노트 

## 1. 강화학습의 핵심 아이디어: 강아지 훈련하기 

강화학습을 이해하는 가장 쉬운 방법은 '강아지 훈련'을 떠올리는 것입니다.

> 강아지(agent)에게 '앉아!'라는 정답 행동을 직접 가르쳐주지 않습니다. 대신, 강아지가 거실(environment)에서 여러 행동을 시도하다가 우연히 앉았을 때, 즉시 '간식(reward)'을 줍니다. 이 과정을 반복하면 강아지는 점차 **"어떤 상황에서 어떤 행동을 해야 간식을 받을 수 있는지"** 스스로 터득하게 됩니다.

강화학습은 이처럼 'reward'이라는 신호를 통해 AI가 최적의 행동 전략, 즉 **Policy**을 스스로 학습하도록 만드는 방법론입니다.

---

## 2. "컵 잡는 로봇팔"을 위한 5대 요소 설계

우리 프로젝트의 목표를 달성하기 위해서 강화학습의 5가지 핵심 요소를 다음과 같이 구체적으로 설계합니다.

### ① Agent: The Brain

* **정의:** 행동의 주체. cup-grabbing-robot-arm 프로젝트에서는 로봇팔 하드웨어가 아니라, 그 팔을 어떻게 움직일지 결정하는 **심층 신경망(Deep Neural Network) 모델**이 바로 에이전트입니다.
* **역할:** 현재 상태(`State`)를 입력받아, 어떤 행동(`Action`)을 할지 출력합니다.

### ② Environment: The World & The Rules

* **정의:** 에이전트가 상호작용하는 세상. cup-grabbing-robot-arm 프로젝트의 구체적인 실체는 **`PyBullet`이나 `MuJoCo`와 같은 물리 시뮬레이터**입니다.
* **역할:**
    1.  **물리 엔진:** 에이전트의 행동이 어떤 물리적 결과를 낳는지 계산합니다. 
    2.  **심판:** 행동의 결과에 따라 보상 점수를 계산합니다.
    3.  **상황 보고관:** 행동이 끝난 후의 다음 상태 정보를 에이전트에게 전달합니다.

### ③ State: The Observation

* **정의:** 특정 시점에서 에이전트가 관측할 수 있는 모든 정보의 집합. AI의 '눈'에 해당하는 `Perception` 모듈이 이 정보를 가공하여 전달합니다.
* **설계 목표:** AI가 최적의 판단을 내리는 데 필요한 충분하고 명확한 정보를 숫자 벡터 형태로 제공해야 합니다.
* **구체적인 상태 벡터 예시:**
    ```python
    # 로봇팔의 상태 정보를 담을 숫자 배열 (벡터 형태)
    state = [
        # 로봇팔 자체의 정보 (ex: 7축 로봇팔)
        q1, q2, q3, q4, q5, q6, q7,      # 7개 관절의 현재 각도 (rad)
        vq1, vq2, ..., vq7,             # 7개 관절의 현재 각속도 (rad/s)

        # 그리퍼 정보
        gripper_x, gripper_y, gripper_z, # 그리퍼 끝부분의 3D 월드 좌표
        gripper_isOpen,                  # 그리퍼가 열렸는지(1) 닫혔는지(0)

        # 목표물 정보 (Perception 모듈로부터 받음)
        cup_x, cup_y, cup_z,             # 컵의 3D 월드 좌표

        # 관계 정보 (AI가 학습하기 좋은 핵심 정보)
        vec_gripper_to_cup_x,            # 그리퍼에서 컵을 향하는 벡터 X
        vec_gripper_to_cup_y,            # 그리퍼에서 컵을 향하는 벡터 Y
        vec_gripper_to_cup_z             # 그리퍼에서 컵을 향하는 벡터 Z
    ]
    ```

### ④ Action: The Command

* **정의:** 에이전트가 각 상태에서 내릴 수 있는 구체적인 제어 명령.
* **설계 목표:** 로봇팔의 움직임을 충분히 표현할 수 있으면서 AI가 학습하기에 너무 복잡하지 않아야 합니다.
* **구체적인 행동 벡터 예시 (카테시안 제어):**
    ```python
    # 그리퍼의 움직임을 직접 명령하는 숫자 배열 (벡터 형태)
    action = [
        delta_x,      # X축으로 이동할 거리 (-1 ~ +1 사이 값)
        delta_y,      # Y축으로 이동할 거리
        delta_z,      # Z축으로 이동할 거리
        delta_roll,   # Roll 방향으로 회전할 각도
        delta_pitch,  # Pitch 방향으로 회전할 각도
        delta_yaw,    # Yaw 방향으로 회전할 각도
        gripper_force # 그리퍼를 쥘지(1) 놓을지(-1)
    ]
    ```

### ⑤ Reward: The Feedback

* **정의:** 에이전트의 행동이 얼마나 좋았는지를 평가하는 점수. 보상 함수를 어떻게 설계하느냐가 학습의 성패를 좌우합니다.
* **설계 철학:**
    > 좋은 행동은 즉시, 그리고 꾸준히 칭찬하여 길을 알려준다. (Reward Shaping)
* **구체적인 보상 함수 설계 예시 (Pseudo-code):**
    ```python
    def calculate_reward(state, action, next_state):
        # 1. 컵에 가까워지는 것을 칭찬 (가장 중요한 Dense Reward)
        prev_dist = distance(state.gripper_pos, state.cup_pos)
        new_dist = distance(next_state.gripper_pos, next_state.cup_pos)
        reward_reaching = (prev_dist - new_dist) * 100 # 거리가 줄면 플러스 보상

        # 2. 컵을 잡는 성공에 대한 큰 보상 (Sparse Reward)
        reward_grasp = 0
        if next_state.is_grasping_cup:
            reward_grasp = 1000

        # 3. 불필요한 행동과 실패에 대한 페널티
        reward_time = -1 # 매 순간마다 시간 페널티
        reward_fail = 0
        if next_state.cup_is_toppled:
            reward_fail = -500 # 컵을 넘어뜨리면 큰 벌점

        # 모든 점수를 합산하여 최종 보상 결정
        total_reward = reward_reaching + reward_grasp + reward_time + reward_fail
        return total_reward
    ```

---

## 3. 전체 학습 흐름 

1.  **관측(Observe):** 에이전트가 현재 `상태(State)`를 환경으로부터 받는다.
2.  **행동(Action):** 에이전트는 `상태`를 입력받아 `행동(Action)`을 계산하여 출력한다.
3.  **실행 및 피드백(Execute & Feedback):** 환경은 `행동`을 시뮬레이션하고, 그 결과로 발생한 `보상(Reward)`과 `다음 상태(Next State)`를 계산한다.
4.  **학습(Learn):** 에이전트는 `(상태, 행동, 보상, 다음 상태)`라는 하나의 '경험' 데이터를 이용해서 더 높은 보상을 받을 수 있는 방향으로 자신의 신경망을 업데이트한다.
5.  **반복(Repeat):** 1~4번 과정을 수백만 번 반복하며 최적의 컵 잡기 전략을 스스로 터득한다.
