from sisypuss.net import Net_Q
from sisypuss.memory import ReplayBuffer
import sisypuss.utils as sssU

import random

import torch
import torch.nn.functional as F

# CUDA Device
DV = sssU.device_set()

## Agent DDQN
class DDQN:
    def __init__(self, env, play_type='save', play_trace=True,
                 gamma=0.99, alpha=0.0005, epsilon_decay=0.9999, epsilon_range=[0.05,1.0],
                 buffer_size=20000, batch_size=64,
                 episode_limit=20000, step_truncate=1000, step_target_update=500, step_learn_start=5000):
        # Initialize the information of training
        self.info_training()
        
        # 모델명
        self.name = 'DDQN'
        
        # 환경 입력
        self.env = env
        
        # 플레이 타입
        self.play_type = play_type
        self.play_trace = play_trace
        
        # state와 action의 dimension
        self.S_dim = env.state_dim
        self.A_dim = env.action_dim
        
        # policy 초기화
        self.net_Q_outer = Net_Q(alpha, self.S_dim, self.A_dim).to(DV) # evaluation
        self.net_Q_inner = Net_Q(alpha, self.S_dim, self.A_dim).to(DV) # selection (target net)
        self.target_update()
        
        # Hyperparameter
        self.gamma = gamma # Discount Factor
        self.alpha = alpha # Learning Rate
        
        # Exploration factor
        self.epsilon_min = epsilon_range[0]
        self.epsilon_max = epsilon_range[1]
        if (self.epsilon_min<0) or (self.epsilon_max>1):
            raise ValueError('Epsilon range must be between 0 and 1.')
        elif (self.epsilon_min>self.epsilon_max):
            raise ValueError('Epsilon range must be input in the order [min,max].')
        self.epsilon = self.epsilon_max
        self.epsilon_decay = epsilon_decay
        
        # Replay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.step_target_update = step_target_update # target network를 업데이트하는 주기
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 최대 에피소드 & 한 에피소드 당 최대 스텝 수 설정
        self.step_truncate = step_truncate
        self.episode_limit = episode_limit
        
        # 학습 시작 Global 스텝 수 설정
        self.step_learn_start = step_learn_start
    
    # 임시
    def info_training(self):
        self.episode_global = 0 # Total episode through training
        self.step_global = 0 # Total step through training
        self.step_episode = [] # Total step each episode
        self.loss_episode = [] # Mean loss each episode
        self.G_episode = [] # Return(no decay) each episode
    
    
    # # 학습된 모델 저장
    # def model_save(self):
    #     path = f'./Model/{self.name}'
    #     check_dir(path)
    #     torch.save(self.net_Q_outer.state_dict(), f'{path}/net_Q(param).pt')
    #     torch.save(self.net_Q_outer, f'{path}/net_Q(all).pt')
        
    # state가 주어졌을 때 action을 선택하는 방식
    def epsilon_greedy(self, S):
        # 탐색률 업데이트
        self.epsilon = max(self.epsilon, self.epsilon_min)
        if random.random() <= self.epsilon: # 탐색
            return random.randint(0, self.A_dim-1)
        else:
            S = torch.from_numpy(S).float()
            Q = self.net_Q_outer.forward(S)
            A = Q.argmax().item()
            return A
    
    def target_update(self):
        """
        Behavior Network Parameter를 복사하여 Target Network Parameter 교체
        """
        self.net_Q_inner.load_state_dict(self.net_Q_outer.state_dict())
    
    
    def learn(self):
        """
        Q 함수를 예측하는 Q net을 학습
        \n S = state(t)
        A = action(t)
        S_next = state(t+1)
        R = reward(t+1)
        D = done mask (True: 0, False: 1)
        """
        # Replay Memory 샘플링
        S, A, S_next, R, D = self.replay_buffer.make_batch(self.batch_size)
        
        # Q value
        Q = torch.gather(self.net_Q_outer(S), dim=1, index=A)
        max_A_next = torch.max(self.net_Q_inner(S_next), dim=1)[1].unsqueeze(1)
        Q_next = torch.gather(self.net_Q_outer(S_next), dim=1, index=max_A_next)
        
        # Target Q
        Q_target = R + D*self.gamma*Q_next
        
        # Gradient Descent
        loss = F.smooth_l1_loss(Q, Q_target)
        self.net_Q_outer.optimizer.zero_grad()
        loss.backward()
        self.net_Q_outer.optimizer.step()
        
        return loss.item()
    
    
    def train(self, mode_trace=False):
        """
        모델 학습
        \n mode_trace가 True면 실시간으로 에피소드에 대한 정보 출력
        S = state(t)
        A = action(t)
        S_next = state(t+1)
        R = reward(t+1)
        G = return
        """
        # Episode Number
        episode_num = 0
        
        # 사전에 지정한 episode 상한까지 학습
        while episode_num < self.episode_limit:
            S = self.env.reset()
            G = 0
            step = 0
            loss = 0
            
            # 충분한 step이 지나면 학습 시작
            if self.step_global > self.step_learn_start:
                switch_learning = True
            else:
                switch_learning = False
            
            # episode가 끝나거나, step limit에 도달하기 전까지 loop
            while step < self.step_truncate:
                # Experience
                A = self.epsilon_greedy(S)
                S_next, R, done, info = self.env.step(A)
                
                # Replay Buffer 생성
                transition = (S, A, S_next, R, done)
                self.replay_buffer.append(transition)
                
                # return(한 에피소드 내 모든 보상, 따라서 gamma decay 안함)과 state update
                G += R
                S = S_next
                
                # 실시간으로 진행 중인 global step 출력
                if (self.play_trace) and ((self.step_global % 1000)==0):
                    print(f'Total Step:{self.step_global}')
                
                # Target Network Update
                if (self.step_global % self.step_target_update)==0:
                    self.target_update()
                
                # Network 학습
                if switch_learning:
                    loss_step = self.learn()
                    loss += loss_step
                
                # step update
                step += 1
                self.step_global += 1
                
                # 에피소드 종료
                if done:
                    break
            
            # 평균 loss 계산
            loss /= step
            
            # Return, step, mean loss 기록
            self.G_episode.append(G)
            self.step_episode.append(step)
            self.loss_episode.append(loss)
            
            # 에피소드 정보 출력
            if self.play_trace:
                print(f'Episode:{episode_num+1}, Score:{G:.3f}, Step:{step}')
            
            # episode & epsilon update
            episode_num += 1
            self.epsilon *= self.epsilon_decay
        
        # 학습 종료
        self.env.close()
