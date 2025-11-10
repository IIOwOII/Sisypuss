import sisypuss.util as sssU

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.distributions as dist

## Network Set
class Net_Q(nn.Module):
    """
    Network_Q의 output은 각 action에 대한 Q값이다.
    """
    def __init__(self, alpha, state_space, action_space):
        # 상위 클래스인 nn.Module의 __init__ 호출
        # (self.parameters, self.forward() 덮어쓰기를 위함)
        super().__init__()
        
        # Device
        self.DV = sssU.device_set()
        
        # 레이어 설정
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_space)
        
        # Adam 기법으로 최적화
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
    
    def forward(self, S):
        # Input: State, Output: Policy
        S = S.to(self.DV)
        S = F.relu(self.fc1(S))
        S = F.relu(self.fc2(S))
        Q = self.fc3(S)
        return Q