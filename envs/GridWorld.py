#%% Info
# version : python 3.9
import sisypuss.utils as sssU

import os

import gym
from gym import spaces
import pygame as pg

import numpy as np
import random
import itertools

import torch
import torchvision.transforms as T


#%% Main

# 파일 경로 설정
dir_main = os.path.split(os.path.abspath(__file__))[0]

# 게임 상수 설정
G_FPS = 30
C_white = (255,255,255)
C_black = (0,0,0)
C_border = (96,96,96)
C_inside = (192,192,192)
C_lightgreen = (170,240,180)


#%%
class GCEnv(gym.Env):
    def __init__(self, render_mode=None, stage_type=0, grid_num=(9,9), auto_collect=False,
                 reward_set=(3.,1.,1.,-0.01,-0.02), state_type='onehot'):
        """
        render_mode
        - human : full play screen
        - agent : No screen
        
        stage_type
        - 0 : fixed position 5 items.
        - 1 : random position 5 items.
        
        reward_set
        [0] : rule
        [1] : full
        [2] : item
        [3] : step
        [4] : vain
        
        state_type
        - onehot
        - semiconv
        - semiconv_ego
        - conv
        """
        
        ## 화면 출력 여부 설정
        self.render_mode = render_mode
        self.screen = None
        self.render_once = False # 한번이라도 렌더 되었는지 여부
        
        ## 게임 기본 요소 설정
        self.G_clock = None # Time
        self.grid_num = grid_num # 맵의 크기 (나중에 고칠 것)
        self.stage_type = stage_type # 스테이지 타입
        
        ## 스테이지 정보
        # Optimal
        self.optimal_order = [] # Episode마다 가장 짧은 pathway를 기록
        self.optimal_step = [] # Episode마다 이론상 가능한 가장 짧은 step을 기록
        # Biased
        self.biased_order = [] # Episode마다 human이 택할 것 같은 pathway를 기록
        self.biased_step = [] # Episode마다 human이 택할 것 같은 경로의 총 step 예상치를 기록
        self.bias_amount = []
        # Actual
        self.actual_order = [] # Episode 끝날때마다 실제 pathway를 기록
        self.actual_step = [] # Episode 끝날때마다 실제 총 step을 기록
        self.trial_type = []
        # Temp
        self.current_step = 0 # 해당 Episode에서 행한 step 수 실시간 기록
        self.current_order = [-1,-1,-1] # 해당 Episode에서 pathway 실시간 기록
        self.current_episode = 0 # 진행중인 Episode number (주의: 1부터 시작)
        
        # 주요 요소 선언
        self.G_player = self.Player(self, (self.grid_num[0]//2,self.grid_num[1]//2)) # Player
        self.G_item = self.ItemCreator() # Item
        if (stage_type == 4):
            self.G_case = np.array([0,0,0,0,0,1,1,1,1,1]) # 10 episode 마다 sub,con case 샘플링
        
        # 렌더 활성화
        self.render_initialize()
        
        ## Action 유형 선언
        self.auto_collect = auto_collect
        if auto_collect:
            self.action_space = spaces.Discrete(4) # (0,1,2,3) + 1 :move
        else:
            self.action_space = spaces.Discrete(5) # 0:pick, (1,2,3,4):move
        self.action = -1
        
        ## State 크기 정보 (obs와 함께 직접 수정해야 함)
        self.state_type = state_type
        if (state_type == 'onehot'):
            self.state_space = (2*grid_num[0]*grid_num[1]+6)*2
        elif (state_type == 'ego'):
            self.player_sight = 7
            self.state_space = 6+(2*self.player_sight+1)**2
        elif (state_type == 'allo'): # item이 3개인 경우만 가능
            self.item_sight = 7
            self.state_space = 6+((2*self.item_sight+1)**2)*3
        elif (state_type == 'conv'):
            self.state_space = 40
            self.resizer = T.Compose(
                [T.Resize((self.state_space,self.state_space)), T.Normalize(0,255)])
        else:
            raise SystemExit('Please check the state_type')
        
        ## 이전 State
        self.M_map_player = self.FS_map_player()
        self.M_map_item = self.FS_map_item()
        self.M_inv_info = self.FS_inv_info()
        self.M_pos_player = self.G_player.position
        
        ## Reward 설정
        self.R_rule = reward_set[0] # composition
        self.R_full = reward_set[1] # just make 3 item
        self.R_item = reward_set[2] # take 1 item
        self.R_step = reward_set[3] # take 1 step
        self.R_vain = reward_set[4] # wrong forage
    
    
    ## step
    def step(self, action):
        # Update step info
        self.current_step += 1
        self.info = {'Eval': 'None'}
        
        # 액션 이전, 플레이어 정보
        inv_before = self.G_player.inventory_num
        
        # 플레이어 액션
        if self.auto_collect:
            self.action = action + 1
        else:
            self.action = action
        self.G_player.action(self.action)
        
        # 액션 이후, 플레이어 정보
        inv_after = self.G_player.inventory_num
        
        # State 구성
        self.observation = self.FS_observe(output_type='flatten')
        
        # reward and done
        if (inv_after == inv_before): # 아이템 미수집 시
            if (action==0): 
                self.reward = self.R_vain # 허공에서 수집 액션한 경우
                self.info['Eval'] = 'vain'
            else:
                R = self.R_step
                self.reward = R * self.FR_distance()
                self.info['Eval'] = 'step'
            self.done = False
        elif (inv_after > inv_before): # 아이템 수집 시
            if (inv_after == 3):
                R, info = self.FR_inv_check() # 아이템 모두 모은 경우 인벤토리 검사
                self.reward = R
                self.done = True
                self.info['Eval'] = info
            else: 
                self.reward = self.R_item # 아직 아이템을 모두 모으지 못한 경우
                self.done = False
                self.info['Eval'] = 'item'
        else: # 버그 색출
            self.close()
            raise SystemExit('Cannot find action command')
        
        if self.render_mode == 'human':
            for event in pg.event.get():
                if event.type == pg.QUIT: #game의 event type이 QUIT 명령이라면
                    pg.quit()
        
        return self.observation, self.reward, self.done, self.info
    
    
    ## 한 에피소드가 끝났을 때
    def reset(self):
        # Reset Step info
        if (self.current_step != 0): # 첫 리셋은 기록 생략
            self.actual_step.append(self.current_step)
            self.actual_order.append(self.current_order)
        self.current_step = 0
        self.current_order = [-1,-1,-1]
        
        # Update Episode info
        self.current_episode += 1
        if (self.stage_type == 4) and (self.current_episode%10 == 1):
            np.random.shuffle(self.G_case)
        
        # Reset
        self.G_item = self.ItemCreator(self.stage_type) # 아이템 생성 (나중에 고칠 것)
        if (self.stage_type == 3) or (self.stage_type == 4):
            self.G_player.reset((self.grid_num[0]//2,self.grid_num[1]//2))
        else:
            self.G_player.reset()
        self.render_reset()
        self.done = False
        
        # State
        self.observation = self.FS_observe(output_type='flatten')
        
        # # 스테이지 정보 기록
        # opt_step, opt_order = self.FI_optimal()
        # self.optimal_step.append(opt_step)
        # self.optimal_order.append(opt_order)
        
        return self.observation
    
    
    ## (Function of State) State와 관련된 함수
    # Observation 구성
    def FS_observe(self, output_type='flatten'):
        # Current State
        map_player = self.FS_map_player()
        map_item = self.FS_map_item()
        inv_info = self.FS_inv_info()
        
        # State
        if (self.state_type == 'onehot'):
            O = [[self.M_map_player, self.M_map_item, self.M_inv_info],
                 [map_player, map_item, inv_info]]
        elif (self.state_type == 'ego'):
            ps = self.player_sight
            pp = self.G_player.position
            minimap = pseudo_conv2d(image=map_item, sigma=0.7, padding=ps, scaling=10)
            map_sight = minimap[pp[0]:pp[0]+2*ps+1, pp[1]:pp[1]+2*ps+1]
            O = [map_sight, inv_info]
        elif (self.state_type == 'allo'):
            ps = self.item_sight
            minimap = pseudo_conv2d(image=map_player, sigma=0.7, padding=ps, scaling=10)
            map_sight = []
            for item in self.G_item:
                if item.collected:
                    map_temp = np.zeros(((ps*2+1),(ps*2+1)))
                else:
                    pp = item.position
                    map_temp = minimap[pp[0]:pp[0]+2*ps+1, pp[1]:pp[1]+2*ps+1]
                map_sight.append(map_temp)
            O = [map_sight, inv_info]
        elif (self.state_type == 'conv'):
            image = np.transpose(pg.surfarray.pixels3d(self.screen), axes=(1,0,2)) # H, W, C
            O = self.render_crop(image=image)
            return O
        else: # 버그 예외 처리
            raise SystemExit('Please check the state_type. It seems None')
        
        # Past State Update
        self.M_map_player = map_player.copy()
        self.M_map_item = map_item.copy()
        self.M_inv_info = inv_info.copy()
        self.M_pos_player = self.G_player.position # tuple이기 때문에 copy를 쓰지 않아도 무방.
        
        # Output Type
        if (output_type == 'flatten'):
            O = all_flatten(O)
        else: # 버그 방지
            raise SystemExit('Please check the output_type.')
        
        return O
    
    # 플레이어 맵 반환
    def FS_map_player(self):
        """
        플레이어의 위치를 onehot 형식으로 반환해준다.
        """
        # 그리드 크기만큼의 map array 생성
        map_player = np.zeros(self.grid_num)
        
        # 플레이어가 존재하는 좌표에 1 설정
        map_player[self.G_player.position] = 1
        
        return map_player
    
    # 아이템 맵 반환
    def FS_map_item(self):
        """
        현재 맵에 존재하는 아이템의 정보를 반환해준다.
        1. gridworld size가 9*7이면, 총 63칸의 array를 만든다.
        2. 아이템이 (x,y)에 위치하고 content는 (a,b,c)라면, map[x,y]에 cba(4진법)을 입력한다.
        예1_ 위치(4,0),content(2,0)[삼각형,색없음]: map[4,0]=2
        예2_ 위치(3,2),content(1,3)[원,파랑색]: map[3,2]=13
        """
        # 그리드 크기만큼의 map array 생성
        map_item = np.zeros(self.grid_num)
        
        # 아이템이 존재하는 좌표에 content 설정
        for item in self.G_item:
            if not item.collected:
                map_item[item.position] = 1
                #map_item[item.position] = item.content[0]+4*item.content[1]
            
        return map_item
    
    # 인벤토리 정보 반환
    def FS_inv_info(self):
        """
        content는 1,2,3의 숫자를 사용하여 특성을 구분짓는다.
        다만 이 방식은 딥러닝의 input으로 쓰는데 문제가 있다.
        
        예를 들어, shape content를 생각해보자.
        사람은 content가 1인 item과 content가 3인 item을 원, 사각형과 같이 특성을 부여하는 index로써 이해한다.
        그러나 딥러닝에서는 단순히 <사각형은 원의 3배구나!> 와 같이 index가 아닌 하나의 양으로써 이해한다.
        이는 우리의 추구 방향과는 맞지 않으므로, 인벤토리 내의 아이템 content를 딥러닝의 input으로 사용할 때는
        단순히 on/off(1/0)값으로 분류해줄 필요가 있다.
        
        다만, 현재 인벤토리에 어떤 content가 특히 많은지를 나타낼 수 있도록,
        (RL agent가 현 인벤토리에 존재하는 아이템의 content가 얼마나 중복하게 가지고 있는지 구분하기를 기대하므로)
        CC의 index는 content의 종류, value는 content의 보유수로 설정하고자 한다.
        """
        inv = self.G_player.inventory
        
        inv_info = np.zeros((2,3)) # Dimension, Content
        for i in range(3):
            for d in range(2):
                C_idx = inv[i].content[d]
                if C_idx != 0:
                    inv_info[d, C_idx-1] += 1
        return inv_info
    
    
    ## (Function of Reward) Reward와 관련된 함수
    # 인벤토리 체크
    def FR_inv_check(self):
        """
        인벤토리에 있는 각 아이템의 content를 곱하면, 아래와 같은 숫자를 반환할 것이다.
        
        모두 같은 content: 1^3=1, 2^3=8, 3^3=27
        각각 다른 content: 1*2*3=6
        존재하지 않는 content: 0을 적어도 1개 곱하므로 0
        
        여기서 reward를 줘야하는 상황은 바로, 세 아이템 모두 한 dimension에 대하여 
        각각 다른 content를 가지고 있거나, 모두 같은 content를 가지고 있는 상황이다.
        """
        # 인벤토리 내부에는 3개의 아이템 content가 들어있을 것
        if (self.G_player.inventory_num != 3):
            raise SystemExit('Cannot find three items in inventory!')
        
        inv = self.G_player.inventory
        CM = (inv[0].content[0]*inv[1].content[0]*inv[2].content[0],
             inv[0].content[1]*inv[1].content[1]*inv[2].content[1])
        
        # CM를 통해 reward를 주는 방식은 아래 코드를 수정하여 고칠 수 있다.
        if any((CM[0]==1,CM[0]==8,CM[0]==27,CM[0]==6)):
            R = self.R_rule
            info = 'rule'
        else:
            R = self.R_full
            info = 'full'
        
        return R, info
    
    # 플레이어와 맵에 남아있는 아이템의 유클리드 거리 기하 평균에 기반한 cost
    def FR_distance(self):
        player_pos = self.G_player.position
        num_item = 0
        geo_mean = 1
        for item in self.G_item:
            if not item.collected:
                num_item += 1
                distance = ((player_pos[0]-item.position[0])**2 + (player_pos[1]-item.position[1])**2)**(1/2)
                geo_mean *= distance
        geo_mean = geo_mean ** (1/num_item)
        return geo_mean
    
    
    ## (Function of Info) Info와 관련된 함수
    # 아이템 이론상 가능한 최단 step 계산
    # 주의 : 수집 액션 또한 step에 포함하여 계산
    def FI_optimal(self):
        # 플레이어 위치 저장
        pos_P = self.G_player.position
        
        # 아이템 위치 저장
        pos_I = []
        for item in self.G_item:
            pos_I.append(item.position)
        num_I = len(pos_I) # 총 아이템 개수
        
        # 플레이어-아이템 거리 행렬
        D = np.zeros(num_I)
        for idx_I in range(num_I):
            D[idx_I] = abs(pos_P[0]-pos_I[idx_I][0]) + abs(pos_P[1]-pos_I[idx_I][1])
        
        # 아이템 간 거리 행렬
        L = np.zeros((num_I,num_I))
        for idx_I in itertools.permutations(range(num_I), 2):
            L[idx_I[0], idx_I[1]] = abs(pos_I[idx_I[0]][0]-pos_I[idx_I[1]][0]) + abs(pos_I[idx_I[0]][1]-pos_I[idx_I[1]][1])
        
        # 최단 step 계산
        # 전체 아이템 중 3개를 고르고, 해당 아이템들을 수집할 수 있는 최단 step을 계산하는 방식
        opt_step = 10000
        for idx_tri in itertools.combinations(range(num_I), 3):
            perimeter = L[idx_tri[0], idx_tri[1]] + L[idx_tri[1], idx_tri[2]] + L[idx_tri[0], idx_tri[2]]
            for idx_line in itertools.permutations(idx_tri, 2):
                path_step = D[idx_line[0]] + perimeter - L[idx_line[0], idx_line[1]]
                if path_step < opt_step:
                    opt_step = path_step
                    opt_order = (idx_line[0], (set(idx_tri)^set(idx_line)).pop(), idx_line[1])
        opt_step = opt_step + 3 # 수집 step 추가
        
        return opt_step, opt_order
    
    
    ## 렌더 함수
    # 렌더 초기화
    def render_initialize(self):
        # 렌더 모드 체크
        if (self.render_mode == 'agent'): # Agent 모드
            return
        elif (self.render_mode == 'human'): # human 모드
            pass
        else: # 렌더 모드가 부적절한 경우
            raise SystemExit('You have to specify the render_mode. etc: human')
        
        # 첫 렌더 여부
        self.render_once = True
        
        ## 렌더에 필요한 요소 준비
        # 화면에 창 띄우기
        pg.init()
        pg.display.set_caption('Hello!') # Title
        
        # 해상도 설정
        self.G_resolution = (1280,720)
        self.screen = pg.display.set_mode(self.G_resolution) # Display Setting

        # clock 생성
        self.G_clock = pg.time.Clock()
        
        # 그리드 설정
        self.grid_size = 64
        self.grid_SP = (96,80) # 그리드가 시작되는 위치, 즉 그리드 왼쪽 상단의 위치
        self.grid_EP = (self.grid_SP[0]+self.grid_num[0]*self.grid_size, 
                        self.grid_SP[1]+self.grid_num[1]*self.grid_size)
        
        # Inventory 설정
        self.Inv_SP = (self.G_resolution[0]-self.grid_size*3-128, 128) # 인벤토리 표시가 시작되는 위치(왼쪽 상단)
        self.Inv_height = 32 # 인벤토리 텍스트 표시 폭
        
        # 스프라이트 생성
        self.G_background = pg.transform.scale(load_image("BG.png"), self.G_resolution)
        self.grid_spr = pg.transform.scale(load_image("Grid.png"), (self.grid_size,self.grid_size))

        # 텍스트 설정
        self.G_font = pg.font.Font(None, 24)
        self.text_inventory = self.G_font.render("Inventory", True, C_black) # render(text, antialias, color)
        self.text_inventory_rect = self.text_inventory.get_rect()
        self.text_inventory_rect.center = (self.Inv_SP[0]+self.grid_size*3//2, 
                                           self.Inv_SP[1]+self.Inv_height//2)
        
        # 플레이어 설정
        self.G_player.spr = pg.transform.scale(load_image("Agent.png"),
                                               (self.grid_size,self.grid_size)) # 그리드 1칸 크기만큼 캐릭터 사이즈 조절
        self.G_player.rect = self.G_player.spr.get_rect() # 스프라이트 히트박스
        
        ## 렌더 리셋
        self.render_reset()
        
        # 배경 그리기
        self.screen.blit(self.G_background, self.G_background.get_rect())
        

    # 렌더 리셋
    def render_reset(self):
        # 렌더가 안되었다면 reset 취소
        if not self.render_once:
            return
        
        # 플레이어 위치 리셋
        self.G_player.rect.topleft = (self.grid_SP[0]+self.G_player.position[0]*self.grid_size, 
                                      self.grid_SP[1]+self.G_player.position[1]*self.grid_size)
        # 아이템 리셋
        for item in self.G_item:
            item.spr = pg.Surface((self.grid_size,self.grid_size), pg.SRCALPHA)
            item.rect = item.spr.get_rect()
            item.rect.topleft = (self.grid_SP[0]+item.position[0]*self.grid_size, 
                                 self.grid_SP[1]+item.position[1]*self.grid_size)
            item.draw()
        
    # 렌더
    def render(self):
        # 렌더 모드가 agent인 경우
        if (self.render_mode == 'agent'):
            return
        
        ### Draw in Display
        ## 그리드 그리기
        for row in range(self.grid_num[0]):
            for col in range(self.grid_num[1]):
                self.screen.blit(self.grid_spr,(self.grid_SP[0]+row*self.grid_size,
                                                self.grid_SP[1]+col*self.grid_size, 
                                                self.grid_size, self.grid_size))
        ## 인벤토리 그리기
        # 상단
        pg.draw.rect(self.screen, C_lightgreen, 
                     (self.Inv_SP[0], self.Inv_SP[1], self.grid_size*3, self.Inv_height))
        pg.draw.rect(self.screen, C_black, 
                     (self.Inv_SP[0], self.Inv_SP[1], self.grid_size*3, self.Inv_height), 2)
        self.screen.blit(self.text_inventory, self.text_inventory_rect)
        # 하단 (내용물)
        for idx in range(3):
            pg.draw.rect(self.screen, C_inside, 
                         (self.Inv_SP[0]+idx*self.grid_size, self.Inv_SP[1]+self.Inv_height, 
                          self.grid_size, self.grid_size))
            pg.draw.rect(self.screen, C_border, 
                         (self.Inv_SP[0]+idx*self.grid_size, self.Inv_SP[1]+self.Inv_height, 
                          self.grid_size, self.grid_size), 2)
        ## 아이템 그리기
        for item in self.G_item:
            self.screen.blit(item.spr, item.rect)
        ## 플레이어 그리기
        self.G_player.rect.topleft = (self.grid_SP[0]+self.G_player.position[0]*self.grid_size, 
                                      self.grid_SP[1]+self.G_player.position[1]*self.grid_size)
        self.screen.blit(self.G_player.spr, self.G_player.rect)
        
        # 화면 업데이트
        if (self.render_mode == 'human'):
            self.G_clock.tick(G_FPS)
            pg.display.update()
    
    # 화면 크롭
    def render_crop(self, image):
        ## image(height, width, channel)
        # Grid만 crop
        img_ref = image[self.grid_SP[1]:self.grid_EP[1], self.grid_SP[0]:self.grid_EP[0], :]
        img_ref = np.transpose(np.ascontiguousarray(img_ref), axes=(2,0,1)) # C, H, W
        img_ref = torch.tensor(img_ref, dtype=torch.float).to(DV)
        img_ref = self.resizer(img_ref).unsqueeze(0)
        
        return img_ref
    
    
    ## 게임 종료
    def close(self):
        # Last Recording
        self.actual_step.append(self.current_step)
        self.actual_order.append(self.current_order)
        
        # Exit
        pg.quit()
    
    
    ## Game Component
    # 플레이어 (Agent)
    class Player:
        def __init__(self, env, position):
            self.env = env # 플레이어가 위치한 환경
            self.position = position # position=(4,6)이라면, 5열 7행에 플레이어 배치
            self.inventory = [env.Item(env),
                              env.Item(env),
                              env.Item(env)]
            self.inventory_num = 0
            self.spr = None
            self.rect = None # 스프라이트 히트박스
        
        def action(self, command):
            if (command == 0): # 아이템 먹기
                self.pick()
            elif (command in (1,2,3,4)): # 이동
                self.move(command)
                if self.env.auto_collect:
                    self.pick()
            elif (command == -1): # Null (아무 행동하지 않음)
                pass
            else: # 혹시나 버그로 인해 command가 이상한 값으로 설정되어 있을 때
                self.env.close()
                raise SystemExit('Cannot find action command')
        
        # 아이템 수집
        def pick(self):
            for item_idx, item in enumerate(self.env.G_item):
                # 플레이어와 위치도 겹치고, 아직 수집 되지 않은 아이템인 경우
                if (item.position == self.position) and not item.collected:
                    if self.env.render_once:
                        item.rect.topleft = (self.env.Inv_SP[0]+self.inventory_num*self.env.grid_size, 
                                             self.env.Inv_SP[1]+self.env.Inv_height)
                    item.collected = True
                    self.env.current_order[self.inventory_num] = item_idx # 수집한 아이템 인덱스 저장
                    self.inventory[self.inventory_num] = item
                    self.inventory_num += 1
        
        # 이동
        def move(self, direction):
            if (direction == 1) and (self.position[0] < self.env.grid_num[0]-1): # Right
                self.position = (self.position[0]+1, self.position[1])
            elif (direction == 2) and (self.position[1] > 0): # Up
                self.position = (self.position[0], self.position[1]-1)
            elif (direction == 3) and (self.position[0] > 0): # Left
                self.position = (self.position[0]-1, self.position[1])
            elif (direction == 4) and (self.position[1] < self.env.grid_num[1]-1): # Down
                self.position = (self.position[0], self.position[1]+1)
            
        # 리셋
        def reset(self, position=None):         
            # 위치 설정
            if position is None: # 입력 변수가 없다면 현재 자리 그대로
                pass
            else:
                self.position = position
            
            # 인벤토리 비우기
            self.inventory = [self.env.Item(self.env),
                              self.env.Item(self.env),
                              self.env.Item(self.env)]
            self.inventory_num = 0


    # 아이템
    class Item:
        """
        content: tuple (shape, color, texture)
        shape: 1=circle, 2=triangle, 3=square
        color: (0=None) 1=Red, 2=Green, 3=Blue
        texture: (0=None) 1=?, 2=?, 3=?
        """
        def __init__(self, env, content=(0,0), position=(0,0)):
            self.env = env # 아이템이 위치하는 환경
            self.position = position # pos=(4,6)이라면, 5열 7행에 아이템 배치
            self.content = content
            self.collected = False # 인벤토리에 수집된 여부
            self.spr = None
            self.rect = None
            
            """
            self.spr = pg.Surface((grid_size,grid_size), pg.SRCALPHA)
            self.rect = self.spr.get_rect()
            self.rect.topleft = (grid_SP[0]+pos[0]*grid_size, grid_SP[1]+pos[1]*grid_size)
            """
        
        def draw(self):
            # Dimension: 색 설정
            if self.content[1]==0:
                color = (0,0,0)
            elif self.content[1]==1:
                color = (255,0,0)
            elif self.content[1]==2:
                color = (0,255,0)
            elif self.content[1]==3:
                color = (0,0,255)
            else:
                self.env.close()
                raise SystemExit('Cannot find color content')
            # Dimension: 모양 설정
            grid_size = self.env.grid_size
            if self.content[0]==1:
                pg.draw.circle(self.spr, color, (grid_size//2,grid_size//2), grid_size//2-8)
            elif self.content[0]==2:
                pg.draw.polygon(self.spr, color, ((8,grid_size-8),(grid_size//2,8),(grid_size-8,grid_size-8)))
            elif self.content[0]==3:
                pg.draw.rect(self.spr, color, (8,8,grid_size-16,grid_size-16))
        

    ## 아이템 생성기
    def ItemCreator(self, stage_type=0):
        if (stage_type==0): # 고정, rule
            ItemSet = [self.Item(self,(1,1),(0,0)), 
                       self.Item(self,(2,2),(3,2)), 
                       self.Item(self,(3,3),(4,6)), 
                       self.Item(self,(2,1),(7,1)), 
                       self.Item(self,(2,3),(6,1))]
            
        elif (stage_type==1): # 랜덤, rule
            position_list = random.sample(range(self.grid_num[0]*self.grid_num[1]),5)
            content_list = [(1,1),(2,2),(3,3),(2,1),(2,3)]
            ItemSet = []
            for idx in range(5):
                pos = (position_list[idx]%self.grid_num[0], position_list[idx]//self.grid_num[0])
                ItemSet.append(self.Item(self,content_list[idx],pos))
                
        elif (stage_type==2): # 랜덤, no rule
            position_list = random.sample(range(self.grid_num[0]*self.grid_num[1]),5)
            ItemSet = []
            for idx in range(5):
                pos = (position_list[idx]%self.grid_num[0], position_list[idx]//self.grid_num[0])
                ItemSet.append(self.Item(self,(1,1),pos))
                
        elif (stage_type==3): # Suboptimal Case
            item_pos, opt_info, bias_info, bias_amount = self.Item_Suboptimal()
            ItemSet = [self.Item(self,(1,1),item_pos[0]), 
                       self.Item(self,(1,1),item_pos[1]), 
                       self.Item(self,(1,1),item_pos[2])]
            self.optimal_order.append(opt_info[0])
            self.optimal_step.append(opt_info[1])
            self.biased_order.append(bias_info[0])
            self.biased_step.append(bias_info[1])
            self.bias_amount.append(bias_amount)
            
        elif (stage_type==4): # Mixed Case
            case_type = self.G_case[self.current_episode%10-1]
            print(' ')
            if (case_type == 0): # Suboptimal Case
                item_pos, opt_info, bias_info, bias_amount = self.Item_Suboptimal()
                print('sub',opt_info[1],bias_info[1])
            elif (case_type == 1): # Control Case
                item_pos, opt_info, bias_info, bias_amount = self.Item_Control()
                print('con',opt_info[1],bias_info[1])
            ItemSet = [self.Item(self,(1,1),item_pos[0]), 
                       self.Item(self,(1,1),item_pos[1]), 
                       self.Item(self,(1,1),item_pos[2])]
            self.optimal_order.append(opt_info[0])
            self.optimal_step.append(opt_info[1])
            self.biased_order.append(bias_info[0])
            self.biased_step.append(bias_info[1])
            self.bias_amount.append(bias_amount)
            self.trial_type.append(case_type)
            
        return ItemSet
    
    # Suboptimal Case
    # (Biased pathway > Optimal pathway)
    def Item_Suboptimal(self):
        """
        opt = D_B+L_C+L_B (B-A-C)
        biased(1) = D_A+L_C+L_A (A-B-C) [L_C<L_B]
        biased(2) = D_A+L_B+L_A (A-C-B) [L_B<L_C]
        """
        # 주의 : 적어도 5x5 grid 이상만 사용할 것
        # 주의 : 플레이어는 항상 정중앙 스폰 (subopt 배치 확률을 높이기 위함)
        available_A = False
        G_W = self.grid_num[0]//2 # Grid half width
        G_H = self.grid_num[1]//2 # Grid half height
        
        # (Cond 1. D_A<D_B<D_C)
        # (Cond 2. D_B-D_A < L_A-L_mid) [opt가 biased보다 짧을 조건] [L_A = L_max]
        while not (available_A):
            # First, Set the position of (B,C) according to agent position.
            D_B = random.randint(2, G_W+G_H-1) # Agent, Item B 사이 거리
            D_C = random.randint(D_B+1, G_W+G_H) # D_C가 D_B보다 더 길도록 설정 (Cond 1)
            Area_BC = random.sample(tuple(itertools.permutations([(1,1),(-1,1),(-1,-1),(1,-1)],2)), 1)[0] # B,C가 존재할 구역 선정
            B_x = random.randint(max(D_B-G_H, 0), min(D_B, G_W))
            B_y = D_B - B_x
            C_x = random.randint(max(D_C-G_H, 0), min(D_C, G_W))
            C_y = D_C - C_x
            B_pos = (G_W+B_x*Area_BC[0][0], G_H+B_y*Area_BC[0][1])
            C_pos = (G_W+C_x*Area_BC[1][0], G_H+C_y*Area_BC[1][1])
            
            # Second, Set the position of A according to Agent, B, C position.
            for D_A in range(1, D_B): # (Cond 1)
                for A_x in range(0, min(D_A, G_W)+1):
                    A_y = D_A - A_x
                    for Area_A in [(1,1),(-1,1),(-1,-1),(1,-1)]:
                        A_pos = (G_W+A_x*Area_A[0], G_H+A_y*Area_A[1])
                        L_A = abs(B_pos[0]-C_pos[0]) + abs(B_pos[1]-C_pos[1]) # B, C 사이 거리 (A의 대변)
                        L_B = abs(A_pos[0]-C_pos[0]) + abs(A_pos[1]-C_pos[1]) # A, C 사이 거리 (B의 대변)
                        L_C = abs(A_pos[0]-B_pos[0]) + abs(A_pos[1]-B_pos[1]) # A, B 사이 거리 (C의 대변)
                        if (D_B-D_A < L_A-L_B) and (L_C < L_B): # (Cond 2-1) subopt : A-B-C
                            bias_order = (0,1,2)
                            bias_step = D_A+L_C+L_A
                            available_A = True
                        elif (D_B-D_A < L_A-L_C) and (L_B < L_C): # (Cond 2-2) subopt : A-C-B
                            bias_order = (0,2,1)
                            bias_step = D_A+L_B+L_A
                            available_A = True
                        if (available_A):
                            break
                    if (available_A):
                        break
                if (available_A):
                    break
        bias_amount = D_B-D_A
        item_pos = (A_pos, B_pos, C_pos)
        bias_info = (bias_order, bias_step)
        opt_order = (1,0,2)
        opt_step = D_B+L_C+L_B
        opt_info = (opt_order, opt_step)
        return item_pos, opt_info, bias_info, bias_amount
    
    # Control Case
    # (Biased pathway = Optimal pathway)
    def Item_Control(self):
        """
        * L_A = L_max 가 아니어도 되는 경우가 있으나, 조건 동일화를 위해 Item_Suboptimal의 기본 가정인 L_A = L_max를 따르기로 함.
        biased(1) = D_A+L_C+L_A (A-B-C) [L_C<L_B]
        biased(2) = D_A+L_B+L_A (A-C-B) [L_B<L_C]
        """
        # 주의 : 적어도 5x5 grid 이상만 사용할 것
        # 주의 : 플레이어는 항상 정중앙 스폰 (subopt 배치 확률을 높이기 위함)
        available_A = False
        G_W = self.grid_num[0]//2 # Grid half width
        G_H = self.grid_num[1]//2 # Grid half height
        
        # (Cond 1. D_A<D_B<D_C)
        # (Cond 2. L_A-L_mid < D_B-D_A) [opt가 biased와 같을 조건]
        while not (available_A):
            # First, Set the position of (B,C) according to agent position.
            D_B = random.randint(2, G_W+G_H-1) # Agent, Item B 사이 거리
            D_C = random.randint(D_B+1, G_W+G_H) # D_C가 D_B보다 더 길도록 설정 (Cond 1)
            Area_BC = random.sample(tuple(itertools.permutations([(1,1),(-1,1),(-1,-1),(1,-1)],2)), 1)[0] # B,C가 존재할 구역 선정
            B_x = random.randint(max(D_B-G_H, 0), min(D_B, G_W))
            B_y = D_B - B_x
            C_x = random.randint(max(D_C-G_H, 0), min(D_C, G_W))
            C_y = D_C - C_x
            B_pos = (G_W+B_x*Area_BC[0][0], G_H+B_y*Area_BC[0][1])
            C_pos = (G_W+C_x*Area_BC[1][0], G_H+C_y*Area_BC[1][1])
            
            # Second, Set the position of A according to Agent, B, C position.
            for D_A in range(1, D_B): # (Cond 1)
                for A_x in range(0, min(D_A, G_W)+1):
                    A_y = D_A - A_x
                    for Area_A in [(1,1),(-1,1),(-1,-1),(1,-1)]:
                        A_pos = (G_W+A_x*Area_A[0], G_H+A_y*Area_A[1])
                        L_A = abs(B_pos[0]-C_pos[0]) + abs(B_pos[1]-C_pos[1]) # B, C 사이 거리 (A의 대변)
                        L_B = abs(A_pos[0]-C_pos[0]) + abs(A_pos[1]-C_pos[1]) # A, C 사이 거리 (B의 대변)
                        L_C = abs(A_pos[0]-B_pos[0]) + abs(A_pos[1]-B_pos[1]) # A, B 사이 거리 (C의 대변)
                        if (L_A > L_B) and (L_A > L_C): # L_A = L_max
                            if (D_B-D_A > L_A-L_B) and (L_C < L_B): # (Cond 2-1) A-B-C
                                bias_order = (0,1,2)
                                bias_step = D_A+L_C+L_A
                                available_A = True
                            elif (D_B-D_A > L_A-L_C) and (L_B < L_C): # (Cond 2-2) A-C-B
                                bias_order = (0,2,1)
                                bias_step = D_A+L_B+L_A
                                available_A = True
                        if (available_A):
                            break
                    if (available_A):
                        break
                if (available_A):
                    break
        bias_amount = D_B-D_A
        item_pos = (A_pos, B_pos, C_pos)
        bias_info = (bias_order, bias_step)
        opt_info = bias_info
        return item_pos, opt_info, bias_info, bias_amount