"""
AlphaZero-lite 黑白棋 AI Bot  v3  (12×12 版)
─────────────────────────────────────────
主要更新：
  • BOARD_SIZE = 12，開局位置對齊 12×12 中心
  • 時間預算 MCTS（固定 time_budget 秒，非固定 sim 數）
  • 動態 PS 表：角落被佔後，X格/C格危險度自動解除
  • 斯通納陷阱偵測：對手在 X格 時，同側 edge 步優先度加成
  • Endgame exact solver 閾值調整為 18 空格
  • 全部先前 bug fix 保留
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random, math, os, argparse, pickle, time, gc

from othello.OthelloUtil import getValidMoves, executeMove, isEndGame

# ═══════════════════════════════════════════
#  常數
# ═══════════════════════════════════════════
BOARD_SIZE        = 12
BLACK             =  1
WHITE             = -1
MODEL_PATH        = "othello_model.pt"
REPLAY_PATH       = "othello_replay.pkl"
ENDGAME_THRESHOLD = 18   # 空格 ≤ 此值啟用 exact solver

# 每手時間預算（秒），依棋局進度動態調整
TIME_BUDGET_EARLY  = 1.0   # 前 20 手
TIME_BUDGET_MID    = 2.5   # 中盤
TIME_BUDGET_LATE   = 4.0   # 收官前（exact solver 接管前）

# ═══════════════════════════════════════════
#  12×12 基礎 PS 表
#  設計原則：
#    角落=120（永久穩定），X格=-80（送角風險最高），
#    C格=-40（同側角落易送），邊位=15（穩定後有價值），
#    邊位近X格=5（謹慎），內圈遞減
# ═══════════════════════════════════════════
BASE_PS = np.array([
# A     B     C    D    E    F    G    H    I    J     K     L
[120,  -40,   10,  10,  10,  10,  10,  10,  10,  10,  -40,  120],  # 1
[-40,  -80,   -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -80,  -40],  # 2
[ 10,   -2,    5,   3,   3,   3,   3,   3,   3,   5,   -2,   10],  # 3
[ 10,   -2,    3,   1,   1,   1,   1,   1,   1,   3,   -2,   10],  # 4
[ 10,   -2,    3,   1,   1,   1,   1,   1,   1,   3,   -2,   10],  # 5
[ 10,   -2,    3,   1,   1,   1,   1,   1,   1,   3,   -2,   10],  # 6
[ 10,   -2,    3,   1,   1,   1,   1,   1,   1,   3,   -2,   10],  # 7
[ 10,   -2,    3,   1,   1,   1,   1,   1,   1,   3,   -2,   10],  # 8
[ 10,   -2,    3,   1,   1,   1,   1,   1,   1,   3,   -2,   10],  # 9
[ 10,   -2,    5,   3,   3,   3,   3,   3,   3,   5,   -2,   10],  # 10
[-40,  -80,   -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -80,  -40],  # 11
[120,  -40,   10,  10,  10,  10,  10,  10,  10,  10,  -40,  120],  # 12
], dtype=np.float32)

# 角落相關資料結構：全部依盤面大小 n 動態生成，不寫死座標
def get_corners(n):
    return [(0,0), (0,n-1), (n-1,0), (n-1,n-1)]

def get_corner_x(n):
    """每個角落對應的 X格（斜鄰）"""
    return {
        (0,0):     (1,1),
        (0,n-1):   (1,n-2),
        (n-1,0):   (n-2,1),
        (n-1,n-1): (n-2,n-2),
    }

def get_corner_c(n):
    """每個角落對應的 C格（邊位緊鄰）"""
    return {
        (0,0):     [(0,1),(1,0)],
        (0,n-1):   [(0,n-2),(1,n-1)],
        (n-1,0):   [(n-2,0),(n-1,1)],
        (n-1,n-1): [(n-2,n-1),(n-1,n-2)],
    }

def get_corner_edge(n):
    """每個角落同側的兩條 edge（用於斯通納偵測）"""
    return {
        (0,0):     ([(0,c)   for c in range(1,n-1)], [(r,0)   for r in range(1,n-1)]),
        (0,n-1):   ([(0,c)   for c in range(1,n-1)], [(r,n-1) for r in range(1,n-1)]),
        (n-1,0):   ([(n-1,c) for c in range(1,n-1)], [(r,0)   for r in range(1,n-1)]),
        (n-1,n-1): ([(n-1,c) for c in range(1,n-1)], [(r,n-1) for r in range(1,n-1)]),
    }

# 預先算好 12×12 版本供訓練/比賽主路徑用（避免每次重算）
CORNERS     = get_corners(BOARD_SIZE)
CORNER_X    = get_corner_x(BOARD_SIZE)
CORNER_C    = get_corner_c(BOARD_SIZE)
CORNER_EDGE = get_corner_edge(BOARD_SIZE)


def dynamic_ps(board):
    """
    根據目前盤面動態調整 PS 表（自動依盤面大小 n 計算）：
    1. 角落已被任一方佔據 → 同側 X格/C格 不再危險（設為 +5）
    2. 對手佔據 X格（斯通納陷阱前置）→ 同側 edge 步加分 +20
    """
    n = len(board)
    # 依實際盤面大小取對應的 PS（crop 或 pad）
    if n == BOARD_SIZE:
        ps = BASE_PS.copy()
    elif n < BOARD_SIZE:
        ps = BASE_PS[:n, :n].copy()
    else:
        ps = np.ones((n, n), dtype=np.float32)
        ps[:BOARD_SIZE, :BOARD_SIZE] = BASE_PS

    corners     = get_corners(n)
    corner_x    = get_corner_x(n)
    corner_c    = get_corner_c(n)
    corner_edge = get_corner_edge(n)

    for corner in corners:
        cy, cx = corner
        xy, xx = corner_x[corner]

        # 角落被佔 → X格和C格解除危險
        if board[cy][cx] != 0:
            ps[xy][xx] = max(ps[xy][xx], 5)
            for ry, rx in corner_c[corner]:
                ps[ry][rx] = max(ps[ry][rx], 5)
            continue

        # 對手佔了 X格（斯通納陷阱前置條件）
        opp_on_x = (board[xy][xx] != 0 and board[cy][cx] == 0)
        if opp_on_x:
            for edge_row in corner_edge[corner]:
                for ey, ex in edge_row:
                    if board[ey][ex] == 0:
                        ps[ey][ex] += 20

    return ps


PS_NORM_BASE = BASE_PS / 120.0   # 正規化當 channel 4 輸入


# ═══════════════════════════════════════════
#  OthelloUtil 包裝
# ═══════════════════════════════════════════
def valid_moves(board, color):
    result = getValidMoves(board, color)
    if result.ndim == 1 and result.size == 0:
        return []
    return [tuple(m) for m in result]

def apply_move(board, color, y, x):
    b = board.copy()
    executeMove(b, color, (y, x))
    return b

def game_over(board):
    result = isEndGame(board)
    if result is None:
        return False, None
    return True, result


# ═══════════════════════════════════════════
#  盤面 → 4-channel Tensor
# ═══════════════════════════════════════════
def _get_ps_norm(n):
    """實際盤面大小 n×n 對應的 PS_NORM（自動 crop 或 pad）"""
    if n == BOARD_SIZE:
        return PS_NORM_BASE
    if n < BOARD_SIZE:
        return PS_NORM_BASE[:n, :n]          # crop（如 6×6 比賽）
    # n > BOARD_SIZE：用 1.0 padding
    ps = np.ones((n, n), dtype=np.float32)
    ps[:BOARD_SIZE, :BOARD_SIZE] = PS_NORM_BASE
    return ps

def board_to_tensor(board, color):
    n = len(board)                           # 實際盤面大小（不寫死 12）
    b = np.array(board, dtype=int)           # 確保是純 numpy，不是 OthelloGame 子類
    my_ch   = (b == color).astype(np.float32)
    opp_ch  = (b == -color).astype(np.float32)
    mask_ch = np.zeros((n, n), dtype=np.float32)
    for y, x in valid_moves(b, color):
        mask_ch[y][x] = 1.0
    ps_ch = _get_ps_norm(n)
    return np.stack([my_ch, opp_ch, mask_ch, ps_ch], axis=0)  # (4, n, n)


# ═══════════════════════════════════════════
#  對稱增強（8 種）
# ═══════════════════════════════════════════
def _transform_board(board, k, flip):
    b = np.rot90(board, k)
    return np.fliplr(b) if flip else b

def _transform_policy(policy, k, flip):
    p = policy.reshape(BOARD_SIZE, BOARD_SIZE)
    p = np.rot90(p, k)
    if flip: p = np.fliplr(p)
    return p.flatten()

def augment(board, color, policy, value):
    samples = []
    for k in range(4):
        for flip in [False, True]:
            b2 = _transform_board(board, k, flip)
            p2 = _transform_policy(policy, k, flip)
            samples.append((board_to_tensor(b2, color), p2, value))
    return samples


# ═══════════════════════════════════════════
#  Endgame Exact Solver
# ═══════════════════════════════════════════
def _exact_search(board, color, alpha, beta, my_color):
    over, winner = game_over(board)
    if over:
        return int(np.sum(board == my_color)) - int(np.sum(board == -my_color))
    moves = valid_moves(board, color)
    if not moves:
        return _exact_search(board, -color, alpha, beta, my_color)
    if color == my_color:
        best = -999
        for y, x in moves:
            score = _exact_search(apply_move(board,color,y,x), -color, alpha, beta, my_color)
            best = max(best, score); alpha = max(alpha, score)
            if alpha >= beta: break
        return best
    else:
        best = 999
        for y, x in moves:
            score = _exact_search(apply_move(board,color,y,x), -color, alpha, beta, my_color)
            best = min(best, score); beta = min(beta, score)
            if beta <= alpha: break
        return best

def endgame_best_move(board, color):
    moves = valid_moves(board, color)
    # 優先搶角落（依實際盤面大小動態生成角落座標）
    corners_free = [(y,x) for y,x in get_corners(len(board))
                    if board[y][x]==0 and (y,x) in moves]
    if corners_free:
        return corners_free[0]
    best_move, best_score = None, -999
    for y, x in moves:
        score = _exact_search(apply_move(board,color,y,x), -color, -999, 999, color)
        if score > best_score:
            best_score = score; best_move = (y, x)
    return best_move


# ═══════════════════════════════════════════
#  神經網路（12×12 版）
# ═══════════════════════════════════════════
NET_CHANNELS = 128   # 可在 argparse 覆蓋；6GB GPU 建議 64 或 128

class OthelloNet(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        ch = channels or NET_CHANNELS
        self.ch = ch
        self.conv1 = nn.Conv2d(4,  ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv4 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1, self.bn2 = nn.BatchNorm2d(ch), nn.BatchNorm2d(ch)
        self.bn3, self.bn4 = nn.BatchNorm2d(ch), nn.BatchNorm2d(ch)

        flat = BOARD_SIZE * BOARD_SIZE
        self.p_conv = nn.Conv2d(ch, 4, 1); self.p_bn = nn.BatchNorm2d(4)
        self.p_fc   = nn.Linear(4 * flat, flat)

        self.v_conv = nn.Conv2d(ch, 2, 1); self.v_bn = nn.BatchNorm2d(2)
        self.v_fc1  = nn.Linear(2 * flat, ch)
        self.v_fc2  = nn.Linear(ch, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = self.p_fc(p.view(p.size(0), -1))
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v))))
        return p, v


# ═══════════════════════════════════════════
#  MCTS
# ═══════════════════════════════════════════
class MCTSNode:
    __slots__ = ('board','color','parent','prior','children','visit_count','value_sum')
    def __init__(self, board, color, parent=None, prior=0.0):
        self.board=board; self.color=color; self.parent=parent; self.prior=prior
        self.children={}; self.visit_count=0; self.value_sum=0.0

    @property
    def q(self):
        return self.value_sum/self.visit_count if self.visit_count else 0.0

    def ucb(self, c=1.5):
        return self.q + c*self.prior*math.sqrt(self.parent.visit_count)/(1+self.visit_count)


class MCTS:
    def __init__(self, net, device, c_puct=1.5):
        self.net    = net
        self.device = device
        self.c_puct = c_puct
        self._cache = {}

    def clear_cache(self):
        self._cache.clear()

    @torch.no_grad()
    def _net_eval(self, board, color):
        key = (board.tobytes(), color)
        if key in self._cache:
            return self._cache[key]
        x = torch.tensor(board_to_tensor(board, color),
                         dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, v = self.net(x)
        # normalize policy 到合法步
        policy_raw = torch.softmax(logits, dim=1).cpu().numpy()[0]
        moves  = valid_moves(board, color)
        policy = np.zeros(BOARD_SIZE*BOARD_SIZE, dtype=np.float32)
        if moves:
            for y,x in moves:
                policy[y*BOARD_SIZE+x] = policy_raw[y*BOARD_SIZE+x]
            total = policy.sum()
            policy = policy/total if total>0 else np.full(BOARD_SIZE*BOARD_SIZE, 1/len(moves))
        result = (policy, v.item())
        self._cache[key] = result
        return result

    def _select(self, root):
        node = root
        while node.children:
            node = max(node.children.values(), key=lambda n: n.ucb(self.c_puct))
        return node

    def _expand(self, node):
        moves = valid_moves(node.board, node.color)
        if not moves:
            # pass turn：換色，不落子
            node.children[-1] = MCTSNode(node.board, -node.color, parent=node, prior=1.0)
            return

        policy, value = self._net_eval(node.board, node.color)

        # 動態 PS（含斯通納偵測）計算 prior boost
        dps = dynamic_ps(node.board)
        ps_boost = np.zeros(BOARD_SIZE*BOARD_SIZE, dtype=np.float32)
        for y,x in moves:
            ps_boost[y*BOARD_SIZE+x] = max(dps[y][x], 0)
        if ps_boost.sum() > 0:
            ps_boost /= ps_boost.sum()

        # 角落優先：如果有空角落可落，prior 強制拉到 0.9
        corner_moves = [(y,x) for y,x in moves if (y,x) in CORNERS]

        for y,x in moves:
            a     = y*BOARD_SIZE+x
            if corner_moves:
                # 有角落可下時，角落 prior=0.9，其他均分剩餘 0.1
                prior = 0.9/len(corner_moves) if (y,x) in corner_moves else 0.1/max(1,len(moves)-len(corner_moves))
            else:
                prior = 0.7*policy[a] + 0.3*ps_boost[a]
            node.children[a] = MCTSNode(apply_move(node.board,node.color,y,x), -node.color, node, prior)

        # convention：node.q = parent 視角（傳 -value）
        self._backprop(node, -value)

    def _backprop(self, node, value):
        while node:
            node.visit_count+=1; node.value_sum+=value
            value=-value; node=node.parent

    def get_probs(self, board, color, temperature=1.0, time_budget=2.0):
        """時間預算 MCTS：在 time_budget 秒內盡量多跑模擬"""
        root = MCTSNode(board, color)

        # 根節點 Dirichlet 噪音（訓練時）
        if temperature > 0:
            moves = valid_moves(board, color)
            if moves:
                policy, _ = self._net_eval(board, color)
                dps = dynamic_ps(board)
                ps_boost = np.zeros(BOARD_SIZE*BOARD_SIZE, dtype=np.float32)
                for y,x in moves: ps_boost[y*BOARD_SIZE+x] = max(dps[y][x],0)
                if ps_boost.sum()>0: ps_boost/=ps_boost.sum()
                dir_n = np.random.dirichlet([0.3]*len(moves))
                for i,(y,x) in enumerate(moves):
                    a     = y*BOARD_SIZE+x
                    prior = 0.55*policy[a]+0.25*ps_boost[a]+0.20*dir_n[i]
                    root.children[a] = MCTSNode(apply_move(board,color,y,x),-color,root,prior)

        deadline = time.time() + time_budget
        sim_count = 0
        while time.time() < deadline:
            for _ in range(10):   # 每次批跑 10 sim，減少 time.time() 呼叫開銷
                leaf = self._select(root)
                over, winner = game_over(leaf.board)
                if over:
                    result = 0.0 if winner==0 else (1.0 if winner==leaf.color else -1.0)
                    self._backprop(leaf, -result)   # convention: -result for parent
                else:
                    self._expand(leaf)
                sim_count += 1

        counts = np.zeros(BOARD_SIZE*BOARD_SIZE)
        for a, child in root.children.items():
            if a >= 0:
                counts[a] = child.visit_count

        if temperature == 0:
            probs = np.zeros(BOARD_SIZE*BOARD_SIZE); probs[np.argmax(counts)]=1.0
        else:
            counts **= (1.0/temperature)
            probs = counts/counts.sum() if counts.sum()>0 else counts
        return probs, sim_count


# ═══════════════════════════════════════════
#  自我對弈（含對稱增強）
# ═══════════════════════════════════════════
def _init_board():
    """12×12 開局：黑 (5,6)(6,5)，白 (5,5)(6,6)"""
    b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    n = BOARD_SIZE
    b[n//2-1][n//2-1] = WHITE; b[n//2][n//2]   = WHITE
    b[n//2-1][n//2]   = BLACK; b[n//2][n//2-1] = BLACK
    return b

def self_play(mcts, temp_threshold=15, time_budget=1.0):
    board   = _init_board()
    color   = BLACK
    history = []

    for step in range(BOARD_SIZE*BOARD_SIZE*2):
        over, _ = game_over(board)
        if over: break
        moves = valid_moves(board, color)
        if not moves: color=-color; continue

        temp  = 1.0 if step < temp_threshold else 0.0
        probs, _ = mcts.get_probs(board, color, temp, time_budget=time_budget)
        history.append((board.copy(), color, probs))

        action = np.random.choice(len(probs), p=probs) if temp>0 else np.argmax(probs)
        y, x   = divmod(action, BOARD_SIZE)
        board  = apply_move(board, color, y, x)
        color  = -color

    over, winner = game_over(board)
    data = []
    for board_arr, turn, probs in history:
        result = 0.0 if winner==0 else (1.0 if winner==turn else -1.0)
        data.extend(augment(board_arr, turn, probs, result))
    return data, winner


# ═══════════════════════════════════════════
#  訓練
# ═══════════════════════════════════════════
def train_batch(net, optimizer, replay, batch_size, device):
    if len(replay)<batch_size: return None, None
    batch=random.sample(replay, batch_size)
    s,tp,tv=zip(*batch)
    s  = torch.tensor(np.array(s),  dtype=torch.float32).to(device)
    tp = torch.tensor(np.array(tp), dtype=torch.float32).to(device)
    tv = torch.tensor(np.array(tv), dtype=torch.float32).unsqueeze(1).to(device)
    net.train()
    logits, values = net(s)
    p_loss = -(tp*F.log_softmax(logits,dim=1)).sum(dim=1).mean()
    v_loss = F.mse_loss(values, tv)
    optimizer.zero_grad()
    (p_loss+v_loss).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step(); net.eval()
    return p_loss.item(), v_loss.item()


# ═══════════════════════════════════════════
#  Checkpoint
# ═══════════════════════════════════════════
def save_checkpoint(net, optimizer, scheduler, replay, iteration):
    torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'iter':iteration}, MODEL_PATH)
    with open(REPLAY_PATH,'wb') as f: pickle.dump(list(replay), f)
    print(f"  ✓ iter={iteration}，replay={len(replay)}")

def load_checkpoint(net, optimizer, scheduler, replay):
    start=1
    if os.path.exists(MODEL_PATH):
        ckpt=torch.load(MODEL_PATH, map_location='cpu')
        net.load_state_dict(ckpt['model'] if isinstance(ckpt,dict) else ckpt)
        if isinstance(ckpt,dict):
            if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
            start=ckpt.get('iter',0)+1
        print(f"載入模型 iter={start-1}")
    if os.path.exists(REPLAY_PATH):
        with open(REPLAY_PATH,'rb') as f: saved=pickle.load(f)
        replay.extend(saved); print(f"載入 replay {len(replay)} 筆")
    return start


# ═══════════════════════════════════════════
#  訓練主流程
# ═══════════════════════════════════════════
def _gpu_info(device):
    if device.type != 'cuda': return ""
    used  = torch.cuda.memory_allocated(device) / 1e6
    rsvd  = torch.cuda.memory_reserved(device)  / 1e6
    total = torch.cuda.get_device_properties(device).total_memory / 1e6
    return f"  GPU {used:.0f}/{total:.0f}MB(rsv {rsvd:.0f})"

def run_training(iterations=100, games_per_iter=10, batch_size=128,
                 selfplay_budget=1.0, channels=128, replay_maxlen=100_000):
    global NET_CHANNELS
    NET_CHANNELS = channels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"裝置: {device}  ch={channels}  batch={batch_size}  replay_maxlen={replay_maxlen}")
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.85)  # 最多用 85% VRAM
        torch.backends.cudnn.benchmark = True

    net       = OthelloNet(channels=channels).to(device); net.eval()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    replay    = deque(maxlen=replay_maxlen)
    start     = load_checkpoint(net, optimizer, scheduler, replay)
    mcts      = MCTS(net, device)

    for it in range(start, start+iterations):
        print(f"\n── Iteration {it} ──{_gpu_info(device)}")

        # ── 自我對弈 ──
        mcts.clear_cache()
        bw=ww=dw=0
        for g in range(games_per_iter):
            data, winner = self_play(mcts, time_budget=selfplay_budget)
            replay.extend(data)
            if winner==BLACK: bw+=1
            elif winner==WHITE: ww+=1
            else: dw+=1
            print(f"  自我對弈 {g+1}/{games_per_iter} 黑{bw}白{ww}平{dw}", end='\r')

        # 自我對弈後清 CUDA 碎片
        if device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        print(f"\n  buffer={len(replay)} 筆{_gpu_info(device)}")

        # ── 訓練 ──
        pl_s=vl_s=n=0
        for _ in range(100):
            pl,vl=train_batch(net,optimizer,replay,batch_size,device)
            if pl: pl_s+=pl; vl_s+=vl; n+=1
        if device.type == 'cuda': torch.cuda.empty_cache()
        if n: print(f"  P={pl_s/n:.4f} V={vl_s/n:.4f} lr={optimizer.param_groups[0]['lr']:.6f}{_gpu_info(device)}")
        scheduler.step()
        if it%10==0 or it==start+iterations-1:
            save_checkpoint(net, optimizer, scheduler, replay, it)


# ═══════════════════════════════════════════
#  比賽用 BOT
# ═══════════════════════════════════════════
class BOT:
    """
    getAction(board, color) → (row, col)
    自動依剩餘空格數決定策略：
      • 空格多（> ENDGAME_THRESHOLD）：時間預算 MCTS
      • 空格少（≤ ENDGAME_THRESHOLD）：Endgame exact solver
    """
    def __init__(self, model_path=MODEL_PATH):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net=OthelloNet().to(self.device); self.net.eval()
        if os.path.exists(model_path):
            ckpt=torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(ckpt['model'] if isinstance(ckpt,dict) else ckpt)
            print(f"[NeuralBot] 模型載入: {model_path}")
        else:
            print("[NeuralBot] 找不到模型，使用隨機初始化（記得先訓練！）")
        self.mcts=MCTS(self.net, self.device)
        self._move_count=0

    def _time_budget(self, empty):
        """依空格數動態配時（總預算 3 分鐘，保守配）"""
        if empty > 80:   return TIME_BUDGET_EARLY
        if empty > 25:   return TIME_BUDGET_MID
        return TIME_BUDGET_LATE

    def _ps_best(self, moves, board):
        """純 PS 分數選步（最後防線，絕對不會 crash）"""
        n = len(board)
        dps = dynamic_ps(board)
        return max(moves, key=lambda m: dps[m[0]][m[1]])

    def _to_move(self, m):
        """確保回傳純 Python int tuple，不回傳 numpy.int64（平台可能不認識）"""
        if m is None: return None
        return (int(m[0]), int(m[1]))

    def getAction(self, board, color):
        """
        主要入口。所有分支都包在 try-except 內，
        任何 exception 都會退回「PS 分數最高步」確保有回傳值。
        回傳值一律用 _to_move() 轉成純 Python int，避免平台不認識 numpy.int64。
        """
        try:
            b = np.array(board, dtype=int)
            n = len(b)

            moves = valid_moves(b, color)
            if not moves:
                return None   # 合法 pass

            # ── 1. 立刻搶角落 ──
            corners = get_corners(n)
            for cy, cx in corners:
                if (cy, cx) in moves:
                    print(f"[NeuralBot] 搶角落 ({cy},{cx})")
                    return self._to_move((cy, cx))

            empty = int(np.sum(b == 0))

            # ── 2. Endgame exact solver（空格少時精確搜尋）──
            if empty <= ENDGAME_THRESHOLD:
                try:
                    mv = endgame_best_move(b, color)
                    if mv:
                        print(f"[NeuralBot] Endgame solver → {mv}（空格={empty}）")
                        return self._to_move(mv)
                except Exception as e:
                    print(f"[NeuralBot] Endgame solver 失敗: {e}，改用 PS")

            # ── 3. 盤面大小不符網路 → PS fallback（不呼叫網路）──
            if n != BOARD_SIZE:
                mv = self._ps_best(moves, b)
                print(f"[NeuralBot] 盤面 {n}×{n}（PS fallback）→ {mv}")
                return self._to_move(mv)

            # ── 4. MCTS（主要策略，盤面=BOARD_SIZE）──
            budget = self._time_budget(empty)
            try:
                probs, sims = self.mcts.get_probs(b, color, temperature=0,
                                                   time_budget=budget)
                print(f"[NeuralBot] MCTS {sims}sims/{budget:.1f}s 空格={empty}")
                best_move, best_p = None, -1.0
                for y, x in moves:
                    p = probs[y * BOARD_SIZE + x]
                    if p > best_p:
                        best_p = p; best_move = (y, x)
                if best_move:
                    self._move_count += 1
                    return self._to_move(best_move)
            except Exception as e:
                print(f"[NeuralBot] MCTS 失敗: {e}，改用 PS")

            # ── 5. 最後防線：PS 分數最高步 ──
            mv = self._ps_best(moves, b)
            print(f"[NeuralBot] PS fallback → {mv}")
            return self._to_move(mv)

        except Exception as e:
            # 保證一定回傳，不讓 bot 卡死
            print(f"[NeuralBot] getAction 嚴重錯誤: {e}")
            import traceback; traceback.print_exc()
            try:
                b2 = np.array(board, dtype=int)
                moves2 = valid_moves(b2, color)
                if moves2:
                    return self._to_move(moves2[0])
            except:
                pass
            return None


# ═══════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════
if __name__=='__main__':
    parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--train',      action='store_true')
    parser.add_argument('--iterations', type=int,   default=100,
                        help='訓練迭代次數')
    parser.add_argument('--games',      type=int,   default=10,
                        help='每次迭代自我對弈局數')
    parser.add_argument('--batch',      type=int,   default=128,
                        help='訓練 batch size（6GB GPU 建議 64~128）')
    parser.add_argument('--budget',     type=float, default=0.8,
                        help='自我對弈每手時間預算（秒，訓練用；越小越快）')
    parser.add_argument('--channels',   type=int,   default=128,
                        help='網路 channels（64=輕量，128=標準，256=大型）')
    parser.add_argument('--replay',     type=int,   default=100_000,
                        help='Replay buffer 上限（每筆≈3KB，100K≈300MB RAM）')
    args=parser.parse_args()
    if args.train:
        run_training(args.iterations, args.games, args.batch,
                     args.budget, args.channels, args.replay)
    else:
        print("加上 --train 開始訓練，建議指令：")
        print()
        print("# 6GB GPU 標準（推薦）")
        print("  python NeuralBot.py --train --iterations 200 --games 10 --batch 128 --channels 128 --replay 100000")
        print()
        print("# 6GB GPU 保守（記憶體嚴格）")
        print("  python NeuralBot.py --train --iterations 200 --games 10 --batch 64  --channels 64  --replay 50000")
        print()
        print("# 繼續之前訓練（自動讀取 checkpoint）")
        print("  python NeuralBot.py --train --iterations 100")