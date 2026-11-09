from typing import Dict, List, Tuple, Set, Optional
import heapq
import math


# 按题目表格整理后的 7x7 地图。
# A、B、T1~T4 所在格代价为 1；X 为障碍。
GRID4 = [
    ['A', 1, 1, 'X', 1, 1, 'T1'],
    [1, 'X', 1, 1, 1, 'X', 1],
    [1, 1, 'T2', 1, 1, 1, 1],
    [1, 1, 'X', 1, 'X', 8, 1],
    [1, 1, 1, 1, 'T3', 1, 1],
    [1, 'X', 9, 'X', 1, 'X', 1],
    ['T4', 1, 1, 1, 1, 1, 'B'],
]

KEY_POS: Dict[str, Tuple[int, int]] = {
    'A': (0, 0),
    'B': (6, 6),
    'T1': (0, 6),
    'T2': (2, 2),
    'T3': (4, 4),
    'T4': (6, 0),
}

TREASURE_VALUE: Dict[str, int] = {
    'T1': 9,
    'T2': 6,
    'T3': 7,
    'T4': 7,
}

KEY_NAMES = ['A', 'B', 'T1', 'T2', 'T3', 'T4']
TREASURE_NAMES = ['T1', 'T2', 'T3', 'T4']


def _cell_cost(cell) -> Optional[int]:
    """
    进入某格子的代价。
    X -> None，表示不可通行。
    """
    if cell == 'X':
        return None

    if isinstance(cell, int):
        return cell

    s = str(cell)
    if s.isdigit():
        return int(s)
    if s in ('A', 'B') or s.startswith('T'):
        return 1
    if s == 'E':
        return 0
    return 1


def _is_walkable(cell) -> bool:
    return cell != 'X'


def astar_cost(start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    题目四内部使用：计算从 start 到 goal 的最小移动代价。
    进入目标格的代价也按地形计算；A/B/T 格代价为 1。
    """
    if start == goal:
        return 0

    rows, cols = len(GRID4), len(GRID4[0])

    if not _is_walkable(GRID4[start[0]][start[1]]):
        return 10**18
    if not _is_walkable(GRID4[goal[0]][goal[1]]):
        return 10**18

    def h(r: int, c: int) -> int:
        return abs(r - goal[0]) + abs(c - goal[1])

    open_heap = [(h(start[0], start[1]), 0, start)]
    g_cost = {start: 0}
    closed = set()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        _, cur_g, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue
        if cur == goal:
            return cur_g

        closed.add(cur)
        r, c = cur

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not _is_walkable(GRID4[nr][nc]):
                continue

            step_cost = _cell_cost(GRID4[nr][nc])
            if step_cost is None:
                continue

            new_g = cur_g + step_cost
            if new_g < g_cost.get(nxt, 10**18):
                g_cost[nxt] = new_g
                f = new_g + h(nr, nc)
                heapq.heappush(open_heap, (f, new_g, nxt))

    return 10**18


def compute_distance_matrix() -> List[List[int]]:
    """
    输出 6x6 距离矩阵。
    顺序：A, B, T1, T2, T3, T4。
    """
    n = len(KEY_NAMES)
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            else:
                p = KEY_POS[KEY_NAMES[i]]
                q = KEY_POS[KEY_NAMES[j]]
                matrix[i][j] = astar_cost(p, q)

    return matrix


def _dist_by_pos(p: Tuple[int, int], q: Tuple[int, int]) -> int:
    return astar_cost(p, q)


def _dist_to_treasure(pos: Tuple[int, int], treasure: str) -> int:
    return astar_cost(pos, KEY_POS[treasure])


def minimax_alpha_beta(
    state,
    depth: int,
    alpha: float,
    beta: float,
    use_alpha_beta: bool,
    stats: Optional[Dict[str, int]] = None
) -> float:
    """
    博弈搜索。

    state = (
        pos_a,          # A 当前位置
        pos_b,          # B 当前位置
        remaining,      # 剩余宝藏集合，如 {'T1', 'T3'}
        score_a,        # A 累计收益
        score_b,        # B 累计收益
        player          # 当前玩家，'A' 或 'B'
    )
    """
    if stats is None:
        stats = {}

    stats["visited"] = stats.get("visited", 0) + 1

    pos_a, pos_b, remaining, score_a, score_b, player = state

    if depth <= 0 or not remaining:
        return score_a - score_b

    ordered_treasures = sorted(remaining, key=lambda x: int(x[1:]))

    if player == 'A':
        best = -math.inf
        for t in ordered_treasures:
            d = _dist_to_treasure(pos_a, t)
            gain = TREASURE_VALUE[t] - d

            new_state = (
                KEY_POS[t],
                pos_b,
                remaining - {t},
                score_a + gain,
                score_b,
                'B'
            )

            value = minimax_alpha_beta(
                new_state,
                depth - 1,
                alpha,
                beta,
                use_alpha_beta,
                stats
            )

            best = max(best, value)
            alpha = max(alpha, best)

            if use_alpha_beta and beta <= alpha:
                stats["pruned"] = stats.get("pruned", 0) + 1
                break

        return best

    else:
        best = math.inf
        for t in ordered_treasures:
            d = _dist_to_treasure(pos_b, t)
            gain = TREASURE_VALUE[t] - d

            new_state = (
                pos_a,
                KEY_POS[t],
                remaining - {t},
                score_a,
                score_b + gain,
                'A'
            )

            value = minimax_alpha_beta(
                new_state,
                depth - 1,
                alpha,
                beta,
                use_alpha_beta,
                stats
            )

            best = min(best, value)
            beta = min(beta, best)

            if use_alpha_beta and beta <= alpha:
                stats["pruned"] = stats.get("pruned", 0) + 1
                break

        return best


def best_first_move(
    depth_h: int,
    use_alpha_beta: bool = True
) -> Tuple[str, int, Dict[str, int]]:
    """
    计算 A 的最优首步。
    返回 (best_target, best_value, stats)
    """
    remaining = set(TREASURE_NAMES)
    pos_a = KEY_POS['A']
    pos_b = KEY_POS['B']

    best_target = None
    best_value = -math.inf
    total_stats: Dict[str, int] = {"visited": 0, "pruned": 0}

    for t in sorted(remaining, key=lambda x: int(x[1:])):
        d = _dist_to_treasure(pos_a, t)
        gain = TREASURE_VALUE[t] - d

        new_state = (
            KEY_POS[t],
            pos_b,
            remaining - {t},
            gain,
            0,
            'B'
        )

        local_stats: Dict[str, int] = {"visited": 0, "pruned": 0}
        value = minimax_alpha_beta(
            new_state,
            depth_h - 1,
            -math.inf,
            math.inf,
            use_alpha_beta,
            local_stats
        )

        total_stats["visited"] += local_stats.get("visited", 0)
        total_stats["pruned"] += local_stats.get("pruned", 0)

        if value > best_value:
            best_value = value
            best_target = t

    return best_target, int(best_value), total_stats


if __name__ == "__main__":
    print("题目四：距离矩阵（顺序：A, B, T1, T2, T3, T4）")
    matrix = compute_distance_matrix()
    for row in matrix:
        print(row)

    print()
    for h in [2, 4]:
        target_mm, value_mm, stats_mm = best_first_move(h, use_alpha_beta=False)
        target_ab, value_ab, stats_ab = best_first_move(h, use_alpha_beta=True)

        print(f"H={h}")
        print(f"  Minimax    => {{'best_target': '{target_mm}', 'best_value': {value_mm}}}")
        print(f"  Alpha-Beta => {{'best_target': '{target_ab}', 'best_value': {value_ab}}}")
        print(f"  Minimax 访问节点数: {stats_mm['visited']}")
        print(f"  Alpha-Beta 访问节点数: {stats_ab['visited']}, 剪枝次数: {stats_ab['pruned']}")
        print()