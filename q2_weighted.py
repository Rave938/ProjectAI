from typing import List, Tuple
import heapq


def _is_walkable(cell) -> bool:
    return cell != 'X'


def _enter_cost(cell) -> int:
    """
    进入某个格子的代价。
    规则：
        数字：数字本身
        E：0
        S、A、B、T1...：1
    """
    if cell == 'X':
        raise ValueError("X 不可通行")

    if isinstance(cell, int):
        return cell

    s = str(cell)

    if s == 'E':
        return 0
    if s == 'S':
        return 0
    if s.isdigit():
        return int(s)
    if s in ('A', 'B') or s.startswith('T'):
        return 1

    # 空白等默认按 1 处理
    return 1


def astar_weighted(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid: List[List]
) -> int:
    """
    题目二：加权地形 A*。

    返回从 start 到 goal 的最小总代价；无解返回 -1。
    进入节点的代价为该节点地形权值；若目标为 E，则进入 E 的代价为 0。
    """
    if not grid or not grid[0]:
        return -1

    rows, cols = len(grid), len(grid[0])

    if start == goal:
        return 0

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return -1
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return -1
    if not _is_walkable(grid[start[0]][start[1]]) or not _is_walkable(grid[goal[0]][goal[1]]):
        return -1

    # 计算最小正进入代价，用于启发函数可采纳性
    positive_costs = []
    for r in range(rows):
        for c in range(cols):
            if _is_walkable(grid[r][c]):
                cost = _enter_cost(grid[r][c])
                if cost > 0:
                    positive_costs.append(cost)

    min_positive_cost = min(positive_costs) if positive_costs else 1

    goal_is_E = (str(grid[goal[0]][goal[1]]) == 'E')

    def heuristic(r: int, c: int) -> int:
        manhattan = abs(r - goal[0]) + abs(c - goal[1])
        if goal_is_E:
            # 最后一步进入 E 的代价为 0，因此 h = max(0, manhattan - 1) * min_cost
            return max(0, manhattan - 1) * min_positive_cost
        return manhattan * min_positive_cost

    open_heap = [(heuristic(start[0], start[1]), 0, start)]
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
            if not _is_walkable(grid[nr][nc]):
                continue

            step_cost = _enter_cost(grid[nr][nc])
            new_g = cur_g + step_cost

            if new_g < g_cost.get(nxt, 10**18):
                g_cost[nxt] = new_g
                f = new_g + heuristic(nr, nc)
                heapq.heappush(open_heap, (f, new_g, nxt))

    return -1


if __name__ == "__main__":
    start = (0, 0)
    goal = (4, 4)
    grid = [
        [5, 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, 'E'],
    ]

    ans = astar_weighted(start, goal, grid)
    print("题目二输出：")
    print(ans)