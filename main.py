from q1_maze import astar_maze
from q2_weighted import astar_weighted
from q3_alpha_beta import build_tree, minimax, alpha_beta
from q4_game import compute_distance_matrix, best_first_move
import math


def run_q1():
    start = (0, 0)
    goal = (3, 4)
    grid = [
        ['S', ' ', ' ', 'X', ' '],
        [' ', 'X', ' ', ' ', ' '],
        [' ', 'X', ' ', ' ', ' '],
        [' ', 'X', ' ', ' ', ' '],
        [' ', 'X', ' ', ' ', ' '],
    ]
    path = astar_maze(start, goal, grid)
    print("=" * 60)
    print("题目一：迷宫寻路")
    print("输出：", path)


def run_q2():
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
    print("=" * 60)
    print("题目二：加权地形")
    print("输出：", ans)


def run_q3():
    tree_structure = {
        "is_maximizing_player": True,
        "children": [
            {
                "is_maximizing_player": False,
                "children": [
                    {
                        "is_maximizing_player": True,
                        "children": [
                            {"value": 3},
                            {"value": 5}
                        ]
                    }
                ]
            },
            {
                "is_maximizing_player": False,
                "children": [
                    {
                        "is_maximizing_player": True,
                        "children": [
                            {"value": 6},
                            {"value": 9}
                        ]
                    },
                    {
                        "is_maximizing_player": True,
                        "children": [
                            {"value": 1},
                            {"value": 2}
                        ]
                    }
                ]
            }
        ]
    }

    root = build_tree(tree_structure)

    minimax_stats = {}
    alpha_stats = {}

    minimax_score = minimax(root, True, minimax_stats)
    alpha_score = alpha_beta(root, -math.inf, math.inf, True, alpha_stats)

    print("=" * 60)
    print("题目三：Alpha-Beta 剪枝")
    print("Minimax 最佳得分：", int(minimax_score))
    print("Alpha-Beta 最佳得分：", int(alpha_score))
    print("Minimax 访问节点数：", minimax_stats.get("visited", 0))
    print("Alpha-Beta 访问节点数：", alpha_stats.get("visited", 0))
    print("Alpha-Beta 剪枝次数：", alpha_stats.get("pruned", 0))


def run_q4():
    print("=" * 60)
    print("题目四：竞争环境下的宝藏分配")
    print("距离矩阵（顺序：A, B, T1, T2, T3, T4）")
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


if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
    run_q4()