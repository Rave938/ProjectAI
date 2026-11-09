from typing import Any, Dict, Optional
import math


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


def build_tree(obj: Dict[str, Any]) -> TreeNode:
    """
    根据字典构建 TreeNode。
    叶子节点包含 "value"；非叶子节点包含 "children"。
    """
    if "value" in obj:
        return TreeNode(value=obj["value"])

    children = [build_tree(child) for child in obj.get("children", [])]
    return TreeNode(children=children)


def minimax(
    node: TreeNode,
    maximizing: bool = True,
    stats: Optional[Dict[str, int]] = None
) -> float:
    """
    普通 Minimax。
    """
    if stats is None:
        stats = {}
    stats["visited"] = stats.get("visited", 0) + 1

    if not node.children:
        return node.value

    if maximizing:
        best = -math.inf
        for child in node.children:
            best = max(best, minimax(child, False, stats))
        return best
    else:
        best = math.inf
        for child in node.children:
            best = min(best, minimax(child, True, stats))
        return best


def alpha_beta(
    node: TreeNode,
    alpha: float = -math.inf,
    beta: float = math.inf,
    maximizing: bool = True,
    stats: Optional[Dict[str, int]] = None
) -> float:
    """
    Alpha-Beta 剪枝。
    """
    if stats is None:
        stats = {}
    stats["visited"] = stats.get("visited", 0) + 1

    if not node.children:
        return node.value

    if maximizing:
        value = -math.inf
        for child in node.children:
            value = max(
                value,
                alpha_beta(child, alpha, beta, False, stats)
            )
            alpha = max(alpha, value)
            if beta <= alpha:
                stats["pruned"] = stats.get("pruned", 0) + 1
                break
        return value
    else:
        value = math.inf
        for child in node.children:
            value = min(
                value,
                alpha_beta(child, alpha, beta, True, stats)
            )
            beta = min(beta, value)
            if beta <= alpha:
                stats["pruned"] = stats.get("pruned", 0) + 1
                break
        return value


def solve_tree(tree_structure: Dict[str, Any]) -> int:
    root = build_tree(tree_structure)
    return int(alpha_beta(root, -math.inf, math.inf, True))


if __name__ == "__main__":
    # 按题目输入样例构造树：
    # 根为 MAX，下面两个 MIN 子节点。
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

    print("题目三输出：")
    print("Minimax 最佳得分：", int(minimax_score))
    print("Alpha-Beta 最佳得分：", int(alpha_score))
    print("Minimax 访问节点数：", minimax_stats.get("visited", 0))
    print("Alpha-Beta 访问节点数：", alpha_stats.get("visited", 0))
    print("Alpha-Beta 剪枝次数：", alpha_stats.get("pruned", 0))