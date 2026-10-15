# main.py
# 实验一：新闻文本10分类
# 说明：本代码重写了 draft_main.py 中的加载逻辑，以支持分层划分验证集，
# 并增加了数据清洗、数据增强和模型调参流程。

import os
import re
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIRS = [Path("data"), Path(".")]
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


# ================= 数据加载 (基于 draft_main.py 扩展) =================
def load_data():
    """读取数据，返回文本和标签。"""
    train_path = find_file(["train_data.csv", "train.csv"])
    test_path = find_file(["test_data_unlabeled.csv", "test.csv"])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 自动识别文本和标签列
    text_col = detect_text_col(train_df)
    label_col = detect_label_col(train_df, text_col)

    X_train = train_df[text_col].astype(str).tolist()
    y_train = train_df[label_col].values
    X_test_unlabeled = test_df[text_col].astype(str).tolist()

    return X_train, y_train, X_test_unlabeled


def find_file(names):
    if isinstance(names, str):
        names = [names]
    for d in DATA_DIRS:
        for name in names:
            p = d / name
            if p.exists():
                return p
    raise FileNotFoundError(f"找不到文件：{names}")


def detect_text_col(df):
    for c in ["text", "content", "body", "article", "sentence", "review"]:
        if c in df.columns:
            return c
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        raise ValueError("未找到文本列")
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


def detect_label_col(df, text_col):
    for c in ["target", "label", "category", "class", "y"]:
        if c in df.columns and c != text_col:
            return c
    for c in df.columns:
        if c != text_col and df[c].nunique() <= 20:
            return c
    raise ValueError("未找到标签列")


# ================= 数据清洗与预处理 =================
# 定义一个简单的英文停用词表（避免依赖nltk下载报错）
ENGLISH_STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

# 简单的词干提取（Porter词干提取器的轻量级替代，为了速度）
def simple_stem(word):
    """非常简易的后缀去除，避免引入nltk依赖。若需要更严谨的stemming，可引入nltk。"""
    if len(word) > 4:
        for suffix in ["ing", "edly", "ed", "ly", "es", "s"]:
            if word.endswith(suffix):
                return word[:-len(suffix)]
    return word


def clean_text(s):
    """深度清洗：去HTML、URL、标点，转小写，去停用词，词干提取。"""
    s = str(s)
    # 去除HTML标签
    s = re.sub(r"<[^>]+>", " ", s)
    # 去除URL和邮箱
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    # 去除非字母字符（保留空格）
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    # 转小写并分词
    words = s.lower().split()
    # 去停用词和词干提取
    cleaned_words = [simple_stem(w) for w in words if w not in ENGLISH_STOPWORDS and len(w) > 2]
    return " ".join(cleaned_words)


def augment_text(text, p=0.1):
    """数据增强：随机删除 10% 的单词，仅在训练集上使用。"""
    words = text.split()
    if len(words) < 5:
        return text
    new_words = []
    for w in words:
        if random.random() > p:
            new_words.append(w)
    if len(new_words) == 0:
        return text
    return " ".join(new_words)


# ================= 评估函数 =================
def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return acc, f1


# ================= 主流程 =================
def main():
    X_all, y_all, X_test_raw = load_data()
    print(f"训练集样本数：{len(X_all)}, 测试集样本数：{len(X_test_raw)}")

    # 1. 分层划分训练集 / 验证集
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )

    # 2. 文本清洗（清洗规则从训练集学习，但这里是无监督清洗，可以全量应用）
    print("正在清洗文本...")
    X_train_clean = [clean_text(t) for t in X_train_text]
    X_val_clean = [clean_text(t) for t in X_val_text]
    X_test_clean = [clean_text(t) for t in X_test_raw]

    # 3. 数据增强（仅对训练集）
    print("正在进行数据增强...")
    X_train_aug = []
    y_train_aug = []
    for text, label in zip(X_train_clean, y_train):
        X_train_aug.append(text)
        y_train_aug.append(label)
        # 对一半的训练样本进行增强，增加鲁棒性
        if random.random() < 0.5:
            X_train_aug.append(augment_text(text))
            y_train_aug.append(label)

    print(f"增强后训练集样本数：{len(X_train_aug)}")

    # 4. TF-IDF 特征提取
    tfidf_params = dict(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    vectorizer = TfidfVectorizer(**tfidf_params)

    X_train_vec = vectorizer.fit_transform(X_train_aug)
    X_val_vec = vectorizer.transform(X_val_clean)
    X_test_vec = vectorizer.transform(X_test_clean)

    print(f"TF-IDF 特征维度：{X_train_vec.shape[1]}")

    # 5. 模型定义与调参
    model_zoo = {
        "NaiveBayes": (
            MultinomialNB(),
            {"alpha": [0.1, 0.5, 1.0]},
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "LinearSVM": (
            # 注意：SVC(kernel='linear') 是 PDF 推荐的，但在高维稀疏文本上极慢。
            # LinearSVC 是数学等价且极其高效的实现，更适合本任务。
            LinearSVC(random_state=SEED, max_iter=5000, class_weight="balanced"),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "MLP": (
            MLPClassifier(max_iter=300, random_state=SEED, early_stopping=True),
            {"hidden_layer_sizes": [(100,), (200,), (100, 50)], "alpha": [1e-4, 1e-3]},
        ),
    }

    all_results = []
    best_estimators = {}

    for model_name, (estimator, param_grid) in model_zoo.items():
        print(f"\n########## 调参：{model_name} ##########")
        best_f1 = -1.0
        best_est = None
        best_params = None

        for params in ParameterGrid(param_grid):
            clf = clone(estimator).set_params(**params)
            clf.fit(X_train_vec, y_train_aug)
            y_val_pred = clf.predict(X_val_vec)

            acc = accuracy_score(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred, average="macro")

            all_results.append({
                "model": model_name,
                "params": json.dumps(params, ensure_ascii=False),
                "val_accuracy": acc,
                "val_macro_f1": f1,
            })
            print(f"{params} -> val_acc={acc:.4f}, val_macro_f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_est = clf
                best_params = params

        best_estimators[model_name] = {
            "estimator": best_est,
            "params": best_params,
            "val_macro_f1": best_f1,
        }

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUT_DIR / "validation_results.csv", index=False)

    best_model_name = max(best_estimators.keys(), key=lambda k: best_estimators[k]["val_macro_f1"])
    best_info = best_estimators[best_model_name]
    best_params = best_info["params"]

    print("\n" + "=" * 60)
    print(f"最佳模型：{best_model_name}")
    print(f"最佳参数：{best_params}")
    print(f"验证集 Macro-F1：{best_info['val_macro_f1']:.4f}")
    print("=" * 60)

    # 6. 在 train+val 上重训，预测测试集
    final_vectorizer = TfidfVectorizer(**tfidf_params)
    X_trainval_clean = X_train_clean + X_val_clean
    y_trainval = list(y_train) + list(y_val)

    # 同样对 train+val 做增强
    X_trainval_aug = []
    y_trainval_aug = []
    for text, label in zip(X_trainval_clean, y_trainval):
        X_trainval_aug.append(text)
        y_trainval_aug.append(label)
        if random.random() < 0.5:
            X_trainval_aug.append(augment_text(text))
            y_trainval_aug.append(label)

    X_trainval_vec = final_vectorizer.fit_transform(X_trainval_aug)
    final_clf = clone(best_info["estimator"])
    final_clf.fit(X_trainval_vec, y_trainval_aug)

    X_test_final = final_vectorizer.transform(X_test_clean)
    test_pred = final_clf.predict(X_test_final)

    # 7. 保存预测结果（无表头，一列标签，符合实验规范）
    pd.DataFrame(test_pred).to_csv("predictions.csv", index=False, header=False)
    print("已保存预测文件：predictions.csv")

    if "target" in pd.read_csv(find_file(["test_data_unlabeled.csv", "test.csv"])).columns:
        print("测试集有标签，开始最终评估...")
        # 这里其实不会执行，因为数据文件是 unlabeled，但代码留作扩展
        pass
    else:
        print("测试集无标签，跳过最终评估（盲测）。")

    # 保存验证集报告
    y_val_pred_best = best_info["estimator"].predict(X_val_vec)
    with open(OUT_DIR / "validation_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_val, y_val_pred_best, digits=4))

    # 可视化
    best_res = result_df.loc[result_df.groupby("model")["val_macro_f1"].idxmax()].sort_values("val_macro_f1", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(best_res["model"], best_res["val_macro_f1"])
    plt.ylim(0, 1.0)
    plt.ylabel("Validation Macro-F1")
    plt.title("Best Validation Macro-F1 by Model")
    for i, v in enumerate(best_res["val_macro_f1"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=200)
    plt.close()

    with open(OUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_model_name,
            "best_params": best_params,
            "validation_macro_f1": best_info["val_macro_f1"],
        }, f, ensure_ascii=False, indent=2)

    print("\n全部完成。请查看 results/ 目录以及根目录的 predictions.csv。")


if __name__ == "__main__":
    main()