from github import Github
from github import Auth
import pandas as pd
import time
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
GITHUB_TOKEN = "这里填入你自己的GitHub令牌"
auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)
repo = g.get_repo("pandas-dev/pandas")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据采集
print("开始爬取纯Issue数据")

def crawl_pure_issues(repo, total_limit=1000):

    pure_issues_list = []
    issues_paginator = repo.get_issues(state="closed")  # 爬取已处理的纯Issue
    
    for issue in issues_paginator:
        # 过滤PR
        if issue.pull_request:
            continue
        if len(pure_issues_list) >= total_limit:
            break
        issues_dict = {
            "issue_number": issue.number,
            "title": issue.title,
            "content": issue.body if issue.body else "",
            "labels": [label.name for label in issue.labels],
            "create_time": issue.created_at,
            "close_time": issue.closed_at,
            "comments_count": issue.comments,
            "state": issue.state
        }
        pure_issues_list.append(issues_dict)
        time.sleep(0.1)  # 避免API限流
        if len(pure_issues_list) % 100 == 0:
            print(f"已爬取{len(pure_issues_list)}条Issue")
    
    return pd.DataFrame(pure_issues_list)

def crawl_contributors(repo, top_n=50):
    contributors_list = []
    contributors = repo.get_contributors()
    count = 0
    for contributor in contributors:
        if count >= top_n:
            break
        contributors_dict = {
            "user_id": contributor.id,
            "user_name": contributor.login,
            "contributions": contributor.contributions,
            "email": contributor.email if contributor.email else "",
            "github_url": contributor.html_url
        }
        contributors_list.append(contributors_dict)
        count += 1
    return pd.DataFrame(contributors_list)

# 执行爬取
issues_raw_df = crawl_pure_issues(repo, total_limit=1000)
contributors_raw_df = crawl_contributors(repo)
# 输出原始数据
issues_raw_df.to_csv("github_pure_issues_raw.csv", index=False, encoding="utf-8-sig")
contributors_raw_df.to_csv("github_contributors_raw.csv", index=False, encoding="utf-8-sig")

print("===== 纯Issue原始数据已输出为CSV =====")
print(f"爬取的纯Issue数据前5行：\n{issues_raw_df.head()}")
print(f"本次爬取纯Issue总数：{len(issues_raw_df)}")

# 数据预处理（过滤无标签+清洗）
print("\n===== 开始数据预处理 =====")

issues_clean_df = issues_raw_df.copy()
# 去重
issues_clean_df = issues_clean_df.drop_duplicates(subset="issue_number", keep="first")
# 删空内容Issue
issues_clean_df = issues_clean_df[issues_clean_df["content"].str.strip() != ""]
# 剔除异常
issues_clean_df = issues_clean_df[~((issues_clean_df["close_time"].isna()) & (issues_clean_df["state"] == "closed"))]
# 时间格式统一
issues_clean_df["create_time"] = pd.to_datetime(issues_clean_df["create_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
issues_clean_df["close_time"] = pd.to_datetime(issues_clean_df["close_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

# 统计纯Issue的标签分布
print("\n📊 纯Issue标签分布统计：")
no_label_count = len(issues_clean_df[issues_clean_df["labels"].apply(lambda x: len(x) == 0)])
has_label_count = len(issues_clean_df) - no_label_count
print(f"预处理后纯Issue总数：{len(issues_clean_df)}")
print(f"其中：无标签={no_label_count}条，有标签={has_label_count}条")

# 贡献者数据填充
contributors_clean_df = contributors_raw_df.copy()
contributors_clean_df["email"] = contributors_clean_df["email"].fillna("未公开")

issues_clean_df.to_csv("github_pure_issues_clean.csv", index=False, encoding="utf-8-sig")
contributors_clean_df.to_csv("github_contributors_clean.csv", index=False, encoding="utf-8-sig")


# 贡献者分析
contributors_clean_df = contributors_clean_df.sort_values(by="contributions", ascending=False)
top10_contrib = contributors_clean_df.head(10)["contributions"].sum()
total_contrib = contributors_clean_df["contributions"].sum()
top10_ratio = (top10_contrib / total_contrib) * 100

contributors_analysis_df = contributors_clean_df.copy()
contributors_analysis_df["contribution_ratio"] = (contributors_analysis_df["contributions"] / total_contrib) * 100
contributors_analysis_df.to_csv("github_contributors_analysis.csv", index=False, encoding="utf-8-sig")
print(f"\n👥 贡献者分析结果：")
print(f"前10位核心贡献者提交占比：{top10_ratio:.1f}%")

# Issue标签分析
filtered_issues = issues_clean_df[issues_clean_df["labels"].apply(lambda x: len(x) > 0)]
# 展开标签
labels_flat = []
for labels in filtered_issues["labels"]:
    labels_flat.extend(labels)
labels_count = pd.Series(labels_flat).value_counts()
top_n = 10
auto_core_labels = labels_count.head(top_n).index.tolist()

print(f"\n提取的Top{top_n}高频业务标签\n{auto_core_labels}")

# 统计标签占比
core_labels_count = labels_count[auto_core_labels].reset_index()
core_labels_count.columns = ["label_type", "count"]
core_labels_count["ratio"] = (core_labels_count["count"] / core_labels_count["count"].sum()) * 100
core_labels_count.to_csv("github_issues_labels_analysis.csv", index=False, encoding="utf-8-sig")
print(f"\n纯IssueTop{top_n}业务标签分析结果：\n{core_labels_count}")

# Issue解决时长
closed_issues = filtered_issues[filtered_issues["state"] == "closed"].copy()
closed_issues["create_time"] = pd.to_datetime(closed_issues["create_time"])
closed_issues["close_time"] = pd.to_datetime(closed_issues["close_time"])
closed_issues["resolve_days"] = (closed_issues["close_time"] - closed_issues["create_time"]).dt.days

resolve_time_by_label = []
for label in auto_core_labels:
    label_issues = closed_issues[closed_issues["labels"].apply(lambda x: label in x)]
    avg_days = label_issues["resolve_days"].mean() if len(label_issues) > 0 else 0.0
    resolve_time_by_label.append({"label_type": label, "avg_resolve_days": round(avg_days, 1)})

resolve_time_df = pd.DataFrame(resolve_time_by_label)
resolve_time_df.to_csv("github_issues_resolve_time.csv", index=False, encoding="utf-8-sig")
print(f"\n⏱纯IssueTop{top_n}业务标签解决时长分析结果：\n{resolve_time_df}")

# 数据可视化
print("\n===== 开始生成可视化图表 =====")

# 图表1：贡献者占比饼图
pie_data = [top10_contrib, total_contrib - top10_contrib]
pie_labels = [f"前10位贡献者\n({top10_ratio:.1f}%)", f"其他40位\n({100-top10_ratio:.1f}%)"]
plt.figure(figsize=(8, 8))
plt.pie(pie_data, labels=pie_labels, autopct="%1.1f%%", colors=["#ff7f0e", "#2ca02c"])
plt.title("pandas仓库贡献者提交占比", fontsize=14)
plt.tight_layout()
plt.savefig("contributors_contrib_ratio.png", dpi=300)

# 图表2：Top10业务标签分布
plt.figure(figsize=(12, 6))
sns.barplot(data=core_labels_count, x="label_type", y="count", palette="Set2")
plt.title(f"pandas仓库纯Issue Top{top_n}业务标签分布", fontsize=14)
plt.xlabel("标签类型")
plt.ylabel("数量")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("issues_labels_dist.png", dpi=300)

print("\n 全流程完成！输出文件清单：")
print("【原始数据】：github_pure_issues_raw.csv、github_contributors_raw.csv")
print("【预处理数据】：github_pure_issues_clean.csv、github_contributors_clean.csv")
print("【分析结果】：github_contributors_analysis.csv、github_issues_labels_analysis.csv、github_issues_resolve_time.csv")
print("【可视化图表】：contributors_contrib_ratio.png、issues_labels_dist.png")