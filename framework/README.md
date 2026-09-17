# 半身 upper/lower 训练框架入口

实体框架目录是 `legged_lab_upper_lower/`，它来自原 `资料/legged_lab_gitee`，包含 upper/lower 配置、
AMP MDP、`pitchRoll2UpperLower` 映射模型以及独立的 `rsl_rl` 分支。

项目目录中的 `data/models` 和 `data/training` 通过软链接暴露框架内的实际模型/动作数据，避免同一份大文件
复制两遍。`rsl_rl/.git` 是原嵌套历史，公开根仓库前按 GitHub 清单处理。
