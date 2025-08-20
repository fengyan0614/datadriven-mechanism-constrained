# datadriven-mechanism-constrained
机理约束+数据驱动 
使用机理约束+数据驱动实现一种裂缝延伸模式识别的判别规则
对于某一压裂段，我已经有了净压力折算及根据净压力曲线的斜率初步判断裂缝延伸模式（定义了一个初始的判断规则进行裂缝延伸模式识别，这也就是我的机理约束）的py文件判断出的结果，excel文件（每分钟得到的n_smoothed和mode）。在此基础上，我整理了该段的微地震数据（event_srv_ratio、length_height_ratio、fracture_height），微地震特征只有一个整体值，无法逐分钟变化 → 直接把它作为「全局约束」 附加给 整段作为数据驱动，对初始的判断规则进行优化和校正。

让模型在使用判断规则判断n_smoothed 时得到的裂缝延伸模式 主动靠拢 微地震结果。
① 缝网扩展（mode 4）←→ event_srv_ratio 越大 → 该模式点应越多
② 缝高受限（mode 3）←→ length_height_ratio 越大 → 该模式点应越多
③ 缝高扩展顺利（mode 2）←→ fracture_height 越大 → 该模式点应越多）

