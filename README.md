# regression4ger

曲阜师范大学数学学院毕业设计，使用线性回归 (Linear Regression) 预测中国高中教育毛入学率 (Gross Enrollment Rate, GER)，附带原始数据.

## 方法

我们首先将从国家统计局中获得的大量原始数据转换为特征，即线性回归的自变量。具体地，我们对每一年设定了一个时间跨度为 K 的滑动窗口，我们对滑动窗口内的数据采用非线性变化构造了若干个特征；之后，我们对生成的特征采用Pearson相关系数进行初步筛选；最后采用前向搜索的方式确定出最终 F 个特征。我们采用留一交叉验证的方式对数据进行拟合和测试，最终模型对各测试年份的预测百分误差均未超过 0.36%，取得了出色的预测效果。

## 安装依赖

```shell
pip install -r requirements.txt
```

## 使用

```shell
python  main.py [-h] [-d DATA] [--window-size K] [--min-pearson r] [--feature-size F]
```

### 命令行参数说明

> Predict GER (Gross Enrollment Rate) using linear regression.
>
> optional arguments:
>   -h, --help        show this help message and exit
>   -d DATA           The path to the raw data (default: data.csv).
>   --window-size K   Size of the time sliding window K (default: 3).
>   --min-pearson r   Minimum Pearson correlation coefficient between feature and GER (default: 0.9).
>   --feature-size F  Size of generated features (default: 20).
