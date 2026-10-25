# 从零理解 Transformer：单头注意力 → 多头注意力 → Nanochat 源码

我查阅了 3Blue1Brown 的 Attention 讲解、Jay Alammar 的图解，以及 Sebastian Raschka 的多头注意力教程。

这些教程有一个共同的讲解思路：先理解一个 Token 如何从其他 Token 获取信息，再理解如何同时使用多个不同的注意力机制。&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.3blue1brown.com\&sz=32)

3Blue1Brown

+2



你刚才最大的困惑，其实不是多头注意力本身，而是没有弄清楚三个概念：

- Token 是什么，进入 Attention 后变成了什么？
- 一个 Attention Head 到底干了什么？
- 多个 Head 是把 Token 分开处理，还是把每个 Token 的向量分开处理？

我们先解决这三个问题，再解释代码。

## 一、先直接回答你最关心的两个问题

### 1. Attention 是训练完成之后、推理时才使用的吗？

不是。训练和推理都会执行 Attention，而且执行的是基本相同的注意力计算。

区别主要在参数是否更新：

|               | 训练阶段            | 推理阶段             |
| ------------- | --------------- | ---------------- |
| Q、K、V 投影矩阵    | 从初始化开始，通过训练不断更新 | 使用训练得到的参数，通常不再修改 |
| Attention 计算  | 执行              | 执行               |
| Softmax 注意力权重 | 根据当前输入动态计算      | 根据当前输入动态计算       |
| 反向传播          | 有               | 没有               |
| 参数更新          | 有               | 没有               |

这里尤其需要注意两种容易混淆的「权重」。

第一种：模型的可学习权重。

例如：

\\[ W_Q,\quad W_K,\quad W_V \\]

它们是需要训练的参数矩阵。模型训练时不断调整这些参数，训练完成后，普通推理中保持不变。

第二种：Attention 计算出来的注意力权重。

例如：

```
当前正在处理：“它”

关注“小明”：0.10
关注“电脑”：0.80
关注“它”：  0.10
```

这三个数加起来是 1，表示当前 Token 从不同位置获取信息的比例。

它们不是固定的模型参数，而是根据输入动态计算出来的。

所以，即使训练完成，模型每次推理仍然需要计算 Attention。

### 2. 多头 Attention 是不是把 Token 均匀拆开？

不是！这一点非常重要。

假设输入是三个 Token：

```
Token 1：小明
Token 2：电脑
Token 3：它
```

两个注意力头并不是这样分工的：

```
错误理解：

Head 1：小明、电脑
Head 2：它
```

而是：

```
正确理解：

                    ┌── Head 1：看全部三个 Token
小明、电脑、它 ───────┤
                    └── Head 2：看全部三个 Token
```

当然，对于 GPT 这种自回归语言模型，一个位置只能关注自己和之前的位置，不能偷看未来的 Token。

多头中的每个头通常都能看到当前允许关注的全部 Token。它们的区别不是处理不同的 Token，而是用不同的参数、不同的表示空间，分别分析这些 Token 之间的关系。

这个结论是理解后面所有内容的基础。

## 二、先彻底理解单头注意力

### 2.1 为什么需要 Attention？

假设输入一句话：

> 小明昨天买了一台电脑，因为它运行 Python 很快。

现在模型处理「它」。

我们人类知道，「它」大概率指的是「电脑」。

但对于计算机来说：

```
小明 → 昨天 → 买了 → 一台 → 电脑 → 因为 → 它
```

这些 Token 最开始只是一串数字向量。

模型需要一种机制，让处理「它」时，能够获取之前「电脑」的信息。

这就是 Self-Attention（自注意力）要解决的问题：

让一个 Token 根据当前上下文，决定应该从其他哪些 Token 获取多少信息。

### 2.2 什么是 Token 向量？

假设为了方便演示，我们把句子简化成三个 Token：

```
["小明", "电脑", "它"]
```

真实 Tokenizer 未必会这样切分。我们这里把它们当作三个 Token，只是为了方便理解。

每个 Token 都会被转换成一个向量。

例如，我们假设每个 Token 用四个数字表示：

```
小明 = [0.2, 0.5, 0.1, 0.8]
电脑 = [0.9, 0.1, 0.7, 0.3]
它   = [0.3, 0.6, 0.2, 0.4]
```

这些数字只是教学示例，不是实际模型生成的 Embedding。

为什么使用向量？因为神经网络无法直接对「电脑」两个汉字做矩阵运算，却可以对数字做运算。

我们将三个向量放在一起：

\\[ X= \begin{bmatrix} 0.2&0.5&0.1&0.8\\\ 0.9&0.1&0.7&0.3\\\ 0.3&0.6&0.2&0.4 \end{bmatrix} \\]

这里：

\\[ X.shape=(3,4) \\]

含义是：

- `3`：三个 Token。
- `4`：每个 Token 有四个数字。

Attention 接收的就是这样的 Token 向量，而不是直接接收汉字。

### 2.3 单头 Attention 要做的事情

想象我们正在处理「它」这个 Token。

理想情况下，模型可能会计算出这样的关注比例：

「它」对三个位置的注意力分布（假设）

小明

10%

电脑

80%

它

10%

注意：这些比例是人为举例，不是训练模型测得的结果。

这里有一个十分重要的区别：

关注「电脑」80%，并不是指模型认为「它」有 80% 的概率等于「电脑」。

而是指模型在本次信息聚合时，给「电脑」对应的信息向量分配了 80% 的权重。

接下来问题来了：

模型怎么知道应该分配 80% 给「电脑」，而不是「小明」？

这就轮到 Q、K、V 出场了。

### 2.4 Q、K、V 到底是什么？

先不用数学公式，想象一个图书馆。

你想找一本讲 Python 编程的书。

你手里有一张纸，上面写着「我想找 Python 编程资料」。

这张纸就是 Query（查询）。

图书馆每本书都有标签，例如「编程」「Python」「小说」。

这些标签就是 Key（索引）。

书里面真正的知识内容，就是 Value（信息）。

于是你做了三件事情：

第一步，拿着自己的 Query，和所有书的 Key 比较，看看哪些书更符合需求。

第二步，根据匹配程度，计算每本书的关注比例。

第三步，按关注比例从这些书的 Value 中取出信息，组合成一个结果。

这就是 Attention 的基本思想。

| 名称       | 中文理解        | 实际作用        |
| -------- | ----------- | ----------- |
| Q（Query） | 我在寻找什么      | 发起匹配        |
| K（Key）   | 我有什么可供匹配的特征 | 与 Q 计算相似程度  |
| V（Value） | 我能提供什么信息    | 根据匹配权重被加权汇总 |

注意，Q、K、V 都是数字向量，不是真正的文字标签。图书馆只是一个理解它们分工的比喻。

### 2.5 模型怎么得到 Q、K、V？

我们刚刚有三个 Token，每个 Token 是一个四维向量。

现在我们需要三套参数矩阵：

\\[ W_Q,\quad W_K,\quad W_V \\]

它们在训练开始时初始化，随后不断学习。

对于一个 Token 的输入向量 \\(x\\)，我们计算：

\\[ Q=xW_Q \\]

\\[ K=xW_K \\]

\\[ V=xW_V \\]

什么意思？

就是让同一个输入向量，经过三次不同的线性变换，得到三个不同的向量。

例如输入：

```
它 = [0.3, 0.6, 0.2, 0.4]
```

通过三个不同的参数矩阵（下面数值仅为示意）：

```
           原始 Token 向量
        [0.3, 0.6, 0.2, 0.4]
                  │
           ┌──────┼──────┐
           │      │      │
          Wq     Wk     Wv
           │      │      │
           ▼      ▼      ▼
          Q向量   K向量   V向量
```

而且，每个 Token 都要生成自己的 Q、K、V。

对于三个 Token：

```
小明 → Q1、K1、V1
电脑 → Q2、K2、V2
它   → Q3、K3、V3
```

这样，我们就可以开始计算注意力了。

### 2.6 计算「它」应该关注谁

我们现在只关注最后一个 Token「它」。

第一步，拿出它的 Query：

\\[ Q\_{\text{它}} \\]

第二步，分别和三个 Token 的 Key 计算点积：

\\[ Q\_{\text{它}}\cdot K\_{\text{小明}} \\]

\\[ Q\_{\text{它}}\cdot K\_{\text{电脑}} \\]

\\[ Q\_{\text{它}}\cdot K\_{\text{它}} \\]

点积就是把对应位置的数字相乘再相加。

例如：

\\[ [1,2]\cdot[3,4]=1\times3+2\times4=11 \\]

在 Attention 中，点积得到的是用于判断匹配程度的分数。

假设经过计算，三个分数分别为：

```
小明：1
电脑：4
它：  2
```

接着经过 Softmax，把这些分数转换成非负、总和为 1 的注意力权重。

假设最后得到了：

```
小明：0.04
电脑：0.84
它：  0.12
```

这时，还没有得到 Attention 的最终输出。

这些只是关注比例。下一步才是真正获取信息。

### 2.7 V 是干什么的？

现在我们已经知道应该关注哪些 Token 了。

假设三个 Token 的 V 向量分别是：

```
V小明 = [1, 0]
V电脑 = [0, 2]
V它   = [1, 1]
```

Attention 会按照刚才计算的关注比例，把 V 加权求和：

\\[ Output=0.04V\_{\text{小明}}+0.84V\_{\text{电脑}}+0.12V\_{\text{它}} \\]

代入数字：

\\[ =0.04[1,0]+0.84[0,2]+0.12[1,1] \\]

最后得到：

\\[ \boxed{Output=[0.16,1.80]} \\]

这就是「它」经过这个单头 Attention 后得到的新向量。

原来「它」只有自己的向量信息。现在，它的新向量已经融合了其他 Token 的信息，而且大部分来自「电脑」。

这不是直接把「它」替换成「电脑」，而是更新「它」的数字表示，让它包含与上下文有关的信息。

对其他 Token 也执行同样的操作，就可以得到所有 Token 的 Attention 输出。

### 2.8 单头 Attention 的完整公式

现在再来看那个经典公式：

\\[ \boxed{\operatorname{Attention}(Q,K,V)= \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V} \\]

经过前面的例子，你已经能理解它了。

其中 \\(QK^T\\) 负责给 Token 两两计算匹配分数，\\(\sqrt{d_k}\\) 用于对分数进行缩放（\\(d_k\\) 是 Key 向量的维度），Softmax 把分数变成注意力权重，最后乘 \\(V\\)，对信息进行加权求和。

还有一个 GPT 特有的要求：不能关注未来的 Token。因此在 Softmax 前，会把未来位置屏蔽掉。这叫 Causal Mask（因果掩码）。

[image](https://www.google.com/s2/favicons?domain=https://www.sebastianraschka.com\&sz=32)

Sebastian Raschka, PhD

+1



## 三、现在进入多头注意力：到底「多」在哪里？

这是最重要的一部分。

### 3.1 单头存在什么限制？

我们刚才训练了一个单头 Attention，它可能学会：

处理「它」时，重点从「电脑」获取信息。

但是，你可能还希望模型同时学习其他关联。

例如，处理「它」时，除了关注「电脑」，也可能需要关注「它」自己，或者与附近的其他词建立联系。

一个 Head 当然也有可能学习复杂关系，但同一查询位置在一个 Head 中只产生一套注意力分布。

多个 Head 则可以在不同的表示空间里，同时产生多套分布。

例如：

同样处理「它」，两个 Head 可以得到不同的关注比例

Head 1

一种可能的关注模式

小明

10%

电脑

80%

它

10%

Head 2

另一种可能的关注模式

小明

30%

电脑

10%

它

60%

两组比例均为假设。实际模型不会预先指定哪个 Head 负责哪种语言关系。

为什么两个 Head 的结果会不一样？

因为它们有不同的投影参数：

```
Head 1：
    Wq1、Wk1、Wv1

Head 2：
    Wq2、Wk2、Wv2
```

即使输入完全相同，不同的参数也会生成不同的 Q、K、V，从而计算出不同的关注比例和信息向量。

这就是多头注意力最核心的思想。

### 3.2 两个 Head 到底怎么分配 Token？

我们现在使用你原文中的配置：

```
T = 3C = 8n_head = 2head_dim = 4
```

有三个 Token，每个 Token 的输入向量是八维。

输入 X：每个 Token 都是 8 维

小明

1

2

3

4

5

6

7

8

电脑

1

2

3

4

5

6

7

8

它

1

2

3

4

5

6

7

8

数字 1～8 在这里仅代表维度的位置，不是实际的向量数值。

很多教程讲到这里，会说一句「将 Embedding 均匀分成两个头」，然后就直接进入下一步。

这句话很容易让人产生误解。

你可能认为：

```
原始向量：
[a, b, c, d, e, f, g, h]

Head 1 得到：
[a, b, c, d]

Head 2 得到：
[e, f, g, h]
```

但标准多头注意力不是简单地对原始输入这么切片。

实际是：

每个 Head 都从完整的八维输入出发，通过自己学习的投影矩阵，生成四维的 Q、K、V。

例如：

举一个纯数字例子。

假设某个 Token 的完整输入是：

```
X = [1, 2, 3, 4, 5, 6, 7, 8]
```

Head 1 通过自己的 Q 投影矩阵可能得到：

```
Q1 = [0.5, -0.2, 0.8, 1.1]
```

Head 2 通过另一个 Q 投影矩阵可能得到：

```
Q2 = [0.9, 0.3, -0.4, 0.7]
```

这些数值都只是示意。

两个 Head 都利用了原来的整个八维向量，但提取出了不同的四维表示。

同样地，K、V 也分别生成两套。

### 3.3 为什么还要把两个头拼起来？

我们继续。

两个 Head 独立完成 Attention 之后，假设当前 Token 得到：

```
Head 1 的输出：
[0.1, 0.2, 0.3, 0.4]

Head 2 的输出：
[0.5, 0.6, 0.7, 0.8]
```

注意，这里已经是两个头完成注意力计算后的输出了，不再是原来的输入向量。

现在我们得到了两份结果。

但下一个 Transformer 层的输入，依然需要每个 Token 使用八维向量。

所以我们将两份结果拼起来：

```
Head 1: [0.1, 0.2, 0.3, 0.4]
                         │
                         ├── 拼接
                         │
Head 2: [0.5, 0.6, 0.7, 0.8]
                         │
                         ▼
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
```

这里的拼接叫 `Concat`。

它并不是把两个 Head 的结果求平均，也不是把对应位置直接相加，而是把它们按顺序接成一个更长的向量。

为什么不直接相加？

假设：

```
Head 1 = [1, 2, 3, 4]
Head 2 = [5, 6, 7, 8]
```

如果相加：

```
结果 = [6, 8, 10, 12]
```

得到的是四维，而且两份结果已经混合，无法分别保留原来的各个分量。

而如果拼接：

```
结果 = [1, 2, 3, 4, 5, 6, 7, 8]
```

得到八维，而且两份 Head 的信息都保留下来了。

下一步，我们可以让模型学习如何组合这些信息。

这就是最后一个线性层 `c_proj` 要做的事。

### 3.4 为什么拼接之后还需要 c_proj？

你可能会问：

「既然拼接之后已经是八维了，为什么还要计算一次？直接给下一层不就行了？」

直接拼接当然可以得到八维，但此时前四维来自 Head 1，后四维来自 Head 2。

我们希望模型能够把这些信息充分结合起来。

所以再乘以一个可学习矩阵：

\\[ Y=\operatorname{Concat}(Head_1,Head_2)W_O \\]

在我们的例子中：

\\[ (1,8)\times(8,8)=(1,8) \\]

例如，输出的第一个数字可以综合使用来自两个 Head 的不同分量，而不是只取 Head 1 的第一个数字。

最终结果仍然是八维，方便继续交给后面的 Transformer 层，并与原输入执行残差相加。

这也是 Jay Alammar 在 The Illustrated Transformer 和 The Illustrated GPT-2 中采用的解释顺序：先让每个 Head 分别处理信息，随后拼接，再通过输出投影把信息混合。

[image](https://www.google.com/s2/favicons?domain=https://jalammar.github.io\&sz=32)

Jay Alammar

+1



## 四、把整个多头注意力的 Tensor 形状走一遍

现在你已经知道了原理，我们再看形状。

为了对应真实代码，我们增加一个维度 `B`：

`B` 表示 Batch Size，也就是一次处理多少条样本。

这里设为 1。

| 步骤        | Tensor Shape   | 含义                       |
| --------- | -------------- | ------------------------ |
| 输入 X      | `(1, 3, 8)`    | 1 条样本，3 个 Token，每个 8 维   |
| Q 投影      | `(1, 3, 8)`    | 为每个 Token 生成总共 8 维的 Q    |
| Q 分成两个头   | `(1, 3, 2, 4)` | 2 个头，每个 4 维              |
| 每个头的注意力分数 | `(1, 2, 3, 3)` | 2 个头，各有 3×3 的 Token 关系矩阵 |
| 每个头加权汇总 V | `(1, 3, 2, 4)` | 每个 Token 在两个头中分别得到 4 维结果 |
| 拼接两个头     | `(1, 3, 8)`    | 两个 4 维合成 8 维             |
| 输出投影      | `(1, 3, 8)`    | 保持原始维度，完成信息混合            |

其中最容易看错的是：

```
(1, 3, 8)
     ↓
(1, 3, 2, 4)
```

这不是把三个 Token 拆成两组。

三个 Token 还在。

只是原本每个 Token 的八维 Q 向量，现在被整理为两个四维 Q 向量。

K、V 的处理方式在普通多头注意力中也是一样的。

另外，三个 Token 对应的注意力分数矩阵大小为 \\(3\times3\\)，是因为每个 Token 都要与允许关注的 Token 计算匹配程度。

## 五、回到 nanochat/gpt.py，逐行理解

我对照了 [nanochat 当前 GitHub 源码](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py)。你贴出的实现和当前源码采用了相同的核心投影结构，但不同版本还可能包含一些额外功能。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



### 5.1 先看配置

```
self.n_head = config.n_headself.n_kv_head = config.n_kv_headself.n_embd = config.n_embdself.head_dim = self.n_embd // self.n_head
```

先假设：

```
n_head = 2n_kv_head = 2n_embd = 8
```

那么：

```
head_dim = 8 // 2  # 4
```

也就是说，每个头的向量宽度是 4。

### 5.2 看 Q 的投影

```
self.c_q = Linear(    self.n_embd,    self.n_head * self.head_dim,    bias=False)
```

代入刚才的数字：

```
self.c_q = Linear(8, 8, bias=False)
```

`Linear(8, 8)` 代表：

输入一个八维向量，输出一个八维向量。

你可能会问：

「这不还是八维吗？之前不是说每个头四维？」

原因是：这一个线性层一次性计算出了两个头各自的四维 Q 向量。

它的结果可以表示成：

```
Q = [q1, q2, q3, q4, q5, q6, q7, q8]
```

然后把它整理成：

```
Head 1 的 Q = [q1, q2, q3, q4]
Head 2 的 Q = [q5, q6, q7, q8]
```

你要注意顺序：

先对完整输入进行可学习投影，然后才拆分投影结果。

这是和「直接均分原始 Embedding」最本质的区别。

### 5.3 再看 view()

在 `forward()` 中：

```
B, T, C = x.size()q = self.c_q(x).view(    B, T, self.n_head, self.head_dim)
```

实际变化：

```
# 原始输入x.shape# (1, 3, 8)# 经过 Linearself.c_q(x).shape# (1, 3, 8)# 使用 view 整理维度q.shape# (1, 3, 2, 4)
```

`view()` 不会重新执行神经网络计算，它只是把已有元素按新的维度组织起来。

当然，PyTorch 的 `view()` 要求底层内存布局满足条件，这段代码中线性层的输出适合这样整理。

### 5.4 K、V 的计算

```
k = self.c_k(x).view(    B, T, self.n_kv_head, self.head_dim)v = self.c_v(x).view(    B, T, self.n_kv_head, self.head_dim)
```

如果：

```
n_head = 2n_kv_head = 2
```

那么 Q、K、V 的形状都是：

```
(1, 3, 2, 4)
```

到这里，它就是我们一直在讲的标准多头注意力结构。

不过 nanochat 还支持一种叫 GQA（Grouped-Query Attention，分组查询注意力） 的技术。

这解释了为什么 `n_head` 和 `n_kv_head` 要分别设置。

例如：

```
n_head = 4n_kv_head = 2n_embd = 8
```

那么：

```
Q：4 个 Head，每个 2 维

K：2 个 Head，每个 2 维

V：2 个 Head，每个 2 维
```

多个 Q Head 会共享对应组的 K、V Head，这样在推理时可以减少 KV Cache 的显存占用。

这属于标准多头注意力的优化版本。你现阶段先理解 `n_head = n_kv_head` 的情况即可。

[image](https://www.google.com/s2/favicons?domain=https://www.sebastianraschka.com\&sz=32)

Sebastian Raschka, PhD

+1



### 5.5 最后看拼接与输出

nanochat 当前源码有：

```
y = y.contiguous().view(B, T, -1)y = self.c_proj(y)
```

在我们这个两头例子里，注意力计算后：

```
y.shape = (1, 3, 2, 4)
```

经过 `view()`：

```
y.shape = (1, 3, 8)
```

这一步就是拼接两个 Head 的结果。

然后：

```
self.c_proj = Linear(8, 8, bias=False)
```

负责把两个 Head 的信息混合，输出仍然为：

```
(1, 3, 8)
```

nanochat 实际计算注意力时使用优化过的 Flash Attention；此外，当前源码还有 RoPE、QK Norm、KV Cache 等机制。但这些不会改变我们刚才讲解的多头注意力核心思想。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 六、你可以亲自运行的最小 PyTorch 实验

下面这段代码不需要 GPU，不训练模型，只演示两个 Head 是如何从同一份输入生成 Q，并且如何拆分的。

```
import torchfrom torch import nntorch.manual_seed(42)# 三个 Token，每个 Token 都有 8 个数字x = torch.randn(3, 8)# 头的数量和每个头的维度n_head = 2head_dim = 4# 一个线性层，一次性为两个 Head 生成 Qc_q = nn.Linear(8, 8, bias=False)# 第一步：将完整的 8 维输入进行投影q = c_q(x)print("原始输入:", x.shape)print("Q 投影后:", q.shape)# 第二步：把每个 Token 的 8 维 Q 整理成两个 4 维q = q.view(3, n_head, head_dim)print("拆分后的 Q:", q.shape)# 第三个 Token 在两个 Head 中的 Qprint("Head 1:", q[2, 0])print("Head 2:", q[2, 1])
```

运行后，形状会是：

```
原始输入: torch.Size([3, 8])
Q 投影后: torch.Size([3, 8])
拆分后的 Q: torch.Size([3, 2, 4])

Head 1: tensor([四个数字])
Head 2: tensor([四个数字])
```

两个头的数据不同，是因为投影矩阵中不同的参数参与了计算。

这里参数还是随机初始化的，并没有训练，因此你不能期待它们已经学会任何语言关系。

真实模型训练的过程，就是通过 Loss 和反向传播，不断调整这些投影参数，让不同的注意力计算对预测下一个 Token 更有帮助。

## 七、我最推荐你看的三份教程

[Attention in transformers, step-by-step | Deep ... | 3Blue1Brown](https://images.openai.com/static-rsc-4/Dbo3vxe21FnoNYoMej5_V3DhkT11-lZEgE7GBMvcPLpb2RbA5EJRNGqmFMD7fVUzs9vBWfMzxgCn-C_gJ7PbW25iWFOa8vtAoxE-_J1jpPACv1HcS0BSsKsTdiMoiUsuhp6FI4DfhvLDAFy81lsfLfZs9t_YHzR79Zm3JWvZnT4?purpose=inline)

[3Blue1Brown — Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention/)

优先推荐。通过动画解释 Q、K、V、注意力权重以及多个 Head，尤其适合建立直觉。视频约 19:19 开始讲多头注意力。

[The Illustrated GPT-2 (Visualizing Transformer Language Models) – Jay Alammar – Visualizing machine learning one concept at a time.](https://images.openai.com/static-rsc-4/V9LgHnyOgCFWZbHH3yqdfQhUO1_h-0MnhG9ks7WYUcM3j2A4_gYthdah5l-VdCQ7eO890WLMewKS9Ca7bFjHd8fEQd1YibBSy6VMqvL6zKzyuyMvETp_9X1BsCN8qlKAlGy6hGgiyaKDqH0WJA2mxLhaehDSCgfZAScZETjbzQw?purpose=inline)

[Jay Alammar — The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

非常适合你目前的学习进度。它不仅讲 Attention，还专门画图解释了如何拆成多个头、合并多个头以及做输出投影。

[Making Sense of the Multi-head Attention Mechanism | by Eng Chong Yock | Medium](https://images.openai.com/static-rsc-4/s6sYpPV6KFjnNOp4tdaznFqpN6b9KeYCr7ioxluueUhKTsDUnuDBNAeSoYvCEi1Lpxe2GCYBUzIHNmFO-a8ewukC9RJW5_L181k4g5UiibiYkC7JOrwYlGSgv0Q7BTC1-kBvvv_f4jEKSl3xsKQJ11LjNZmVUrwmT2-Z2zE1m3A?purpose=inline)

[Sebastian Raschka — Why Transformer LLMs Use Multi-Head Attention](https://www.sebastianraschka.com/faq/docs/multi-head-attention.html)

更适合在你理解基本原理后阅读。重点讲为什么分多个头、每个 Head 的维度怎么算，以及 MHA 和 GQA 的区别。

## 八、最后检查一下你是不是真的理解了

你现在应该能够区分四个动作：

| 动作              | 实际含义                      |
| --------------- | ------------------------- |
| Token Embedding | 将 Token 转成向量              |
| Q、K、V 投影        | 通过可学习矩阵，产生查询、匹配特征和信息向量    |
| 多个 Head         | 对相同的 Token 序列分别计算多套注意力    |
| Concat + 输出投影   | 拼接各个 Head 的结果，再学习如何混合这些信息 |

尤其记住下面这一点：

假设有 100 个 Token，8 个 Head。并不是每个 Head 分配 12～13 个 Token，而是每个 Head 都有机会处理全部 100 个 Token（受到因果掩码或注意力窗口限制），只不过每个 Head 使用的向量宽度通常更小，而且参数不同。

最后用一个问题验证你的理解：

假设有 10 个 Token，每个 Token 的向量长度为 512，配置 8 个 Attention Head。

每个 Head 会处理多少个 Token？它的向量维度是多少？

每个 Head 处理约 1～2 个 Token，向量 512 维

每个 Head 处理 10 个 Token，向量 64 维

每个 Head 处理 10 个 Token，向量 512 维

每个 Head 处理约 1～2 个 Token，向量 64 维
