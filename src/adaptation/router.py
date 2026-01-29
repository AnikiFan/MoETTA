import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRouter:
    def __init__(self, num_expert):
        self.num_expert = num_expert
        pass

    def __call__(self, embedding):
        """decide which expert to activate
        Returns:
            True if activate corresponding expert, otherwise False.
        """
        pass

    def update_buffer(self, expert: int):
        """
        update the buffer related to the
        specific expert with the cached embedding

        Args:
            expert: the No. of the expert to be updated
        """
        pass


class DummyRouter(BaseRouter):
    def __init__(self, num_expert):
        super().__init__(num_expert)
        self.mean = []
        self.var = []

    def __call__(self, embedding):
        """dummy router, activate all the experts"""
        self.mean.append(torch.mean(embedding, dim=[0, 1]))
        self.var.append(torch.var(embedding, dim=[0, 1]))
        return [True] * (self.num_expert)

    def update_buffer(self, expert: int):
        pass


class MLPRouter(BaseRouter):
    def __init__(self, num_expert, d_model, device: str = "cuda"):
        super().__init__(num_expert)
        self.net = nn.Linear(d_model, num_expert).to(device)
        self.net.requires_grad_(True)
        nn.init.xavier_uniform_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def __call__(self, embedding):
        return self.net(embedding)

    def get_params(self):
        return [self.net.weight, self.net.bias]


class CosineSimilarityBasedRouter(BaseRouter):
    def __init__(self, num_expert: int, top_k=3, threshold: float = 0.95):
        super().__init__(num_expert)
        self.mean = []
        self.var = []

        self.num_activated = 1
        self.avg_mean = [None] * self.num_expert
        self.accumulated_num = [0] * self.num_expert
        self.threshold = threshold
        self.mean_cache = None
        self.num_expert = num_expert
        self.top_k = top_k

    def __call__(self, embedding):
        mean = torch.mean(embedding, dim=[0, 1])
        self.mean.append(mean)
        self.var.append(torch.var(embedding, dim=[0, 1]))
        self.mean_cache = mean
        if not any(self.accumulated_num):
            return [True] + [False] * (self.num_expert - 1)
        cosine_similarity = [
            torch.cosine_similarity(self.avg_mean[i], mean, dim=0)
            for i in range(self.num_activated)
        ]
        selected = [c > self.threshold for c in cosine_similarity]
        # 有任何专家超过相似度阈值，则选择这些专家
        if any(selected):
            return selected + [False] * (self.num_expert - self.num_activated)
        # 没有现有专家超过相似度阈值
        # 可以继续添加
        if self.num_activated < self.num_expert:
            self.num_activated += 1
            return (
                [False] * (self.num_activated - 1)
                + [True]
                + [False] * (self.num_expert - self.num_activated)
            )
        # 无法继续添加新的专家，即专家数量已满
        top_k_indices = torch.argsort(torch.tensor(cosine_similarity))[-self.top_k :]
        return [True if i in top_k_indices else False for i in range(self.num_expert)]

    def update_buffer(self, expert):
        """
        update the buffer of the specific expert

        Args:
            expert: the No. of expert to be updated
        """
        self.accumulated_num[expert] += 1
        if self.accumulated_num[expert] == 1:
            self.avg_mean[expert] = self.mean_cache
        self.avg_mean[expert] = (
            self.avg_mean[expert]
            * ((self.accumulated_num[expert] - 1) / self.accumulated_num[expert])
            + self.mean_cache / self.accumulated_num[expert]
        )
        self.mean_cache = None


class SymmetricKLDivergenceBasedRouter(BaseRouter):
    def __init__(self, num_expert: int, top_k=3, threshold: float = 0.1):
        """
        Args:
            num_expert: 专家总数
            top_k: 当所有专家已激活时，选取散度最小的 top_k 个专家
            threshold: 对称 KL 散度阈值，当散度低于该阈值时认为分布相似
                       （注意：KL 散度越低表示分布越接近）
        """
        super().__init__(num_expert)
        self.num_expert = num_expert
        self.top_k = top_k
        self.threshold = threshold

        self.mean = []
        self.var = []

        self.num_activated = 1
        # 分别存储每个专家的历史均值和方差
        self.avg_mean = [None] * self.num_expert
        self.avg_var = [None] * self.num_expert
        self.accumulated_num = [0] * self.num_expert

        self.mean_cache = None
        self.var_cache = None

    def __call__(self, embedding):
        """
        计算当前输入 embedding 的均值和方差，
        与各专家存储的 (avg_mean, avg_var) 计算对称 KL 散度，
        根据阈值决定激活哪些专家，并返回一个布尔列表（长度为 num_expert）。
        """
        # 计算当前 embedding 的均值和方差
        current_mean = torch.mean(embedding, dim=[0, 1])
        current_var = torch.var(embedding, dim=[0, 1])
        self.mean_cache = current_mean
        self.var_cache = current_var

        self.mean.append(current_mean)
        self.var.append(current_var)

        if not any(self.accumulated_num):
            return [True] + [False] * (self.num_expert - 1)

        # 对每个专家计算当前输入与其存储分布之间的对称 KL 散度
        divergence_list = []
        for i in range(self.num_expert):
            if self.avg_mean[i] is None or self.avg_var[i] is None:
                divergence_list.append(float("inf"))
            else:
                # symmetric_kl_divergence 返回一个标量（Tensor或float）
                div = self.symmetric_kl_divergence(
                    self.avg_mean[i], self.avg_var[i], current_mean, current_var
                )
                divergence_list.append(div.item())

        # 判断哪些专家的散度低于阈值，认为其匹配当前输入
        selected = [div < self.threshold for div in divergence_list]

        # 如果有专家的散度低于阈值，则直接返回这些专家的激活状态
        if any(selected):
            return selected

        # 如果没有专家匹配，且专家池未填满，则激活新专家
        if self.num_activated < self.num_expert:
            # 新激活专家的索引为 num_activated
            self.num_activated += 1
            selection = (
                [False] * (self.num_activated - 1)
                + [True]
                + [False] * (self.num_expert - self.num_activated)
            )
            return selection

        # 专家池已满，则选取散度最小的 top_k 个专家
        divergence_array = torch.tensor(divergence_list)
        top_k_indices = divergence_array.argsort()[: self.top_k]
        selected = [
            True if i in top_k_indices else False for i in range(self.num_expert)
        ]
        return selected

    @staticmethod
    def symmetric_kl_divergence(mean1, var1, mean2, var2):
        """
        计算两个对角协方差高斯分布之间的对称 KL 散度。
        为避免数值不稳定，在方差上加上一个很小的 eps。

        Args:
            mean1, var1: 第一个分布的均值和方差
            mean2, var2: 第二个分布的均值和方差
        Returns:
            对称 KL 散度（标量 Tensor）
        """
        eps = 1e-6
        var1 = var1 + eps
        var2 = var2 + eps

        # 计算 KL(p||q)
        kl1 = 0.5 * torch.sum(
            var1 / var2 + ((mean2 - mean1) ** 2) / var2 - 1 + torch.log(var2 / var1)
        )
        # 计算 KL(q||p)
        kl2 = 0.5 * torch.sum(
            var2 / var1 + ((mean1 - mean2) ** 2) / var1 - 1 + torch.log(var1 / var2)
        )
        return kl1 + kl2

    def update_buffer(self, expert: int):
        """
        更新指定专家的缓存（即 avg_mean 和 avg_var）
        使用简单的运行均值更新公式：
            new_avg = old_avg * ((n-1)/n) + new_stat / n
        Args:
            expert: 要更新的专家编号
            embedding: 用于更新的 embedding（假设形状支持 torch.mean/var(embedding, dim=[0,1])）
        """
        self.accumulated_num[expert] += 1
        n = self.accumulated_num[expert]
        if self.avg_mean[expert] is None:
            self.avg_mean[expert] = self.mean_cache
        else:
            self.avg_mean[expert] = (
                self.avg_mean[expert] * ((n - 1) / n) + self.mean_cache / n
            )
        if self.avg_var[expert] is None:
            self.avg_var[expert] = self.var_cache
        else:
            self.avg_var[expert] = (
                self.avg_var[expert] * ((n - 1) / n) + self.var_cache / n
            )
        # 清除当前缓存
        self.mean_cache = None
        self.var_cache = None


class CosOrKLDBasedRouter(BaseRouter):
    def __init__(
        self,
        num_expert: int,
        top_k=3,
        kl_threshold: float = 0.1,
        cos_threshold: float = 0.9,
    ):
        """
        Args:
            num_expert: 专家总数
            top_k: 当所有专家已激活时，选取对称 KL 散度最小的 top_k 个专家
            kl_threshold: 对称 KL 散度阈值，当散度低于该阈值时认为分布相似
            cos_threshold: 余弦相似度阈值，当余弦相似度高于该阈值时认为方向相似
                         （这里的设计是：只要满足任一条件，即认为专家匹配当前输入）
        """
        super().__init__(num_expert)
        self.num_expert = num_expert
        self.top_k = top_k
        self.threshold = kl_threshold  # KL 散度阈值
        self.cos_threshold = cos_threshold

        self.mean = []
        self.var = []

        self.num_activated = 1
        # 分别存储每个专家的历史均值和方差
        self.avg_mean = [None] * self.num_expert
        self.avg_var = [None] * self.num_expert
        self.accumulated_num = [0] * self.num_expert

        self.mean_cache = None
        self.var_cache = None

    def __call__(self, embedding):
        """
        对当前输入 embedding 计算统计量，并计算每个专家的对称 KL 散度与余弦相似度。
        如果某个专家满足“相似性”标准：其对称 KL 散度低于 kl_threshold 或
        余弦相似度高于 cos_threshold，则认为该专家匹配当前输入，返回一个布尔列表。
        如果没有专家匹配，则按照如下逻辑：
            - 如果专家池尚未填满，则激活一个新专家；
            - 否则选择对称 KL 散度最小的 top_k 个专家。
        """
        # 计算当前 embedding 的均值和方差
        current_mean = torch.mean(embedding, dim=[0, 1])
        current_var = torch.var(embedding, dim=[0, 1])
        self.mean_cache = current_mean
        self.var_cache = current_var

        self.mean.append(current_mean)
        self.var.append(current_var)

        # 若还未更新过任何专家，则直接激活第一个专家
        if not any(self.accumulated_num):
            return [True] + [False] * (self.num_expert - 1)

        divergence_list = []
        cosine_list = []
        for i in range(self.num_expert):
            if self.avg_mean[i] is None or self.avg_var[i] is None:
                divergence_list.append(float("inf"))
                cosine_list.append(float("-inf"))
            else:
                # 计算对称 KL 散度
                div = self.symmetric_kl_divergence(
                    self.avg_mean[i], self.avg_var[i], current_mean, current_var
                )
                divergence_list.append(div.item())
                # 计算余弦相似度（当前均值与专家的历史均值）
                cos_sim = F.cosine_similarity(current_mean, self.avg_mean[i], dim=0)
                cosine_list.append(cos_sim.item())

        # 判断匹配标准：只要满足“KL 散度足够低”或“余弦相似度足够高”中的任意一个，即认为匹配
        selected = [
            (divergence_list[i] < self.threshold)
            or (cosine_list[i] > self.cos_threshold)
            for i in range(self.num_expert)
        ]

        if any(selected):
            return selected

        # 如果没有匹配的专家且专家池未填满，则激活新专家
        if self.num_activated < self.num_expert:
            self.num_activated += 1
            selection = (
                [False] * (self.num_activated - 1)
                + [True]
                + [False] * (self.num_expert - self.num_activated)
            )
            return selection

        # TODO: 这里或许也可以用KL和Cos加权来选择？
        # 专家池已满，则选取对称 KL 散度最小的 top_k 个专家
        divergence_array = torch.tensor(divergence_list)
        top_k_indices = divergence_array.argsort()[: self.top_k]
        selected = [
            True if i in top_k_indices else False for i in range(self.num_expert)
        ]
        return selected

    @staticmethod
    def symmetric_kl_divergence(mean1, var1, mean2, var2):
        """
        计算两个对角协方差高斯分布之间的对称 KL 散度。
        为避免数值不稳定，在方差上加上一个很小的 eps。

        Args:
            mean1, var1: 第一个分布的均值和方差
            mean2, var2: 第二个分布的均值和方差
        Returns:
            对称 KL 散度（标量 Tensor）
        """
        eps = 1e-6
        var1 = var1 + eps
        var2 = var2 + eps

        # 计算 KL(p||q)
        kl1 = 0.5 * torch.sum(
            var1 / var2 + ((mean2 - mean1) ** 2) / var2 - 1 + torch.log(var2 / var1)
        )
        # 计算 KL(q||p)
        kl2 = 0.5 * torch.sum(
            var2 / var1 + ((mean1 - mean2) ** 2) / var1 - 1 + torch.log(var1 / var2)
        )
        return kl1 + kl2

    def update_buffer(self, expert: int):
        """
        更新指定专家的缓存（即 avg_mean 和 avg_var）。
        使用简单的运行均值更新公式：
            new_avg = old_avg * ((n-1)/n) + new_stat / n
        Args:
            expert: 要更新的专家编号
        """
        self.accumulated_num[expert] += 1
        n = self.accumulated_num[expert]
        if self.avg_mean[expert] is None:
            self.avg_mean[expert] = self.mean_cache
        else:
            self.avg_mean[expert] = (
                self.avg_mean[expert] * ((n - 1) / n) + self.mean_cache / n
            )
        if self.avg_var[expert] is None:
            self.avg_var[expert] = self.var_cache
        else:
            self.avg_var[expert] = (
                self.avg_var[expert] * ((n - 1) / n) + self.var_cache / n
            )
        # 清除当前缓存
        self.mean_cache = None
        self.var_cache = None
