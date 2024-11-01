import numpy as np
import networkx as nx
import random

class HospitalModel:
    def __init__(self, num_doctors, avg_degree, min_visit_limit, max_visit_limit, weight_mean, weight_std, mean_visits,
                 simulation_days, daily_patient_count, referral_probability, alpha):
        self.num_doctors = num_doctors  # 医生数量
        self.avg_degree = avg_degree  # 平均连接度
        self.min_visit_limit = min_visit_limit  # 每位医生最小接诊量
        self.max_visit_limit = max_visit_limit  # 每位医生最大接诊量
        self.weight_mean = weight_mean  # 边权重均值
        self.weight_std = weight_std  # 边权重标准差
        self.mean_visits = mean_visits  # 转诊次数上限
        self.simulation_days = simulation_days  # 模拟天数
        self.daily_patient_count = daily_patient_count  # 每日新增患者数量
        self.referral_probability = referral_probability  # 完全随机转诊概率
        self.alpha = alpha  # 权重偏好 alpha

        # 生成医院网络
        self.hospital_network = self.generate_hospital_network()

        # 执行模拟患者流程，记录每日损失患者数、转诊次数、剩余容量和停止天数
        self.LP_daily_dict, self.R_daily_dict, self.FC_daily_dict, self.TLP, self.TR, self.FC, self.stop_day = self.simulate_patient_process()

        # 计算并记录模拟结果相关参数
        self.TAR, self.VAR, self.stop_day, self.LP_day_ratio, self.TLP_ratio, self.FC_ratio, self.m, self.Np, self.Q, self.alpha = self.return_parameters()

    def generate_hospital_network(self):
        # 使用 Barabási-Albert 图生成医院网络
        G = nx.barabasi_albert_graph(self.num_doctors, self.avg_degree)

        # 将生成的无向图转换为有向图
        G_directed = nx.DiGraph()
        for i, j in G.edges():
            if random.uniform(0, 1) > 0.5:
                G_directed.add_edges_from([(i, j)])
            else:
                G_directed.add_edges_from([(j, i)])
        G_directed.add_edges_from(G.edges())
        G = G_directed

        # 为每个节点设置随机接诊量和边权重
        for node in G.nodes:
            G.nodes[node]['visit_limit'] = np.random.randint(self.min_visit_limit, self.max_visit_limit)
        for edge in G.edges:
            G.edges[edge]['weight'] = np.abs(np.random.normal(self.weight_mean, self.weight_std))
        return G

    def calculate_referral_probabilities(self, T, day, predecessors_list, successors_list):
        # 计算接收转诊概率矩阵A_ij
        r_i = np.zeros(self.num_doctors, dtype=float)
        for node in range(self.num_doctors):
            if predecessors_list[node]:
                total_r_i = 0
                for neighbor in predecessors_list[node]:
                    if successors_list[neighbor]:
                        weights_sum = sum(
                            self.hospital_network[neighbor][neighbor2]['weight'] ** self.alpha for neighbor2
                            in successors_list[neighbor])
                        total_r_i += (self.hospital_network[neighbor][node]['weight'] ** self.alpha) / weights_sum * \
                                     T[day][neighbor]
                r_i[node] = (1 - self.referral_probability) * total_r_i  # 每个医生接收的邻居权重转诊患者数

        s_i = self.referral_probability * T[day].sum() / self.num_doctors  # 每个医生平均接收的随机转诊患者数
        theta = np.round(r_i + s_i, 2)  # 每个医生接收的转诊患者总数theta,保留两位小数
        return theta

    def simulate_patient_process(self):
        # 初始化每日损失患者数、转诊次数和剩余容量字典，以及其他相关变量
        LP_daily_dict = {}
        R_daily_dict = {}
        FC_daily_dict = {}
        num_referral = np.zeros(self.mean_visits + 2)
        simulation_days = self.simulation_days

        # 初始化每日剩余容量矩阵
        F = np.array([self.hospital_network.nodes[node]['visit_limit'] for node in self.hospital_network.nodes],
                     dtype=float)

        # 初始化转诊矩阵
        T = np.zeros((simulation_days + 1, self.num_doctors))

        # 获取医院网络的前驱和后继节点列表
        predecessors_list = {node: list(self.hospital_network.predecessors(node)) for node in
                             self.hospital_network.nodes}
        successors_list = {node: list(self.hospital_network.successors(node)) for node in self.hospital_network.nodes}

        # 初始化每日统计字典，记录第0天的统计信息
        LP_daily_dict[0] = 0
        R_daily_dict[0] = 0
        FC_daily_dict[0] = F.sum()

        # 初始化模拟停止天数
        stop_day = simulation_days

        # 开始模拟每日患者流程
        for day in range(1, simulation_days + 1):
            LP_daily_dict[day] = 0
            R_daily_dict[day] = 0
            FC_daily_dict[day] = F.sum()

            # 对每日的每一步进行模拟
            for step in range(1, self.mean_visits + 2):
                if step == 1:
                    theta = np.full(self.num_doctors,
                                    round((self.daily_patient_count + LP_daily_dict[day - 1]) / self.num_doctors, 2), dtype=float)
                else:
                    theta = self.calculate_referral_probabilities(T, day, predecessors_list, successors_list)

                # 更新医生状态
                F -= theta
                T[day] = np.where(F < 0, np.abs(F), 0)
                F = np.where(F < 0, 0, F)
                num_referral[step] = T[day].sum()

                if step != self.mean_visits + 1:   # 统计每日转诊数
                    R_daily_dict[day] += num_referral[step]
                else:
                    LP_daily_dict[day] = num_referral[step]  # 统计每日损失患者数
                    print(step)
                    FC_daily_dict[day] = F.sum()
                    if day < simulation_days:
                        T[day + 1] = np.zeros(self.num_doctors, dtype=float)  # 初始化每天转诊向量T
                    else:
                        break

            # 如果剩余容量为零或达到模拟天数，停止模拟
            if F.sum() == 0 or day == self.simulation_days:
                self.simulation_days = day
                stop_day = day
                break

        # 计算总的损失患者数、转诊次数和剩余容量
        TLP = sum(LP_daily_dict.values())
        TR = sum(R_daily_dict.values())
        FC = F.sum()

        return LP_daily_dict, R_daily_dict, FC_daily_dict, TLP, TR, FC, stop_day

    def return_parameters(self):
        total_referrals = self.TR
        LP_day = self.LP_daily_dict[self.stop_day]
        total_patients = self.daily_patient_count * self.stop_day
        if total_patients > 0:
            LP_day_ratio = round(LP_day / total_patients, 2)
            TLP_ratio = round(self.TLP / total_patients, 2)
            TAR = round(total_referrals / total_patients, 2)

        else:
            LP_day_ratio = 0  # 或其他合理的默认值
        if total_patients - LP_day > 0:
            VAR = round(total_referrals / (total_patients - LP_day), 2)
        else:
            VAR = 0
        FC = self.FC_daily_dict[self.stop_day]
        total_FC = self.FC_daily_dict[1] + self.daily_patient_count
        FC_ratio = round(FC / total_FC, 2)
        m = self.mean_visits
        q = self.referral_probability
        delta = self.daily_patient_count
        alpha = self.alpha
        stop_day = self.stop_day

        return TAR, VAR, stop_day, LP_day_ratio, TLP_ratio, FC_ratio, m, delta, q, alpha

if __name__ == '__main__':
    model = HospitalModel(num_doctors=1000, avg_degree=10, weight_mean=1.0, weight_std=0.2,
                          daily_patient_count=750, mean_visits=4, referral_probability=0.15,
                          simulation_days=100, min_visit_limit=20, max_visit_limit=50, alpha=1)
    parameters = model.return_parameters()


# 理论模型（解析模型）