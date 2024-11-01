import numpy as np
import networkx as nx
import random

class HospitalModel:
    def __init__(self, num_doctors, avg_degree, weight_mean, weight_std, daily_patient_count, mean_visits,
                 referral_probability, simulation_days, min_visit_limit, max_visit_limit, alpha):

        self.num_doctors = num_doctors  # 医生数目
        self.avg_degree = avg_degree  # 平均度
        self.weight_mean = weight_mean  # 权重均数
        self.weight_std = weight_std  # 权重方差
        self.mean_visits = mean_visits  # 就诊序列的元素个数
        self.referral_probability = referral_probability  # 完全随机选择概率Q
        self.simulation_days = simulation_days  # 模拟时间
        self.min_visit_limit = min_visit_limit  # 医生就诊量最小值
        self.max_visit_limit = max_visit_limit  # 医生就诊量最大值
        self.daily_patient_count = daily_patient_count
        self.alpha = alpha

        self.G, self.free_capacity = self.generate_hospital_network()  # 医院网络
        self.R_patient, self.LP_patient, self.visited_patients = self.process_patient_visit()
        self.R_daily, self.LP_daily, self.daily_total_visited_patients, self.FC_daily, self.stop_simulation = self.simulate_single_day(daily_patient_count)
        self.R_daily_dict, self.LP_daily_dict, self.daily_visited_patients_dict, self.Np_dict, self.FC_daily_dict, self.stop_day = self.simulate_patient_visits()  # 主要参数

        self.TAR, self.VAR, self.stop_day, self.LP_day_ratio, self.TLP_ratio, self.FC_ratio, self.m, self.delta, self.q, self.alpha = self.return_parameters()

    # 生成医院网络
    def generate_hospital_network(self):
        G = nx.barabasi_albert_graph(self.num_doctors, self.avg_degree)
        G_directed = nx.DiGraph()

        # 复制无向图的边到有向图中
        for i, j in G.edges():
            if random.uniform(0, 1) > 0.5:
                G_directed.add_edges_from([(i, j)])
            else:
                G_directed.add_edges_from([(j, i)])
        G_directed.add_edges_from(G.edges())
        G = G_directed
        free_capacity = 0

        for node in G.nodes:
            G.nodes[node]['visit_limit'] = np.random.randint(self.min_visit_limit, self.max_visit_limit)
            G.nodes[node]['current_visits'] = 0
            G.nodes[node]['free_capacity'] = G.nodes[node]['visit_limit'] - G.nodes[node]['current_visits']
            free_capacity = free_capacity + G.nodes[node]['visit_limit']
        for edge in G.edges:
            G.edges[edge]['weight'] = np.abs(np.random.normal(self.weight_mean, self.weight_std))

        return G, free_capacity

    def choose_next_doctor(self, current_doctor):
        successors = list(self.G.successors(current_doctor))
        if successors:  # 检查是否存在出边指向的节点
            weights = np.array(
                [self.G.edges[current_doctor, successor]['weight'] ** self.alpha for successor in
                 successors])

            if np.sum(weights) > 0:  # 检查存在非零权重的边。
                weights /= np.sum(weights)  # 权重归一化
                next_doctor = np.random.choice(successors, p=weights)  # 使用归一化的权重作为概率分布，随机选择一个出边指向的节点作为下一个医生
            else:
                next_doctor = np.random.choice(list(self.G.nodes))
        else:
            next_doctor = np.random.choice(list(self.G.nodes))

        return next_doctor

    # 模拟每天单个患者的就诊过程
    def process_patient_visit(self):
        current_doctor = np.random.choice(list(self.G.nodes))
        j = 0
        R_patient = 0
        LP_patient = 0
        visited_patients = 0  # 记录得到医生诊治的患者数

        while j <= self.mean_visits:
            if self.G.nodes[current_doctor]['free_capacity'] > 0:
                self.G.nodes[current_doctor]['free_capacity'] -= 1
                visited_patients += 1
                R_patient = j
                break
            # 实际上转诊流程
            else:
                referral_decision = np.random.rand()
                j += 1
                if j >= self.mean_visits + 1:
                    LP_patient += 1
                    R_patient = self.mean_visits
                    break
                else:
                    if referral_decision <= self.referral_probability:
                        current_doctor = np.random.choice(list(self.G.nodes))
                        R_patient += 1
                    else:
                        current_doctor = self.choose_next_doctor(current_doctor)
                        R_patient += 1

        return R_patient, LP_patient, visited_patients

    # 模拟一天的全部新增患者的就诊过程
    def simulate_single_day(self, daily_patient_count):

        R_daily = 0  # 一天转诊数初始为0
        LP_daily = 0  # 一天的损失患者数
        daily_total_visited_patients = 0  # 记录一天内得到医生诊治的患者数

        # 检查是否停止模拟的标志
        stop_simulation = False

        for patient in range(daily_patient_count):
            if stop_simulation:
                break
            # 就诊遍历每天新增患者总数
            R_patient, LP_patient, visited_patients = self.process_patient_visit()
            R_daily += R_patient
            LP_daily += LP_patient
            daily_total_visited_patients += visited_patients
            # 如果空闲容量为零或模拟时间达到100天，停止模拟

        daily_FC_dict = {node: self.G.nodes[node]['free_capacity'] for node in self.G.nodes}
        FC_daily = sum(daily_FC_dict.values())

        return R_daily, LP_daily, daily_total_visited_patients, FC_daily, stop_simulation

    # 模拟整个时间段的患者就诊——时间步迭代
    def simulate_patient_visits(self):
        R_daily_dict = {}
        LP_daily_dict = {}
        daily_visited_patients_dict = {}
        FC_daily_dict = {}
        Np_dict = {}

        stop_day = None  # 初始化 stop_day 变量
        for day in range(1, self.simulation_days + 1):
            if day == 1:
                Np = self.daily_patient_count
                R_daily, LP_daily, daily_total_visited_patients, FC_daily, stop_simulation = self.simulate_single_day(Np)
                R_daily_dict[day] = R_daily  # 每日转诊数
                LP_daily_dict[day] = LP_daily  # 每日损失患者数
                daily_visited_patients_dict[day] = daily_total_visited_patients  # 每日得到就诊患者数
                Np_dict[day] = Np  # 每日新增患者总数=前一天损失患者数+新增患者数
                FC_daily_dict[day] = FC_daily  # 每日空闲容量
                if FC_daily == 0 or day == self.simulation_days:
                    self.simulation_days = day
                    stop_day = day
                    break
            elif day != 1:
                Np = self.daily_patient_count + LP_daily_dict[day - 1]
                R_daily, LP_daily, daily_total_visited_patients, FC_daily, stop_simulation = self.simulate_single_day(Np)
                R_daily_dict[day] = R_daily  # 每日转诊数
                LP_daily_dict[day] = LP_daily  # 每日损失患者数
                daily_visited_patients_dict[day] = daily_total_visited_patients  # 每日得到就诊患者数
                Np_dict[day] = Np  # 每日新增患者总数=前一天损失患者数+新增患者数
                FC_daily_dict[day] = FC_daily  # 每日空闲容量
                if FC_daily == 0 or day == self.simulation_days:
                    self.simulation_days = day
                    stop_day = day
                    break

        return R_daily_dict, LP_daily_dict, daily_visited_patients_dict, Np_dict, FC_daily_dict, stop_day

    def return_parameters(self):
        TR = sum(self.R_daily_dict.values())  # 总转诊数
        TLP = sum(self.LP_daily_dict.values())  # 总损失患者数
        LP_day = self.LP_daily_dict[self.stop_day]

        TP = self.daily_patient_count * self.stop_day  # 总患者数
        TLP_ratio = round(TLP / TP, 2)  # 总损失患者比例
        LP_day_ratio = round(LP_day / TP, 2)  # 实际损失患者比例
        TAR = round(TR / TP, 2)  # 总的平均转诊数
        if TP - TLP > 0:
            VAR = round((TR-TLP*self.mean_visits+1) / (TP - TLP), 2)  # 排除总损失患者数的平均转诊数
        else:
            VAR = 0
        stop_day = self.stop_day
        FC_ratio = round(self.FC_daily_dict[self.stop_day] / self.free_capacity, 2)
        m = self.mean_visits
        q = self.referral_probability
        delta = self.daily_patient_count
        alpha = self.alpha
        return TAR, VAR, stop_day, LP_day_ratio, TLP_ratio, FC_ratio, m, delta, q, alpha

if __name__ == '__main__':
    # 参数设置
    '''num_doctors = 1000  # 医生节点数
    avg_degree = 10  # 平均度
    weight_mean = 1.0  # 权重的均数
    weight_std = 0.2  # 权重的标准差
    mean_visits = 4  # 就诊序列的元素个数的均数
    referral_probability = 0.15  # 概率q（完全随机选择）
    simulation_days = 100  # 模拟天数
    # 医生的就诊上限范围
    min_visit_limit = 20
    max_visit_limit = 50
    alpha = 1  # alpha表示转诊偏好，（偏好权重大/权重小/无偏好）
    '''
    model = HospitalModel(num_doctors=1000, avg_degree=10, weight_mean=1.0, weight_std=0.2,
                          daily_patient_count=750, mean_visits=4, referral_probability=0.15, simulation_days=100,
                          min_visit_limit=20, max_visit_limit=50, alpha=1)
    model.return_parameters()

# 数值模拟（随机模拟） 
