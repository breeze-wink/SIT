import pandas as pd
import pandapower as pp
from pandapower import networks
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件并转换数据类型

load_active = pd.read_csv('load_active_1hour copy.csv', delimiter=',')
load_reactive = pd.read_csv('load_reactive_1hour copy.csv', delimiter=',')
pv_active = pd.read_csv('pv_active_1hour copy.csv',delimiter=',')


# 加载默认网络
net = networks.case33bw()
net.ext_grid['max_p_mw'] = 100
net.ext_grid['min_p_mw'] = -100
net.ext_grid['max_q_mvar'] = 100
net.ext_grid['min_q_mvar'] = -100

# 添加静态发电机（光伏系统）
PV_bus_index = [12, 17, 21, 24, 28, 32]
for i in range(len(PV_bus_index)):
    pp.create_sgen(net, bus=PV_bus_index[i], p_mw=0.5, q_mvar=0.05, name=f'PV_{i}', type='PV', scaling=1, controllable=True, in_service=True)

# 修改节点电压限制
net.bus.loc[1:, 'max_vm_pu'] = 1.20
net.bus.loc[1:, 'min_vm_pu'] = 0.95


# 定义控制区
main_bus_index = [0, 1, 2, 3, 4, 5]
zone1_index = list(range(6, 18))
zone2_index = list(range(18, 22))
zone3_index = list(range(22, 25))
zone4_index = list(range(25, 33))

net.bus.loc[main_bus_index, 'zone'] = 'main'
net.bus.loc[zone1_index, 'zone'] = 'zone1'
net.bus.loc[zone2_index, 'zone'] = 'zone2'
net.bus.loc[zone3_index, 'zone'] = 'zone3'
net.bus.loc[zone4_index, 'zone'] = 'zone4'

# 排序
net.bus.sort_index(inplace=True)
net.line.sort_index(inplace=True)
net.load.sort_index(inplace=True)
net.sgen.sort_index(inplace=True)

for i in range(net.sgen.shape[0]):
    sgen_bus_index = net.sgen['bus'].iloc[i]
    net.sgen.loc[i, 'name'] = net.bus.loc[sgen_bus_index, 'zone']

# 初始电压设置
net.bus['vm_pu'] = 1.0

# 运行 24 小时的仿真并保存结果
num_simulations = (len(pv_active) - 1) // 24  # 计算每小时的仿真次数


vm_pu_hourly = np.zeros((24, 33))  # 用于存储每个小时每个节点的电压平均值

def run_hourly_simulation(net, load_active, load_reactive, pv_active, timeHour):
    for i in range(len(net.load)):
        net.load.at[i, 'p_mw'] = load_active.iloc[timeHour, i + 1]
        net.load.at[i, 'q_mvar'] = load_reactive.iloc[timeHour, i + 1]
    
    # 设置每个光伏发电机的有功功率，确保与PV_bus_index对应
    for i in range(len(PV_bus_index)):
        net.sgen.at[net.sgen[net.sgen['bus'] == PV_bus_index[i]].index[0], 'p_mw'] = pv_active.iloc[timeHour, i + 1] * 3
    # net.bus['vm_pu'] = 1.0  # 重置节点电压

    pp.runpp(net, numba=False, max_iteration=30)  # 增加最大迭代次数

    return net.res_bus.vm_pu.values

# 进行仿真并记录每小时的电压结果
for hour in range(24):

    hourly_voltages = []
    for sim in range(num_simulations):
        index = hour + sim * 24
        voltage_results = run_hourly_simulation(net, load_active, load_reactive, pv_active, index)
        hourly_voltages.append(voltage_results)
    

    hourly_voltages = np.array(hourly_voltages)
    vm_pu_hourly[hour, :] = np.nanmean(hourly_voltages, axis=0) 

vm_pu_hourly_T = vm_pu_hourly.T

vm_pu_df = pd.DataFrame(vm_pu_hourly_T, index=[f'Bus {j}' for j in range(33)], columns=[f'{i}:00' for i in range(24)])

# 保存 DataFrame 到 Excel 文件
vm_pu_df.to_excel('statistic.xlsx', index_label='Bus')


fig, ax = plt.subplots(figsize=(12, 8))

# 绘制每个总线的电压曲线
for i in range(33):
    ax.plot(range(24), vm_pu_hourly_T[i], label=f'Bus {i}')

# 添加图例
ax.legend(loc='upper right', fontsize='small', ncol=3)

# 添加标题和标签
ax.set_title('Voltage Magnitude over 24 Hours for Each Bus')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Voltage Magnitude (p.u.)')

# 设置x轴的刻度和标签
ax.set_xticks(range(24))
ax.set_xticklabels([f'{i}:00' for i in range(24)])

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()

plt.savefig('voltage_magnitude.png')