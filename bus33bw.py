import pandas as pd
import pandapower as pp
from pandapower import networks
import matplotlib.pyplot as plt
import logging
import numpy as np
import os

# 设置日志配置
logging.basicConfig(filename='simulation.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levellevel=%(message)s')

# 读取 CSV 文件并转换数据类型
def read_csv(file_name):
    try:
        df = pd.read_csv(file_name).apply(pd.to_numeric, errors='coerce')
        logging.info(f'Successfully read {file_name}')
        return df
    except Exception as e:
        logging.error(f'Error reading {file_name}: {e}')
        return pd.DataFrame()

load_active = read_csv('load_active_1hour copy.csv')
load_reactive = read_csv('load_reactive_1hour copy.csv')
pv_active = read_csv('pv_active_1hour copy.csv')


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
net.bus.loc[1:, 'max_vm_pu'] = 1.05
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
num_simulations = len(load_active) // 24  # 计算每小时的仿真次数
vm_pu_hourly = np.zeros((24, len(net.bus)))  # 用于存储每个小时每个节点的电压平均值

def run_hourly_simulation(net, load_active, load_reactive, pv_active, hour):
    for i in range(len(net.load)):
        if i < 32:  # 只设置前32个负荷节点，第33个节点保持默认值
            net.load.at[i, 'p_mw'] = load_active.iloc[hour, i]
            net.load.at[i, 'q_mvar'] = load_reactive.iloc[hour, i]
    
    # 设置每个光伏发电机的有功功率，确保与PV_bus_index对应
    for i in range(len(PV_bus_index)):
        net.sgen.at[net.sgen[net.sgen['bus'] == PV_bus_index[i]].index[0], 'p_mw'] = pv_active.iloc[hour, i]

    net.bus['vm_pu'] = 1.0  # 重置节点电压

    try:
        pp.runpp(net, numba=False, max_iteration=30)  # 增加最大迭代次数
        logging.debug(f'Hour {hour}: Voltage magnitudes: {net.res_bus.vm_pu.values}')
        return net.res_bus.vm_pu.values
    except pp.LoadflowNotConverged:
        logging.error(f"Power flow did not converge at hour {hour}")
        return [float('nan')] * len(net.bus)  # 使用 NaN 表示未收敛的情况
    except Exception as e:
        logging.error(f"An error occurred at hour {hour}: {e}")
        return [float('nan')] * len(net.bus)  # 使用 NaN 表示出现其他错误的情况

# 进行仿真并记录每小时的电压结果
for hour in range(24):
    logging.info(f'Running simulations for hour {hour}')
    hourly_voltages = []
    for sim in range(num_simulations):
        index = hour + sim * 24
        voltage_results = run_hourly_simulation(net, load_active, load_reactive, pv_active, index)
        hourly_voltages.append(voltage_results)
    
    # 检查 hourly_voltages 是否为空
    if len(hourly_voltages) > 0:
        hourly_voltages = np.array(hourly_voltages)
        vm_pu_hourly[hour, :] = np.nanmean(hourly_voltages, axis=0)  # 取每小时的电压平均值
    else:
        logging.warning(f"No voltage results for hour {hour}")

# 打印调试信息到日志文件中
logging.debug(f'vm_pu_hourly:\n{vm_pu_hourly}')

# 可视化电压结果
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(24), range(len(net.bus)))
ax.plot_surface(X, Y, vm_pu_hourly.T, cmap='viridis')

ax.set_xlabel('Hour')
ax.set_ylabel('Bus Number')
ax.set_zlabel('Voltage Magnitude (p.u.)')
ax.set_title('Voltage Magnitude over 24 Hours')
plt.savefig('voltage_magnitude_over_24_hours_3d.png')  # 保存图形为文件
plt.show()
