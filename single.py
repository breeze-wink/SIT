# 导入必要的模块
import pandapower as pp
from pandapower import networks, plotting

print(f'pandapower 版本: {pp.__version__}')

# 加载默认网络
net = networks.case33bw()
print(f'net:\n{net}')

# 修改外部电网参数，设置有功和无功功率的最大和最小值
net.ext_grid['max_p_mw'] = 100
net.ext_grid['min_p_mw'] = -100
net.ext_grid['max_q_mvar'] = 100
net.ext_grid['min_q_mvar'] = -100

# 移除特定线路以消除环路
net.line.drop([32, 33, 34, 35, 36], inplace=True)

# 定义新的 PV 在 bus33bw 系统中
PV_bus_index = [12, 17, 21, 24, 28, 32]
for i in range(len(PV_bus_index)):
    pp.create_sgen(net, bus=PV_bus_index[i], p_mw=0.5, q_mvar=0.05, name='undefined',
                   type='PV', scaling=1, controllable=False, in_service=True)

# 修改节点电压约束
net.bus.loc[1:, 'max_vm_pu'] = 1.05
net.bus.loc[1:, 'min_vm_pu'] = 0.95

# 将节点分配到不同的控制区域
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

# 对网络组件按索引排序
net.bus.sort_index(inplace=True)
net.line.sort_index(inplace=True)
net.load.sort_index(inplace=True)

# 根据节点区域重命名静态发电机
for i in range(net.sgen.shape[0]):
    sgen_bus_index = net.sgen['bus'].iloc[i]
    net.sgen.loc[i, 'name'] = net.bus.loc[sgen_bus_index, 'zone']
    


net.sgen.sort_index(inplace=True)

# 设置所有节点类型为 'n'
net.bus['type'] = 'n'

# 运行潮流计算
pp.runpp(net)

# 打印潮流计算结果
print(f'节点电压:\n{net.res_bus}')
print(f'线路功率:\n{net.res_line}')
print(f'发电机出力:\n{net.res_gen}')
print(f'静态发电机出力:\n{net.res_sgen}')
print(f'外部电网出力:\n{net.res_ext_grid}')

# 保存网络到文件
pp.to_pickle(net, "bus33bw.p")
plotting.to_html(net, filename='bus33bw.html', show_tables=True)
plotting.plotly.simple_plotly(net)
