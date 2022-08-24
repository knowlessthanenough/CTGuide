import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import  ConnectionPatch
import matplotlib.ticker as mticker

def draw_one_only(ground_truth:list):
    time_step = range(len(ground_truth))
    fig = plt.figure(figsize=(80,10))
    ax = fig.add_subplot(111)
    ax.plot(time_step, ground_truth, '-r',linewidth=0.5, label='ground truth')
    ax.set_xlabel("times step (40ms/25Hz)")
    ax.set_ylabel("lung size (L)")
    ax.legend(loc='upper left')
    ax.set_title('GT')
    plt.savefig("GT.png")
    # plt.savefig("GT.png",dpi=1000) #png 比 jpeg 更好，因為 png 是一種無失真壓縮格式，而另一種是有失真壓縮格式。dpi (解析度 default:100)
    fig.show()

def draw_compare_result(prediction :list, ground_truth :list, stds:list = None ,zoom:list = None):
    time_step = range(len(prediction))
    ground_truth = ground_truth[(len(ground_truth)-len(prediction)):]
    fig, ax = plt.subplots(1, 1, figsize=(80, 10))
    ax.plot(time_step, ground_truth, '-r', linewidth=0.7, label='ground truth')
    ax.plot(time_step, prediction, '-b', linewidth=0.5, label='prediction')
    if stds is not None:
        ax.fill_between(range(len(prediction)),(prediction-stds),(prediction+stds),alpha=.1)
    if zoom is not None:
        axins = ax.inset_axes((0.2, 0.6, 0.2, 0.3))
        axins.plot(time_step, ground_truth, '-r', linewidth=0.7)
        axins.plot(time_step, prediction, '-b', linewidth=0.5)
        zone_and_linked(ax, axins, zoom[0], zoom[1], time_step, [ground_truth,prediction], 'right')
    ax.set_xlabel("times step (40ms/25Hz)")
    ax.set_ylabel("lung size (L)")
    ax.legend(loc='upper left')
    ax.set_title('accruary')
    plt.savefig("accruary.png")
    plt.show()

# prediction = np.array([3, 5, 1, 8, 4, 6])
# GT = np.array([2, 6, 1, 7, 4, 8])
# stds = np.array([1.3, 2.6, 0.78, 3.01, 2.32, 2.9])
# draw_result(prediction,GT,stds)
# *
# zone_and_linked(ax,axins,800,825,time_step,data,'right')
def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)



if __name__ == "__main__":
    df = pd.read_csv("data\\val\\Belt_1.csv", header = None, sep=' ')
    input = list(df.iloc[:, 2])
    GT = list(df.iloc[:, 3])
    draw_compare_result(input,GT) #blue, red




    # time_step = range(len(data))
    # fig, ax = plt.subplots(1, 1, figsize=(80, 10))
    # ax.plot(time_step, data, '-r', linewidth=0.7, label='GT')
    # ax.legend(loc='upper left')
    # axins = ax.inset_axes((0.2, 0.6, 0.2, 0.3))
    # axins.plot(time_step, data, color='#f0bc94')
    # zone_and_linked(ax,axins,800,900,time_step,[data],'right')
    # plt.show()