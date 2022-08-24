''' 
When using gdx.open_ble(), note that there are a few options:

gdx.open_ble() - When there are no arguments, the function finds all available Go Direct 
                ble devices, prints the list to the terminal, and prompts the user
                to select the device to connect.

gdx.open_ble("GDX-FOR 071000U9") - Use your device's name as the argument. The function will
                search for a ble device with this name. If found it will connect it. If connecting
                to multiple devices simply separate the names with a comma, such as 
                gdx.open_ble("GDX-FOR 071000U9, GDX-HD 151000C1")

gdx.open_ble("proximity_pairing") - Use "proximity_pairing" as the argument and the function will
                find the ble device with the strongest rssi (signal strength) and connect that
                device.

Below is a simple starter program that uses the gdx functions to collect data from a Go Direct device 
connected via Bluetooth. The gdx.open_ble(), gdx.selct_sensors(), and gdx.start() functions do not have 
arguments and will therefore provide you with a prompt to select your device, sensors, and period.

Tip: You can skip the prompts to select the device, the sensors, and the period by entering arguments
in the functions. For example, if you have a Go Direct Motion with serial number 0B1010H3 and you want 
to sample from sensor 5 at a period of 50ms, you would configure the functions like this:
gdx.open_ble("GDX-MD 0B1010H3"), gdx.select_sensors([5]), gdx.start(50)

'''
from model import TCN
import gdx
import torch
from sklearn.preprocessing import scale
from sklearn import preprocessing

gdx = gdx.gdx()

gdx.open_ble("GDX-RB 0K204167")
gdx.select_sensors([1])
gdx.start(20)
run_time=0
queue = []
model = TCN(1, 1, [32] * 5, 5, 0).double()
model.cuda()
model.load_state_dict(torch.load("breath_Aligned_Array.pt"))

try:
    # for i in range(130):
    while True:
        measurements = gdx.read()
        if measurements == None:
            break
        if run_time < 128:
            queue.append(measurements)
            run_time += 1
        else:
            queue.pop(0)
            queue.append(measurements)
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(queue)
            # print(type(queue))
            numpy_queue = scaler.transform(queue)
            # print(type(queue))
            with torch.no_grad():
                tensor_queue = torch.tensor(numpy_queue).type(torch.DoubleTensor).to(device="cuda")
                output = model(tensor_queue.unsqueeze(0)).squeeze(0)
            two_second_later = output[-1].item()
            real_time = output[-40].item()
            print("two_second_later: ",two_second_later)
            print("real_time: ", real_time)
except Exception as e:
    gdx.stop()
    gdx.close()
    print(str(e))
# print(type(queue)) #list of list
# print(type(tensor_queue)) #tensor
gdx.stop()
gdx.close()