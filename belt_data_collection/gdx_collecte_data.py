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

from gdx import gdx
from datetime import datetime,timedelta
gdx = gdx.gdx()
gdx.open_ble("GDX-RB 0K204167")
gdx.select_sensors([1])
gdx.start(20)

now = str(datetime.now())
f= open( "data.txt","w")

now = datetime.now()
base = gdx.read()[0]
try:
    while True:
        measurements = str(gdx.read()[0] - base)
        time_as_str = now.strftime('%Y/%m/%d %H:%M:%S.%f')[:-4]
        now = now + timedelta(seconds=0.05)
        data = (time_as_str + ' ' + measurements + "\n")
        f.write(data)

        if measurements == None:
            break
except KeyboardInterrupt:
    print ('KeyboardInterrupt exception is caught')

f.close()
gdx.stop()
gdx.close()