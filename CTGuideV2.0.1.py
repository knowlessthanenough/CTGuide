import math
from imu_ui import Ui_mainWindow
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import string
from os import listdir
import pydicom as dicom
import tkinter as tk
from tkinter import filedialog
from natsort import natsorted
from hipnuc_module import *
log_file = 'chlog.csv'
import shutil
from multiprocessing import Queue
from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger
import os
from multiprocessing import Process
import sys
from ip_port_ui import Sever
from model import TCN
import gdx
import torch
from statistics import mean
from sklearn.preprocessing import scale
from sklearn import preprocessing
from PyQt5 import QtCore, QtGui
debug_logger()
gdx = gdx.gdx()

class DataThread(QThread):
    Data_trigger = pyqtSignal(object)
    def __init__(self):
        super(DataThread, self).__init__()
        self.Running = True
        self.Calibrate = False
        self._mutex = QMutex()

    def run(self):
        self.m_IMU = hipnuc_module('./config.json')
        while True:
            try:
                data = self.m_IMU.get_module_data(10) #wait for next moment imu data (60hz)
                Roll = data['euler'][0]['Roll']  # float
                Pitch = data['euler'][0]['Pitch'] #float
                Yaw = data['euler'][0]['Yaw'] #float
            except Exception as e:
                print(str(e))
                pass
            self.IMU_Roll=Roll
            self.IMU_Pitch=Pitch
            self.IMU_Yaw = Yaw
            if self.Running==True:
                self.Euler_Angle=[self.IMU_Roll,self.IMU_Pitch,self.IMU_Yaw]
                self.Data_trigger.emit(self.Euler_Angle)
            else:
                pass

    def running(self):
        try:
            self._mutex.lock()
            return self.Running
        finally:
            self._mutex.unlock()

class BeltThread(QThread):
    Belt_trigger = pyqtSignal(object)
    def __init__(self):
        super(BeltThread, self).__init__()
        self.belt_connect = True
        self._mutex = QMutex()

    def run(self):
        gdx.open_ble("GDX-RB 0K204167")
        gdx.select_sensors([1])
        gdx.start(20)
        zero = gdx.read()
        # run_time = 0
        # queue = []
        # model = TCN(1, 1, [32] * 5, 5, 0).double()
        # model.cuda()
        # model.load_state_dict(torch.load("breath_Aligned_Array.pt"))
        try:
            while True:
                measurements = [gdx.read()[0] - zero[0]] # list
                if measurements == None:
                    break
                # if run_time < 128:
                #     queue.append(measurements)
                #     run_time += 1
                else:
                #     queue.pop(0)
                #     queue.append(measurements)
                #     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(queue)
                #     numpy_queue = scaler.transform(queue)
                #     with torch.no_grad():
                #         tensor_queue = torch.tensor(numpy_queue).type(torch.DoubleTensor).to(device="cuda")
                #         output = model(tensor_queue.unsqueeze(0)).squeeze(0)
                #     two_second_later = output[-1].item()
                #     real_time = output[-40].item()
                #     model_output = two_second_later, real_time
                    real_time = measurements[0]
                    two_second_later = real_time + 3
                    model_output = two_second_later, real_time
                    if self.belt_connect == True:
                #         self.Belt_trigger.emit(model_output)
                        self.Belt_trigger.emit(model_output)
        except Exception as e:
            gdx.stop()
            gdx.close()
            # self.belt_connect = False
            print(str(e))

    def Belt_connected(self):
        try:
            self._mutex.lock()
            return self.belt_connect
        finally:
            self._mutex.unlock()

class MainWindow(QMainWindow,Ui_mainWindow):
    imagePath = []

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        qApp.installEventFilter(self)
        self.setupUi(self)
        self._data_thread = DataThread()    # 实例化线程对象
        self._data_thread.Data_trigger.connect(self.updateIMUdata)
        self._belt_thread = BeltThread()  # 实例化线程对象
        self._belt_thread.Belt_trigger.connect(self.Lung_size_data_receiver)
        self.Btn_Start.clicked.connect(self.IMU_Start2Stop) #连接IMU
        self.Btn_Start_Belt.clicked.connect(self.Belt_Start2Stop) #连接Belt
        self.btn_htof.clicked.connect(self.htof)
        self.Btn_Calibration.clicked.connect(self.Calibration)
        self.Btn_loadCTYZ.clicked.connect(self.LoadCTYZ)
        self.Btn_nextYZimage.clicked.connect(self.NextCTYZ)
        self.Btn_lastYZimage.clicked.connect(self.LastCTYZ)
        self.Btn_nextXZimage.clicked.connect(self.NextCTXZ)
        self.Btn_lastXZimage.clicked.connect(self.LastCTXZ)
        self.btn_reset.clicked.connect(self.Reset)
        self.btn_StoP.clicked.connect(self.StoP)
        self.btn_save_ct_lung_volume.clicked.connect(self.active_save_ct_lung_volume)
        self.CT_XZ.mousePressEvent = self.CT_XZMousePress
        self.CT_XZ.mouseMoveEvent = self.CT_XZMouseMove
        self.CT_XZ.mouseReleaseEvent = self.CT_XZMouseRelease
        self.CT_YZ.mousePressEvent = self.CT_YZMousePress
        self.CT_YZ.mouseMoveEvent = self.CT_YZMouseMove
        self.CT_YZ.mouseReleaseEvent = self.CT_YZMouseRelease
        self.CTAngle = [0,0,0] #until calibra this will replace with CT table angle
        self.adjustwindow = False
        self.reflex = False
        self.supine = True
        self.cv_image_yz_o = None
        self.cv_image_xz_o = None
        self.cv_image_yz = None
        self.cv_image_xz = None
        self.newPoint_yz = None
        self.newPoint_xz = None
        self.qtwidth_YZ = self.CT_YZ.width()
        self.qthight_YZ = self.CT_YZ.height()
        self.qtx = self.CT_XZ.geometry().x()
        self.qty = self.CT_XZ.geometry().y()
        self.qtwidth_XZ = self.CT_XZ.width()
        self.qthight_XZ = self.CT_XZ.height()
        self.clicked_time = 0
        self.lung_size_past = 0
        self.lung_size_hold_breath = []
        self.ct_lung_volume = None
        self.volume_data_transmit_to_CT_save_function = False

    def CT_YZMousePress(self, event):
        if self.cv_image_yz_o is not None:
            if event.button() == Qt.LeftButton:
                self.clicked_time += 1
                if self.clicked_time == 1:
                    self.lastPoint_yz = (event.pos().x(),event.pos().y()) # first time click get the insert point mouse position
                    self.newPoint_yz = None
                    self.orginal_image_last_point_location = self.point_location_translate(self.lastPoint_yz,self.cv_image_yz_o_1c.shape,[self.qthight_YZ,self.qtwidth_YZ])
                    self.orginal_image_last_point_location.append(self.no_of_slices_yz) #make it become [x,y,z] it is the location in numpy array
                    # --------------------------------------
                    self.lastPoint_xz = [self.orginal_image_last_point_location[2],self.orginal_image_last_point_location[1]]
                    self.lastPoint_xz = self.point_location_translate(self.lastPoint_xz, [int(self.img3d.shape[1]), int(self.img3d.shape[2] / self.sag_aspect)], [self.img3d.shape[1], self.img3d.shape[2]])  # location after resize image to 1:1
                    self.lastPoint_xz = [self.lastPoint_xz[1], -self.lastPoint_xz[0] + self.cv_image_xz_o_1c.shape[0] - 1]  # location after rotation
                    self.lastPoint_xz = self.point_location_translate(self.lastPoint_xz,[self.qthight_XZ, self.qtwidth_XZ],self.cv_image_yz_o_1c.shape)
                    # np_slices = self.img3d[:, :, self.no_of_slices_yz] #check checked point in orginal image location
                    # np_slices = self.rescaleCT(self.windowWidth, self.windowLabel, np_slices)
                    # np_slices = (np.stack((np_slices, np_slices, np_slices)).transpose(1,2,0)).copy()
                    # cv2.circle(np_slices, [self.orginal_image_last_point_location[0],self.orginal_image_last_point_location[1]], 1, (0, 0, 255), thickness=-1)
                    # cv2.imshow('',np_slices)
                if self.clicked_time == 2:
                    if self.cv_image_yz_o is not None:
                        self.newPoint_yz = (event.pos().x(), event.pos().y())
                        self.orginal_image_new_point_location = self.point_location_translate(self.newPoint_yz,self.cv_image_yz_o_1c.shape,[self.qthight_YZ,self.qtwidth_YZ])
                        self.orginal_image_new_point_location.append(self.no_of_slices_yz)
                        # check checked point in orginal image location
                        # np_slices = self.img3d[:, :, self.no_of_slices_yz]
                        # np_slices = self.rescaleCT(self.windowWidth, self.windowLabel, np_slices)
                        # np_slices = (np.stack((np_slices, np_slices, np_slices)).transpose(1,2,0)).copy()
                        # cv2.circle(np_slices, self.orginal_image_new_point_location, 1, (0, 0, 255), thickness=-1)
                        # cv2.imshow('',np_slices)
                        # ---------------------
                        # xz_np_newpoint = [self.orginal_image_new_point_location[2], self.orginal_image_new_point_location[1]]
                        # xz_newpoint = self.point_location_translate(xz_np_newpoint,[int(self.img3d.shape[1]),int(self.img3d.shape[2] / self.sag_aspect)],[self.img3d.shape[1],self.img3d.shape[2]])
                        # np_slices = self.img3d[:, self.orginal_image_new_point_location[0], :]
                        # np_slices = self.rescaleCT(self.windowWidth, self.windowLabel, np_slices)
                        # np_slices = cv2.resize(np_slices, (int(self.img3d.shape[2] / self.sag_aspect), int(self.img3d.shape[1])),interpolation=cv2.INTER_CUBIC)
                        # np_slices = (np.stack((np_slices, np_slices, np_slices)).transpose(1, 2, 0)).copy()
                        # cv2.circle(np_slices, xz_newpoint, 10, (255, 0, 0), thickness=-1)
                        # np_slices = cv2.rotate(np_slices, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                        # after_rot_xz_newpoint = [xz_newpoint[1],-xz_newpoint[0] + np_slices.shape[0] -1] # real point
                        # print(after_rot_xz_newpoint)
                        # cv2.circle(np_slices, after_rot_xz_newpoint, 9, (0, 0, 255), thickness=-1)
                        # cv2.imshow('', np_slices)

                        self.newPoint_xz = [self.orginal_image_new_point_location[2],self.orginal_image_new_point_location[1]]
                        self.newPoint_xz = self.point_location_translate(self.newPoint_xz, [int(self.img3d.shape[1]), int(self.img3d.shape[2] / self.sag_aspect)], [self.img3d.shape[1], self.img3d.shape[2]]) #location after resize image to 1:1
                        self.newPoint_xz = [self.newPoint_xz[1], -self.newPoint_xz[0] + self.cv_image_xz_o_1c.shape[0] - 1]  # location after rotation
                        self.newPoint_xz = self.point_location_translate(self.newPoint_xz,[self.qthight_XZ,self.qtwidth_XZ],self.cv_image_yz_o_1c.shape)
                        self.no_of_slices_xz = self.orginal_image_new_point_location[0]
                        self.cut_sagital_view()
                        self.rewrite_and_copy_cv_image_xz_o()
                        self.try_draw_line(self.cv_image_xz, self.lastPoint_xz, self.newPoint_xz)
                        self.set_image_to_qlabel_xz(self.cv_image_xz)
                        self.imageXZNum.setText(str(self.no_of_slices_xz + 1))
                        dx = self.newPoint_xz[0] - self.lastPoint_xz[0]
                        dy = self.newPoint_xz[1] - self.lastPoint_xz[1]
                        if dy == 0:
                            self.target_xz_angle = 90
                        else:
                            self.target_xz_angle = math.degrees(math.atan(dx / dy))
                        self.XZ_target_angle.setText(str(round((90-abs(self.target_xz_angle)),1)))
                        # ---------------------------------------------------------------
                        self.clicked_time = 0 #reset clicked time
                        self.cv_image_yz = self.cv_image_yz_o.copy()
                        self.try_draw_line(self.cv_image_yz,self.lastPoint_yz,self.newPoint_yz)
                        self.markLR(self.cv_image_yz)
                        self.set_image_to_qlabel_yz(self.cv_image_yz)
                        dx = self.newPoint_yz[0] - self.lastPoint_yz[0]
                        dy = self.newPoint_yz[1] - self.lastPoint_yz[1]
                        self.target_yz_project_length = cv2.norm(self.newPoint_yz,
                                                                 self.lastPoint_yz) * 512 / self.qtwidth_YZ * self.CT_to_real_world_ratio[0]/ 10
                        self.target_length_yz.setText(str(round(self.target_yz_project_length, 1)))
                        if dx == 0:
                            self.target_yz_angle = 90
                        else:
                            self.target_yz_angle = math.degrees(math.atan(dy / dx))
                        self.YZ_target_angle.setText(str(round((90 - abs(self.target_yz_angle)), 1)))
                        reallength = str(self.real_target_length(self.orginal_image_last_point_location,self.orginal_image_new_point_location,self.CT_to_real_world_ratio[0],self.CT_to_real_world_ratio[1],self.ss))
                        self.target_length.setText(reallength)
                    else:
                        pass
            elif event.button() == Qt.RightButton:
                self.adjustwindow = True
                self.last_adjust_Point_yz = (event.pos().x(),event.pos().y())  # mouse position
        else:
            pass

    def CT_XZMousePress(self, event):
        if self.cv_image_xz_o is not None:
            if event.button() == Qt.RightButton:
                self.adjustwindow = True
                self.last_adjust_Point_xz = (event.pos().x(),event.pos().y())  # mouse position
        else:
            pass

    def CT_YZMouseMove(self, event):
        if self.cv_image_yz_o is not None:
            if self.adjustwindow == True:
                self.new_adjust_Point_yz = (event.pos().x(), event.pos().y())
                self.windowLabel,self.windowWidth = self.adjustWindow(self.new_adjust_Point_yz,self.last_adjust_Point_yz,self.windowLabel,self.windowWidth)
                #----------------------------------------------------------commont
                self.rewrite_and_copy_cv_image_xz_o()
                self.rewrite_and_copy_cv_image_yz_o()
                #----------------------------------------------
                self.last_adjust_Point_yz = self.new_adjust_Point_yz
                self.try_draw_line(self.cv_image_xz,self.lastPoint_xz,self.newPoint_xz)
                self.try_draw_line(self.cv_image_yz,self.lastPoint_yz,self.newPoint_yz)
                self.set_image_to_qlabel_xz(self.cv_image_xz)
                self.set_image_to_qlabel_yz(self.cv_image_yz)
        else:
            pass

    def CT_XZMouseMove(self, event):
        if self.cv_image_xz_o is not None:
            if self.adjustwindow == True:
                self.new_adjust_Point_xz = (event.pos().x(), event.pos().y())
                self.windowLabel,self.windowWidth = self.adjustWindow(self.new_adjust_Point_xz, self.last_adjust_Point_xz,self.windowLabel,self.windowWidth )
                #----------------------------------------------comment
                self.rewrite_and_copy_cv_image_xz_o()
                self.rewrite_and_copy_cv_image_yz_o()
                #-------------------------------------------------
                self.last_adjust_Point_xz = self.new_adjust_Point_xz
                self.try_draw_line(self.cv_image_xz,self.lastPoint_xz,self.newPoint_xz)
                self.try_draw_line(self.cv_image_yz,self.lastPoint_yz,self.newPoint_yz)
                self.set_image_to_qlabel_yz(self.cv_image_yz)
                self.set_image_to_qlabel_xz(self.cv_image_xz)
        else:
            pass

    def CT_YZMouseRelease(self,event):
        if event.button() == Qt.RightButton:
            self.adjustwindow = False

    def CT_XZMouseRelease(self, event):
        if event.button() == Qt.RightButton:
            self.adjustwindow = False

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress ):
            key = event.key()
            if key == Qt.Key_Up:
                self.NextCTYZ()
                return True
            if key == Qt.Key_Down:
                self.LastCTYZ()
                return True
            if key == Qt.Key_Left:
                self.LastCTXZ()
                return True
            if key == Qt.Key_Right:
                self.NextCTXZ()
                return True
        return super(MainWindow, self).eventFilter(obj, event)

    def LoadCTYZ(self):
        self.target_length.setText(" ")
        self.target_length_yz.setText(" ")
        self.YZ_target_angle.setText(" ")
        self.XZ_target_angle.setText(" ")
        self.target_yz_project_length = None
        self.target_xz_angle = None
        self.target_yz_angle = None
        self.lastPoint_xz = None
        self.newPoint_xz = None
        self.lastPoint_yz = None
        self.newPoint_yz = None
        if (self.Btn_Start.text() == "Stop"):
            self.IMU_Start2Stop()
        else:
            pass
        try:
            root = tk.Tk()
            root.withdraw()
            imagePath = filedialog.askopenfilename()
            folderPath = os.path.abspath(os.path.join(imagePath, os.pardir)) #parent directory
            ds = dicom.dcmread(imagePath)
            # create 3d model and slice in to sagital_view
            slices = [dicom.dcmread(folderPath + '/' + s) for s in natsorted(listdir(folderPath))]
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            # shutil.rmtree(folderPath, ignore_errors=True) #delete folder after use

            positionlist = []
            for i in range(len(slices)):
                positionlist.append(slices[i].ImagePositionPatient[2])
            # print(positionlist)
            self.total_no_of_slices_yz = len(positionlist)-1 # this begin form 1, other begin form 0 so need to -1
            self.total_no_of_slices_xz = ds.pixel_array.shape[1] - 1
            # print(self.total_no_of_slices_yz,self.total_no_of_slices_xz)
            self.no_of_slices_xz = self.total_no_of_slices_xz // 2
            self.no_of_slices_yz = positionlist.index(ds.ImagePositionPatient[2])
            self.no_of_slices_yz_text = self.total_no_of_slices_yz-self.no_of_slices_yz
            self.CT_to_real_world_ratio = ds.PixelSpacing

            slice_thickness = np.abs((slices[0].SliceLocation - slices[self.total_no_of_slices_yz].SliceLocation)/self.total_no_of_slices_yz)
            slice_thickness = round(slice_thickness, 2)
            self.ss = slice_thickness
            self.ax_aspect = self.CT_to_real_world_ratio[1] / self.CT_to_real_world_ratio[0]
            self.sag_aspect = self.CT_to_real_world_ratio[1] / self.ss
            # print(self.ax_aspect)
            # print(self.sag_aspect)
            img_shape = list(slices[0].pixel_array.shape)
            img_shape.append(len(slices))
            self.img3d = np.zeros(img_shape)
            for i, s in enumerate(slices):
                img2d = s.pixel_array
                self.img3d[:, :, i] = img2d
            # print(self.img3d.shape)
            self.cut_sagital_view()
            self.cut_axial_view()
            self.windowLabel = 700  # window centre
            self.windowWidth = 1400  # window width total of side
            #-----------------------------------------------------comment
            self.rewrite_and_copy_cv_image_xz_o()
            self.rewrite_and_copy_cv_image_yz_o()
            #------------------------------------------------------------
            self.CT_XZ.setVisible(True)
            self.CT_YZ.setVisible(True)
            if self.cv_image_xz.shape[0] < self.cv_image_xz.shape[1]:
                self.CT_XZ.setGeometry(self.qtx, self.qty + int((self.qthight_XZ - int(self.cv_image_xz.shape[0] * self.qtwidth_XZ / self.cv_image_xz.shape[1])) / 2), self.qtwidth_XZ, int(self.cv_image_xz.shape[0] * self.qtwidth_XZ / self.cv_image_xz.shape[1]))
            elif self.cv_image_xz.shape[0] > self.cv_image_xz.shape[1]:
                self.CT_XZ.setGeometry(self.qtx + int((self.qtwidth_XZ - int(self.cv_image_xz.shape[1] * self.qthight_XZ / self.cv_image_xz.shape[0])) / 2), self.qty , int(self.cv_image_xz.shape[1] * self.qthight_XZ / self.cv_image_xz.shape[0]),self.qthight_XZ)
            else:
                self.CT_XZ.setGeometry(self.qtx, self.qty, self.qtwidth_XZ, self.qthight_XZ)
            self.set_image_to_qlabel_xz(self.cv_image_xz)
            self.set_image_to_qlabel_yz(self.cv_image_yz)
            self.CT_ratio.setText(str(round(self.CT_to_real_world_ratio[0],4)))
            self.imageXZNum.setText(str(self.no_of_slices_xz + 1))
            self.imageYZNum.setText(str(self.no_of_slices_yz_text + 1))
        except Exception as e:
            if self.cv_image_yz_o is None:
                print('Loadimage error:' + str(e))
                pass
            else:
                self.rewrite_and_copy_cv_image_xz_o()
                self.rewrite_and_copy_cv_image_yz_o()
                self.set_image_to_qlabel_xz(self.cv_image_xz)
                self.set_image_to_qlabel_yz(self.cv_image_yz)
                print('Loadimage error:' + str(e))
                pass

    def NextCTYZ(self):
        if self.cv_image_yz_o is not None:
            if self.no_of_slices_yz > 0:
                self.no_of_slices_yz -= 1
                self.no_of_slices_yz_text += 1
                self.cut_axial_view()
                #------------------------------------------comment y
                self.rewrite_and_copy_cv_image_yz_o()
                #------------------------------------------
                self.try_draw_line(self.cv_image_yz,self.lastPoint_yz,self.newPoint_yz)
                self.set_image_to_qlabel_yz(self.cv_image_yz)
                self.imageYZNum.setText(str(self.no_of_slices_yz_text + 1))
            else:
                pass
        else:
            pass

    def LastCTYZ(self):
        if self.cv_image_yz_o is not None:
            if self.no_of_slices_yz < self.total_no_of_slices_yz:
                self.no_of_slices_yz += 1
                self.no_of_slices_yz_text -= 1
                self.cut_axial_view()
                # ------------------------------------------comment y
                self.rewrite_and_copy_cv_image_yz_o()
                # ------------------------------------------
                self.try_draw_line(self.cv_image_yz,self.lastPoint_yz,self.newPoint_yz)
                self.markLR(self.cv_image_yz)
                self.set_image_to_qlabel_yz(self.cv_image_yz)
                self.imageYZNum.setText(str(self.no_of_slices_yz_text + 1))
            else:
                pass
        else:
            pass

    def NextCTXZ(self):
        if self.cv_image_xz_o is not None:
            if self.no_of_slices_xz < self.total_no_of_slices_xz:
                self.no_of_slices_xz += 1
                self.cut_sagital_view()
                #--------------------------------------------------comment x
                self.rewrite_and_copy_cv_image_xz_o()
                #--------------------------------------------
                self.try_draw_line(self.cv_image_xz,self.lastPoint_xz,self.newPoint_xz)
                self.set_image_to_qlabel_xz(self.cv_image_xz)
                self.imageXZNum.setText(str(self.no_of_slices_xz + 1))
            else:
                pass
        else:
            pass

    def LastCTXZ(self):
        if self.cv_image_xz_o is not None:
            if self.no_of_slices_xz > 0:
                self.no_of_slices_xz -= 1
                self.cut_sagital_view()
                #-----------------------------------------------comment x
                self.rewrite_and_copy_cv_image_xz_o()
                #------------------------------------------------
                self.try_draw_line(self.cv_image_xz,self.lastPoint_xz,self.newPoint_xz)
                self.set_image_to_qlabel_xz(self.cv_image_xz)
                self.imageXZNum.setText(str(self.no_of_slices_xz + 1))
            else:
                pass
        else:
            pass

    def StoP(self):
        if (self.btn_StoP.text() == "Supine"):
            self.btn_StoP.setText("Prone")
            self.supine = False
        else:
            self.btn_StoP.setText("Supine")
            self.supine = True

        if self.cv_image_xz_o is not None:
            self.cv_image_yz = self.cv_image_yz_o.copy()
            self.cv_image_xz = self.cv_image_xz_o.copy()
            if self.newPoint_yz is not None:
                cv2.line(self.cv_image_yz, self.lastPoint_yz, self.newPoint_yz, (0, 0, 255), 4)
            if self.newPoint_xz is not None:
                cv2.line(self.cv_image_xz, self.lastPoint_xz, self.newPoint_xz, (0, 0, 255), 4)
            else:
                pass
            self.markLR(self.cv_image_yz)
            self.markHF(self.cv_image_xz)
            self.set_image_to_qlabel_xz(self.cv_image_xz)
            self.set_image_to_qlabel_yz(self.cv_image_yz)
        else:
            pass

    def htof(self):
        if (self.btn_htof.text() == "Head-First"):
            self.btn_htof.setText("Feet-First")
            self.reflex = True
        else:
            self.btn_htof.setText("Head-First")
            self.reflex = False

        if self.cv_image_xz_o is not None:
            if self.newPoint_yz is not None:
                cv2.line(self.cv_image_yz, self.lastPoint_yz, self.newPoint_yz, (0, 0, 255), 4)
            if self.newPoint_xz is not None:
                cv2.line(self.cv_image_xz, self.lastPoint_xz, self.newPoint_xz, (0, 0, 255), 4)
            else:
                pass
            self.set_image_to_qlabel_xz(self.cv_image_xz)
            self.set_image_to_qlabel_yz(self.cv_image_yz)
        else:
            pass

    def Reset(self):
        if self.cv_image_yz_o is not None:
            self.windowLabel = 700  # window centre
            self.windowWidth = 1400  # window width total of side
            #---------------------------------------
            self.rewrite_and_copy_cv_image_xz_o()
            self.rewrite_and_copy_cv_image_yz_o()
            #-----------------------------------------------------------
            self.try_draw_line(self.cv_image_xz,self.lastPoint_xz,self.newPoint_xz)
            self.try_draw_line(self.cv_image_yz,self.lastPoint_yz,self.newPoint_yz)
            self.set_image_to_qlabel_xz(self.cv_image_xz)
            self.set_image_to_qlabel_yz(self.cv_image_yz)
        else:
            pass

    # why use self. if no use in other function? may need to remove
    def updateIMUdata(self,Euler_Angle):
        self.EulerAngle = [round(Euler_Angle[0],1),round(Euler_Angle[1],1),round(Euler_Angle[2],1)]
        self.NeedleAngle = [round(self.EulerAngle[0],1), round(self.EulerAngle[1],1),round(self.EulerAngle[2] - self.CTAngle[2], 1)]  # pitch -90<p<90 cant just use -
        if abs(self.NeedleAngle[2])>180: #normalise all angle to -180 to 180 everytime after calculate
            if self.NeedleAngle[2] > 0:
                self.NeedleAngle[2] = -360 + self.NeedleAngle[2]
            else: # self.NeedleAngle[2]<0
                self.NeedleAngle[2] = 360 + self.NeedleAngle[2]
        else:
            self.NeedleAngle[2] = self.NeedleAngle[2]

        if abs(int(self.NeedleAngle[1])) == 0 : # on horizontal plane
            if abs(int(self.NeedleAngle[2])) == 90 : # parallel to YZ
                self.YZ_projection = 0
                self.XZ_projection = 90
            elif int(self.NeedleAngle[2]) == 0 or abs(int(self.NeedleAngle[2])) == 180: # parallel to XZ
                self.YZ_projection = 90
                self.XZ_projection = 0
            else:
                self.YZ_projection = 0
                self.XZ_projection = 0

        else:
            if abs(int(self.NeedleAngle[2])) == 90: # parallel to XZ
                self.XZ_projection = 90
                self.YZ_projection = abs(self.NeedleAngle[1])
            elif int(self.NeedleAngle[2]) == 0 or abs(int(self.NeedleAngle[2])) == 180: # parallel to YZ
                self.YZ_projection = 90
                self.XZ_projection = abs(self.NeedleAngle[1])
            else:
                self.YZ_projection = abs(math.degrees(math.atan(math.tan(math.radians(self.NeedleAngle[1])) / math.sin(math.radians(self.NeedleAngle[2])))))
                self.XZ_projection = abs(math.degrees(math.atan(math.tan(math.radians(self.NeedleAngle[1])) / math.cos(math.radians(self.NeedleAngle[2])))))
        self.XZ_angle.setText(str(round((90-self.XZ_projection),1)))
        self.YZ_angle.setText(str(round((90-self.YZ_projection),1)))
        self.Roll_angle.setText(str(self.NeedleAngle[0]))
        self.Pitch_angle.setText(str(self.NeedleAngle[1]))
        self.Yaw_angle.setText(str(self.NeedleAngle[2]))

        try:
            if self.adjustwindow == False:
                if self.cv_image_yz is None or self.newPoint_yz is None: #if no CT image or haven't draw line pass
                    pass
                else:
                    self.Needle_image_yz = self.cv_image_yz.copy()
                    if self.NeedleAngle[1] > 0:  # above (image) horizontal plane
                        if self.NeedleAngle[2] < 0:  # on (image) left side
                            yz_drawangle = self.YZ_projection + 180
                        elif self.NeedleAngle[2] > 0:  # on (image) right side
                            yz_drawangle = -self.YZ_projection
                        else:  # self.NeedleAngle[2] == 0: #(image) vertical
                            yz_drawangle = -90

                    elif self.NeedleAngle[1] < 0:  # bolow (image) horizontal plane
                        if self.NeedleAngle[2] < 0:  # on (image) left side
                            yz_drawangle = -self.YZ_projection + 180
                        elif self.NeedleAngle[2] > 0:  # on (image) right side
                            yz_drawangle = self.YZ_projection
                        else:  # self.NeedleAngle[2] == 0: # (image) vertical
                            yz_drawangle = 90

                    else:  # self.NeedleAngle[1] == 0: #on (image) horizontal plane
                        if self.NeedleAngle[2] < 0:  # on (image) left side
                            yz_drawangle = 180
                        elif self.NeedleAngle[2] > 0:  # on (image) right side
                            yz_drawangle = 0
                        else:  # self.NeedleAngle[2] == 0: #(image) vertical
                            yz_drawangle = -90

                    if self.reflex == False:
                        self.Needle_image_yz = self.drawNeedle(self.Needle_image_yz, yz_drawangle, self.lastPoint_yz)
                    else:
                        self.Needle_image_yz = self.drawNeedle(self.Needle_image_yz, 180-yz_drawangle, self.lastPoint_yz)
                    self.set_image_to_qlabel_yz(self.Needle_image_yz)

                if self.cv_image_xz is None or self.newPoint_xz is None: #if no CT image or haven't draw line pass
                    pass
                else:
                    self.Needle_image_xz = self.cv_image_xz.copy()
                    if abs(self.NeedleAngle[2]) < 90: #below (image) horizontal plane
                        if self.NeedleAngle[1] > 0:  # on (image) left side
                            xz_drawangle = (self.XZ_projection) + 90
                        elif self.NeedleAngle[1] < 0:  # on (image) right side
                            xz_drawangle = -(self.XZ_projection) + 90
                        else:# self.NeedleAngle[1] == 0:  # (image) vertical
                            xz_drawangle = 90

                    elif abs(self.NeedleAngle[2]) > 90: #above (image) horizontal plane
                        if self.NeedleAngle[1] > 0:  # on (image) left side
                            xz_drawangle = -(self.XZ_projection) - 90
                        elif self.NeedleAngle[1] < 0:  # on (image) right side
                            xz_drawangle = (self.XZ_projection) - 90
                        else:# self.NeedleAngle[1] == 0:  # (image) vertical
                            xz_drawangle = -90

                    else:# abs(self.NeedleAngle[2]) == 90:  # on (image) horizontal plane
                        if self.NeedleAngle[1] > 0: #on (image) left side
                            xz_drawangle = 180
                        elif self.NeedleAngle[1] < 0: #on (image) right side
                            xz_drawangle = 0
                        else:# self.NeedleAngle[1] == 0: #(image) vertical
                            xz_drawangle = 180

                    if self.reflex == False:
                        self.Needle_image_xz = self.drawNeedle(self.Needle_image_xz, xz_drawangle, self.lastPoint_xz)
                    else:
                        self.Needle_image_xz = self.drawNeedle(self.Needle_image_xz, -xz_drawangle, self.lastPoint_xz)

                    self.set_image_to_qlabel_xz(self.Needle_image_xz)
        except Exception as e:
            print(e)
            pass

        QApplication.processEvents()

    def adjustWindow(self, new_point ,old_point ,windowLabel ,windowWidth):
        adjustWL = (new_point[1] - old_point[1])
        adjustWW = (new_point[0] - old_point[0])
        windowLabel = int(windowLabel + adjustWL)
        windowWidth = int(windowWidth + adjustWW)
        return windowLabel ,windowWidth

    # cut axial_view from 3D array than save in self.cv_image_yz_o_1c
    def cut_axial_view(self): #need change
        axial_view = self.img3d[:, :, self.no_of_slices_yz]
        axial_view = cv2.resize(axial_view, (int(self.img3d.shape[0] / self.ax_aspect), int(self.img3d.shape[1])),interpolation=cv2.INTER_CUBIC)
        self.cv_image_yz_o_1c = axial_view

    # cut sagital_view from 3D array than save in self.cv_image_xz_o_1c
    def cut_sagital_view(self): #need change
        sagital_view = self.img3d[:, self.no_of_slices_xz, :]
        sagital_view = cv2.resize(sagital_view, (int(self.img3d.shape[2] / self.sag_aspect), int(self.img3d.shape[1])),interpolation=cv2.INTER_CUBIC)
        sagital_view = np.rot90(sagital_view)
        self.cv_image_xz_o_1c = sagital_view

    # use new self.cv_image_yz_o_1c to rewrite cv_image_yz_o and copy cv_image_yz_o to cv_image_yz
    def rewrite_and_copy_cv_image_yz_o(self):
        self.cv_image_yz_o = self.rescaleCT(self.windowWidth, self.windowLabel, self.cv_image_yz_o_1c)
        self.cv_image_yz_o = (np.stack((self.cv_image_yz_o, self.cv_image_yz_o, self.cv_image_yz_o)).transpose(1, 2, 0))
        self.cv_image_yz_o = self.cv_image_yz_o.copy()  # deep copy
        self.cv_image_yz_o = cv2.resize(self.cv_image_yz_o, (self.CT_YZ.width(), self.CT_YZ.height()),interpolation=cv2.INTER_CUBIC)
        self.cv_image_yz = self.cv_image_yz_o.copy()
        self.markLR(self.cv_image_yz)

    # use new self.cv_image_xz_o_1c to rewrite cv_image_xz_o and copy cv_image_xz_o to cv_image_xz
    def rewrite_and_copy_cv_image_xz_o(self):
        self.cv_image_xz_o = self.rescaleCT(self.windowWidth, self.windowLabel, self.cv_image_xz_o_1c)
        self.cv_image_xz_o = (np.stack((self.cv_image_xz_o, self.cv_image_xz_o, self.cv_image_xz_o)).transpose(1, 2, 0)).astype(np.uint8)
        self.cv_image_xz_o = self.cv_image_xz_o.copy()  # deep copy
        if self.cv_image_xz_o.shape[0] < self.cv_image_xz_o.shape[1]:
            self.cv_image_xz_o = cv2.resize(self.cv_image_xz_o, (self.qtwidth_XZ, int((self.qtwidth_XZ * self.cv_image_xz_o.shape[0]) / self.cv_image_xz_o.shape[1])),interpolation=cv2.INTER_CUBIC)
        elif self.cv_image_xz_o.shape[0] > self.cv_image_xz_o.shape[1]:
            self.cv_image_xz_o = cv2.resize(self.cv_image_xz_o, (int((self.qthight_XZ * self.cv_image_xz_o.shape[1]) / self.cv_image_xz_o.shape[0]), self.qtwidth_XZ),interpolation=cv2.INTER_CUBIC)
        else:
            self.cv_image_xz_o = cv2.resize(self.cv_image_xz_o, (self.qtwidth_XZ, self.qthight_XZ),interpolation=cv2.INTER_CUBIC)
        self.cv_image_xz = self.cv_image_xz_o.copy()
        self.markHF(self.cv_image_xz)

    def try_draw_line(self,image,last_point,new_point):
        if new_point is not None:
            cv2.line(image, last_point, new_point, (0, 0, 255), 4)
        else:
            pass

    def set_image_to_qlabel_yz(self, image): #need change
        self.qimage_yz = QImage(image.data.tobytes(), self.cv_image_yz.shape[1],self.cv_image_yz.shape[0],self.cv_image_yz.shape[1] * 3, QImage.Format_RGB888)
        self.CT_YZ.setPixmap(QPixmap.fromImage(self.qimage_yz).scaled(self.qtwidth_YZ, self.qthight_YZ, Qt.KeepAspectRatio))

    def set_image_to_qlabel_xz(self, image): #need change
        self.qimage_xz = QImage(image.data.tobytes(), self.cv_image_xz.shape[1],self.cv_image_xz.shape[0],self.cv_image_xz.shape[1] * 3, QImage.Format_RGB888)
        self.CT_XZ.setPixmap(QPixmap.fromImage(self.qimage_xz).scaled(self.qtwidth_XZ, self.qthight_XZ, Qt.KeepAspectRatio))

    def markHF(self, image ):
        cv2.putText(image, 'H', (30, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, 'F', (30, self.cv_image_xz.shape[0]-5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        return image

    def markLR(self, image ):
        if self.supine == True:
            cv2.putText(image, 'L', (30, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'R', (850, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, 'R', (30, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'L', (850, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
        text = 'WW: ' + str(self.windowWidth) + '  WL: ' + str(self.windowLabel)
        cv2.putText(image, text, (20, 890), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        return image

    def drawdotline(self, img, pt1, pt2, color, thickness, gap):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        return img

    def drawNeedle(self,cvimage,projectionAngle,insertPoint):
        image = cvimage
        angle = projectionAngle
        point = insertPoint
        endy = int(point[1] + 100 * math.sin(math.radians(angle)))
        endx = int(point[0] + 100 * math.cos(math.radians(angle)))
        startpointy = int(point[1] - 500 * math.sin(math.radians(angle)))
        startpointx = int(point[0] - 500 * math.cos(math.radians(angle)))
        end = (endx, endy)
        startpoint = (startpointx, startpointy)
        image = cv2.line(image, point, end, (0, 255, 0), 4)
        image = cv2.line(image, startpoint, point, (255, 0, 0), 4)
        return image

    def IMU_Start2Stop(self):
        if (self.Btn_Start.text() == "Start_IMU"):
            self.Btn_Start.setText("Stop_IMU")
            print("IMU connected.")
            if (self._data_thread.Running == True):
                self._data_thread.start()
            else:
                self._data_thread._mutex.lock()
                self._data_thread.Running = True
                self._data_thread._mutex.unlock()
        else:
            self.Btn_Start.setText("Start_IMU")
            print("IMU disconnected.")
            self._data_thread._mutex.lock()
            self._data_thread.Running = False
            self._data_thread._mutex.unlock()

    def Calibration(self):
        if (self.Btn_Start.text() == "Stop"):
            CTAngle_list = []
            t = 0
            while t < 16: #get how many moment of IMU data
                data = self._data_thread.m_IMU.get_module_data(10) # wait until next moment IMU data
                Roll = data['euler'][0]['Roll']
                Pitch = data['euler'][0]['Pitch']
                Yaw = data['euler'][0]['Yaw']
                CTAngle = [Roll,Pitch,Yaw] # CT IMU data in a moment
                CTAngle_list.append(CTAngle) # CT IMU data list for t moment (each moment=1/60s)
                t += 1
            self.CTAngle = np.mean(CTAngle_list,axis=0)
        else:
            pass

    def real_target_length(self, point1, point2,x_real_ratio ,y_real_ratio ,z_real_ratio ):
        length = round((math.sqrt((((point1[0]-point2[0])*x_real_ratio)**2 + ((point1[1]-point2[1])*y_real_ratio)**2 + ((point1[2]-point2[2])*z_real_ratio)**2)))/10,1)
        return length

    def rescaleCT(self, windowWidth, windowCenter, CTimage):
        minWindow = float(windowCenter) - 0.5 * float(windowWidth)
        newimg = (CTimage - minWindow) / float(windowWidth)
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        newimg = (newimg * 255).astype('uint8')
        return newimg

    def point_location_translate(self, pointlocation,new_image_size ,orginal_image_size):
        x = int(pointlocation[0]*new_image_size[1]/orginal_image_size[1])
        y = int(pointlocation[1]*new_image_size[0]/orginal_image_size[0])
        new_image_location = [x,y]
        return new_image_location

    def Belt_Start2Stop(self): #what happen when bluetooth thread error? the button text and restart bluetooth thread
        if (self.Btn_Start_Belt.text() == "Start_Belt"):
            self.Btn_Start_Belt.setText("Stop_Belt")
            print("Belt connected.")
            if (self._belt_thread.belt_connect == True):
                self._belt_thread.start()
            else:
                self._belt_thread._mutex.lock()
                self._belt_thread.belt_connect = True
                self._belt_thread._mutex.unlock()
        else:
            self.Btn_Start_Belt.setText("Start_Belt")
            print("Belt disconnected.")
            self._belt_thread._mutex.lock()
            self._belt_thread.belt_connect = False
            self._belt_thread._mutex.unlock()

    def Lung_size_data_receiver(self,model_output):# (two_second_later,real_time)
        two_second_later, real_time = model_output
        if self.volume_data_transmit_to_CT_save_function == False:
            self.updatebeltdata(two_second_later,real_time)
        else:
            self.save_ct_lung_volume(real_time)
            self.lung_size.setText("saving...")

        QApplication.processEvents()

    def updatebeltdata(self,two_second_later,real_time):
        real_time = round(real_time,2)
        self.lung_size.setText(str(real_time))
        if self.ct_lung_volume is not None: # before press the save btn_save_ct_lung_volume
            bound_1 = self.ct_lung_volume * 1.15
            bound_2 = self.ct_lung_volume * 0.85
            if self.ct_lung_volume>=0:
                up_bound = bound_1
                low_bound = bound_2
            elif self.ct_lung_volume<0:
                up_bound = bound_2
                low_bound = bound_1

            if two_second_later < up_bound and two_second_later > low_bound:
                self.two_second_led.setPixmap(QtGui.QPixmap("./icons/led-blue-on.png"))
            else:
                self.two_second_led.setPixmap(QtGui.QPixmap("./icons/led-red-on.png"))

            if real_time < up_bound and real_time > low_bound:
                self.now_led.setPixmap(QtGui.QPixmap("./icons/led-green-on.png"))
            else:
                self.now_led.setPixmap(QtGui.QPixmap("./icons/led-red-on.png"))

    def active_save_ct_lung_volume(self):
        if self.volume_data_transmit_to_CT_save_function==False:
            self.volume_data_transmit_to_CT_save_function = True
        else:
            self.volume_data_transmit_to_CT_save_function = False
            self.lung_size_hold_breath.clear()

    def save_ct_lung_volume(self, lung_size: float):
        bound_1 = self.lung_size_past * 1.05
        bound_2 = self.lung_size_past * 0.95
        if self.lung_size_past >= 0:
            up_bound = bound_1
            low_bound = bound_2
        elif self.lung_size_past < 0: #if lung size is negtive *1.2 will be low_bound
            up_bound = bound_2
            low_bound = bound_1

        if lung_size > low_bound and lung_size < up_bound:
            self.lung_size_hold_breath.append(lung_size)
            if len(self.lung_size_hold_breath) > 40:
                self.lung_size_hold_breath.pop(0)
                self.ct_lung_volume = mean(self.lung_size_hold_breath)
                print(self.ct_lung_volume)
        else:
            self.lung_size_hold_breath.clear()
        self.lung_size_past = lung_size


def UI():
    # if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# -----------------------------------------------------------------------------------------
# Implement a handler for evt.EVT_C_STORE
def handle_store(event):
    p = './dicom'
    translator = str.maketrans('', '', string.punctuation)
    if not os.path.exists(p):
        os.makedirs(p)

    """Handle a C-STORE request event."""
    # Decode the C-STORE request's *Data Set* parameter to a pydicom Dataset
    ds = event.dataset

    filename = (ds.SOPInstanceUID[-2:].translate(translator)) + ".dcm"

    # Add the File Meta Information
    ds.file_meta = event.file_meta
    # Save the dataset using the SOP Instance UID as the filename
    ds.save_as('./dicom/' + filename, write_like_original=False)

    # Return a 'Success' status
    return 0x0000

def start_server(q):
    AEname = q.get() #str
    IP = q.get() #str can't enter random number if test enter "" or 127.0.0.1
    Port = q.get() #int
    print(AEname,IP,Port)
    handlers = [(evt.EVT_C_STORE, handle_store)]
    # Initialise the Application Entity
    ae = AE(ae_title = str.encode(AEname))
    # Support presentation contexts for all storage SOP Classes
    ae.supported_contexts = AllStoragePresentationContexts
    # Start listening for incoming association requests
    ae.start_server((IP, Port), evt_handlers=handlers)

# ------------------------------------------------
# input AE,IP,Port
class SeverMainWindow(QMainWindow,Sever):
    def __init__(self,q):
        super(SeverMainWindow, self).__init__()
        self.setupUi(self)
        self.Btn_StartServer.clicked.connect(lambda:self.enter(q))

    def enter(self,q):
        q.put(self.lineEdit_AETitle.text())
        q.put(self.lineEdit_IPAddress.text())
        q.put(int(self.lineEdit_PortNumber.text()))
        self.close()

def input_sever_data(q):
    app = QApplication([])
    window = SeverMainWindow(q)
    window.show()
    sys.exit(app.exec_())

# -----------------------------------
if __name__ == '__main__':
    q = Queue(maxsize=3)
    P1 = Process(target=input_sever_data,args=(q,))
    P2 = Process(target=UI)
    P3 = Process(target=start_server,args=(q,))
    P3.daemon = True
    P1.start()
    P1.join()
    P2.start()
    P3.start()
    P2.join()