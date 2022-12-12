# 实时处理图像，是为了简单处理摄像机下图片的简单分别
# 这里主要使用PySimpleGUI、cv2和numpy库文件，PySimpleGUI库文件实现GUI可视化，cv库文件是Python的OpenCV接口文件
# numpy库文件实现数值的转换和运算，均可通过pip导入

import PySimpleGUI as sg     #pip install pysimplegui
import cv2  as cv                 #pip install opencv-python
import numpy as np           #pip install numpy

# 背景色
sg.theme('LightGreen')

# 定义窗口布局
layout = [
    [sg.Image(filename='', key='image')],
    [sg.Radio('None', 'Radio', True, size=(10, 1))],
    [sg.Radio('threshold', 'Radio', size=(10, 1), key='thresh'),
     sg.Slider((0, 255), 128, 1, orientation='h', size=(40, 15), key='thresh_slider')],
    [sg.Radio('canny', 'Radio', size=(10, 1), key='canny'),
     sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='canny_slider_a'),
     sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='canny_slider_b')],
    [sg.Radio('contour', 'Radio', size=(10, 1), key='contour'),
     sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='contour_slider'),
     sg.Slider((0, 255), 80, 1, orientation='h', size=(20, 15), key='base_slider')],
    [sg.Radio('blur', 'Radio', size=(10, 1), key='blur'),
     sg.Slider((1, 11), 1, 1, orientation='h', size=(40, 15), key='blur_slider')],
    [sg.Radio('hue', 'Radio', size=(10, 1), key='hue'),
     sg.Slider((0, 225), 0, 1, orientation='h', size=(40, 15), key='hue_slider')],
    [sg.Radio('enhance', 'Radio', size=(10, 1), key='enhance'),
     sg.Slider((1, 255), 128, 1, orientation='h', size=(40, 15), key='enhance_slider')],
    [sg.Button('Exit', size=(10, 1))]
]

# 窗口设计
window = sg.Window('OpenCV实时图像处理',
                   layout,
                   location=(800, 400),
                   finalize=True)

# 打开内置摄像头
cap = cv.VideoCapture(0)
while True:
    event, values = window.read(timeout=0, timeout_key='timeout')

    # 实时读取图像
    ret, frame = cap.read()

    # GUI实时更新
    imgbytes = cv.imencode('.png', frame)[1].tobytes()
    window['image'].update(data=imgbytes)

window.close()
# 进行阈值二值化操作，大于阈值values['thresh_slider']的，使用255表示，小于阈值values['thresh_slider']的，使用0表示，
# 效果如下所示：
if values['thresh']:
    frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB)[:, :, 0]
    frame = cv.threshold(frame, values['thresh_slider'], 255, cv.THRESH_BINARY)[1]

#





