'''
AstroPi 2018/2019
team:                         WARSAW PIONEERS
team's tutor:                 Nina Tomaszewska
code developed by:            Bartosz Grabek
physics theory expert:        Olaf Swiac
junior programming assistant: Tomasz Siudalski

Note 1: One of our team members - Marta M. - decided to resign from being astronomy
specialist in our team due to her numerous school duties she had to perform. We would like
to thank her for her contribution to the project.

Note 2: We are very sorry to touch this topic again, but we just wanted to
remind You that our team has had the AstroPi kit delivered quite late, just before winter
holidays in our country region, hence it was difficult for us to meet and work together, as
we were in different location mainly doing winter sports. Therefore, we would like to thank
you for extending our Phase 2 Submission perios for a week. Nevertheless, we hope our
programme will be a worthy candidate for deployment on ISS.

Note 3: This main.py programme comes with 3 files in its directory folder:
1) haarcascade_frontalface_alt.xml
2) haarcascade_drontalface_default.xml
3) haarcascade_profileface.xml
Without them significant for the experiment function will not work properly.
'''

"I. Importing the necessary libraries/packages/modules/functions"

# for simpler calculating
import math

# time for keeping track of experiment
from datetime import datetime, timedelta
from time import sleep, strftime, time

# create a datetime var to store the start time
start_time = datetime.now()

# create a datetime variable to store the current time (at start the same as start_time)
now_time = start_time

# checking the Python version used (2.7.x or 3.4.x)
import sys
#print(sys.version)

# for saving files
import logging
import logzero
from logzero import logger

#also for saving data files and photo real time analysis
import io
import numpy
import cv2

# for checking CPU temperature
from gpiozero import CPUTemperature

# connecting to the Sense Hat
from sense_hat import SenseHat
sh = SenseHat()

# for camera use
from picamera import PiCamera, Color

# for localization purposes and special functions
from ephem import readtle, degree

# for visual effects
import colorzero

# directory path check
import os
directory_path = os.path.dirname(os.path.realpath(__file__))

"II Sense Hat images and other visual stuff"

# basic colors defined
green = (0, 255, 0)
yellow = (255, 255, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
red2 = (255, 51, 0)
white = (255,255,255)
nothing = (0,0,0)
pink = (255,105, 180)
light_orange = (255, 153, 102)

# images, animations

def arrow_up():
    m = blue
    o = white
    logo = [
    o,o,o,m,m,o,o,o,
    o,o,m,m,m,m,o,o,
    o,m,o,m,m,o,m,o,
    o,o,o,m,m,o,o,o,
    o,o,o,m,m,o,o,o,
    o,o,o,m,m,o,o,o,
    o,o,o,m,m,o,o,o,
    o,o,o,m,m,o,o,o,
    ]
    return logo

def arrow_right():
    m = blue
    o = white
    logo = [
    o,o,o,o,o,o,o,o,
    o,o,o,o,o,m,o,o,
    o,o,o,o,o,o,m,o,
    m,m,m,m,m,m,m,m,
    m,m,m,m,m,m,m,m,
    o,o,o,o,o,o,m,o,
    o,o,o,o,o,m,o,o,
    o,o,o,o,o,o,o,o,
    ]
    return logo

def arrow_left():
    m = blue
    o = white
    logo = [
    o,o,o,o,o,o,o,o,
    o,o,m,o,o,o,o,o,
    o,m,o,o,o,o,o,o,
    m,m,m,m,m,m,m,m,
    m,m,m,m,m,m,m,m,
    o,m,o,o,o,o,o,o,
    o,o,m,o,o,o,o,o,
    o,o,o,o,o,o,o,o,
    ]
    return logo

def arrow_down():
    m = blue
    o = white
    logo = [
    o,o,o,m,m,o,o,o,
    o,o,o,m,m,o,o,o,
    o,o,o,m,m,o,o,o,
    o,o,o,m,m,o,o,o,
    o,o,o,m,m,o,o,o,
    o,m,o,m,m,o,m,o,
    o,o,m,m,m,m,o,o,
    o,o,o,m,m,o,o,o,
    ]
    return logo

def polish_flag(): # stores polish flag for the LED matrix
    W = white
    R = red
    logo = [W for i in range(0,32)]
    logo += [R for i in range(0,32)]
    return logo

def warsaw_flag(): # changes gradually polish_flag into warsaw flag on the LED
    start = colorzero.Color('white')
    end = colorzero.Color('yellow')
    for color in start.gradient(end, steps=100):
        n = color.rgb_bytes
        for i in range(0,8):
            for i2 in range(0,4):
              sh.set_pixel(i,i2, n)
        sleep(0.1)

def face(): # stores face image for the LED matrix
    O = light_orange
    B = blue
    R = red2
    logo = [
    O, O, O, O, O, O, O, O,
    O, B, B, O, O, B, B, O,
    O, B, B, O, O, B, B, O,
    O, O, O, O, O, O, O, O,
    O, O, O, O, O, O, O, O,
    O, R, O, O, O, O, R, O,
    O, O, R, R, R, R, O, O,
    O, O, O, O, O, O, O, O,
    ]
    return logo

def heart(): # stores heart image for the LED matrix
    P = pink
    O = nothing
    logo = [
    O, O, O, O, O, O, O, O,
    O, P, P, O, P, P, O, O,
    P, P, P, P, P, P, P, O,
    P, P, P, P, P, P, P, O,
    O, P, P, P, P, P, O, O,
    O, O, P, P, P, O, O, O,
    O, O, O, P, O, O, O, O,
    O, O, O, O, O, O, O, O,
    ]
    return logo

def raspi_logo(): # stores raspberry logo for the LED matrix
    G = green
    R = red
    O = nothing
    logo = [
    O, G, G, O, O, G, G, O,
    O, O, G, G, G, G, O, O,
    O, O, R, R, R, R, O, O,
    O, R, R, R, R, R, R, O,
    R, R, R, R, R, R, R, R,
    R, R, R, R, R, R, R, R,
    O, R, R, R, R, R, R, O,
    O, O, R, R, R, R, O, O,
    ]
    return logo

# loading_bar_func is a help function for loading_bar (see below)
def loading_bar_func(n, start, pause, color, mode):
    # mode can be from 0 to 5, where:
    # 0,1,2,3 are different flipping modes
    # 4 is one-color single line
    # 5 is random color and mode
    if mode == 5:
        r = randint(0,4)
    else:
        r = mode
    x = start[0]
    y = start[1]

    for i in range(0, n-1):
        sh.set_pixel(x,y,color)
        y += 1
        if r == 3:
            sh.flip_v()
            sh.flip_h()
        elif r == 2:
            sh.flip_h()
            sh.flip_v()
        elif r == 1:
            sh.flip_h()
        elif r == 0:
            sh.flip_v()
        sleep(pause)

    for i in range(0, n-1):
        sh.set_pixel(x,y,color)
        x += 1
        if r == 3:
            sh.flip_v()
            sh.flip_h()
        elif r == 2:
            sh.flip_h()
            sh.flip_v()
        elif r == 1:
            sh.flip_h()
        elif r == 0:
            sh.flip_v()
        sleep(pause)

    for i in range(0, n-1):
        sh.set_pixel(x,y,color)
        y -= 1
        if r == 3:
            sh.flip_v()
            sh.flip_h()
        elif r == 2:
            sh.flip_h()
            sh.flip_v()
        elif r == 1:
            sh.flip_h()
        elif r == 0:
            sh.flip_v()
        sleep(pause)

    for i in range(0, n-1):
        sh.set_pixel(x,y,color)
        x -= 1
        if r == 3:
            sh.flip_v()
            sh.flip_h()
        elif r == 2:
            sh.flip_h()
            sh.flip_v()
        elif r == 1:
            sh.flip_h()
        elif r == 0:
            sh.flip_v()
        sleep(pause)

# loading_bar animates the LED matrix
def loading_bar(pause, color, mode):
    # mode is a sequence of 4-char str representing modes of
    # each help function loading_bar_func
    if color == "random":
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
    loading_bar_func(8, [0,0], pause, color, int(mode[0]))
    loading_bar_func(6, [1,1], pause, color, int(mode[1]))
    loading_bar_func(4, [2,2], pause, color, int(mode[2]))
    loading_bar_func(2, [3,3], pause, color, int(mode[3]))

# random_bar fills the LED matrix one by one
# with one or many randomly chosen colours
def random_bar(pause, mode):
    # mode can be 'random' then the version is randomly chosen
    # mode can be '0' for single color and '1' for many colours
    pos = [[i,i2] for i in range(0,8) for i2 in range(0,8)] # creating a list of all LED pixels
    if mode == 'random':
        r = randint(0,1)
    else:
        r = mode
    # one color
    if r == 0:
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        while len(pos) > 0 :
            pos2 = choice(pos)
            pos.remove(pos2)
            x = pos2[0]
            y = pos2[1]
            sh.set_pixel(x,y,color)
            sleep(pause)
    # colourful version
    else:
        while len(pos) > 0:
            pos2 = choice(pos)
            pos.remove(pos2)
            x = pos2[0]
            y = pos2[1]
            sh.set_pixel(x,y,(randint(0,255),randint(0,255),randint(0,255)))
            sleep(pause)

# makes gradient on the LED matrix
def gradient(pause, steps):
    a = (randint(0,255),randint(0,255),randint(0,255))
    b = (randint(0,255),randint(0,255),randint(0,255))
    start = colorzero.Color(a) # start colour
    end = colorzero.Color(b)   # end colour
    for color in start.gradient(end, steps):
        sh.clear(color.rgb_bytes)
        sleep(pause)

# alarming animation
def alarm_anim(threat):
    for i in range(0,10):
        if i % 2 == 0:
            R = (255,0,0)
        else:
            R = (0,0,0)
        sh.clear(R)
        sleep(0.5)
    sh.show_message("ALARM: " + threat, text_colour=red, scroll_speed=0.04)

#ISS text LED animation
def iss_anim(pause):
    S = (0, 0, 139)
    W = white
    sh.clear(S)
    sh.set_pixel(7,1)
    sleep(pause)
    sh.set_pixel(6,1)
    sleep(pause)
    sh.set_pixel(5,2)
    sleep(pause)
    sh.set_pixel(6,3)
    sleep(pause)
    sh.set_pixel(7,4)
    sleep(pause)
    sh.set_pixel(7,5)
    sleep(pause)
    sh.set_pixel(6,6)
    sleep(pause)
    sh.set_pixel(5,6)
    sleep(pause)
    sh.set_pixel(4,1)
    sleep(pause)
    sh.set_pixel(3,1)
    sleep(pause)
    sh.set_pixel(2,2)
    sleep(pause)
    sh.set_pixel(3,3)
    sleep(pause)
    sh.set_pixel(4,4)
    sleep(pause)
    sh.set_pixel(4,5)
    sleep(pause)
    sh.set_pixel(3,6)
    sleep(pause)
    sh.set_pixel(2,6)
    sleep(pause)
    for i in range(1,7):
        sh.set_pixel(i,0,W)
        sleep(pause)

    #logo = [
    #X,X,X,X,X,X,X,X,
    #W,X,X,W,W,X,W,W,
    #W,X,W,X,X,W,X,X,
    #W,X,X,W,X,X,W,X,
    #W,X,X,X,W,X,X,W,
    #W,X,X,X,W,X,X,W,
    #W,X,W,W,X,W,W,X,
    #X,X,X,X,X,X,X,X,]

"III. Used functions"

#1# Face detection function
# code developed basing on https://realpython.com/face-recognition-with-python/
# function based on machine learning resources (xml open cv cascades):
# 1) haarcascade_frontalface_alt.xml created by Rainer Lienhart
# 2) haarcascade_frontalface_default.xml created by Rainer Lienhart
# 3) haarcascade_profileface.xml contributed by David Bradley from Princeton
# Copyright Note for files in 1), 2) and 3)
'''
                        Intel License Agreement
                For Open Source Computer Vision Library

 Copyright (C) 2000, Intel Corporation, all rights reserved.
 Third party copyrights are property of their respective owners.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * The name of Intel Corporation may not be used to endorse or promote products
     derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
'''

def detect_faces():

    # create a memory stream, so that the photos don't need to be saved
    stream = io.BytesIO()

    # get the picture (not highest resolution, so it should be quite fast)
    with picamera.PiCamera() as camera:
        # rotate the camera
        camera.rotation = 90
        camera.resolution = (1366, 768) #also possible: (320, 240) or (1024,768)
        camera.capture(stream, format='jpeg')

    # converting the picture into a numpy array
    buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

    # now we create an OpenCV image
    image = cv2.imdecode(buff, 1)

    # load a cascade file for detecting faces
    face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/main_program/haarcascade_frontalface_alt.xml')
    face_cascade2 = cv2.CascadeClassifier('/home/pi/Desktop/main_program/haarcascade_frontalface_default.xml')
    face_cascade3 = cv2.CascadeClassifier('/home/pi/Desktop/main_program/haarcascade_profileface.xml')

    # convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # look for faces in the image using the loaded cascade file
    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20),
        flags = cv2.CASCADE_SCALE_IMAGE)

    faces2 = face_cascade2.detectMultiScale(gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20),
        flags = cv2.CASCADE_SCALE_IMAGE)

    faces3 = face_cascade3.detectMultiScale(gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20),
        flags = cv2.CASCADE_SCALE_IMAGE)

    #only for testing purposes
    #print("With haarcascade_frontalface_alt     found: " + str(len(faces))+ " face(s)")
    #print("With haarcascade_frontalface_default found: " + str(len(faces2))+ " face(s)")
    #print("With haarcascade_profileface         found: " + str(len(faces3))+ " face(s)")

    # creating a list containing number of faces detected by each cascade file
    a = [int(len(faces)),
         int(len(faces2)),
         int(len(faces3))]

    # getting the maximum value
    a = max(a)
    #print ("Found "+str(a)+" face(s)")
    return a

##    just for testing purposes
##    draw a rectangle around every found face
##    for (x,y,w,h) in faces:
##        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)   #blue?
##    for (x,y,w,h) in faces2:
##        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)     #red
##    for (x,y,w,h) in faces3:
##        cv2.rectangle(image,(x,y),(x+w,y+h),(255,153,51),2)  #orange
##
##    # save the result image (just for testing purposes, we do know it's not allowed ;) )
##    cv2.imwrite('result.jpg',image)

#2# Raspberry Pi sensors data saving function and alarm system
def get_data():

    # collecting CPUTemperature
    cpu = gpiozero.CPUTemperature()

    # collecting basic sensor data
    temperature = (round(sh.get_temperature(),3))
    if temperature < 10 or temperature > 40:     #alarming conditions
        alarm_anim("temperature: "+str(temperature))
        logger.info("Temperature is beyond safety borders")
    pressure = (round(sh.get_pressure(),3))
    if pressure < 960 or pressure > 1050:        #alarming conditions
        alarm_anim("pressure: "+str(pressure))
        logger.info("Pressure is beyond safety borders")
    humidity = (round(sh.get_humidity(),3))
    if humidity < 20 or humidity > 80:           #alarming conditions
        alarm_anim("humidity: "+str(humidity))
        logger.info("Humidity is beyond safety borders")

    # collecting the orientational data of SennseHat
    orientation = sh.get_orientation()
    orientationx = (round((orientation["yaw"]),3))
    orientationy = (round((orientation["pitch"]),3))
    orientationz = (round((orientation["roll"]),3))

    # collecting compass readings
    mag = sh.get_compass_raw()
    magx = (round((mag["x"]),3))
    magy = (round((mag["y"]),3))
    magz = (round((mag["z"]),3))

    # collecting accelerometer readings
    acc = sh.get_accelerometer_raw()
    accx = (round((acc["x"]),3))
    accy = (round((acc["y"]),3))
    accz = (round((acc["z"]),3))

    # collecting gyroscope readings
    gyro = sh.get_gyroscope_raw()
    gyrox = (round((gyro["x"]),3))
    gyroy = (round((gyro["y"]),3))
    gyroz = (round((gyro["z"]),3))

    # current time
    nt = datetime.now()

    #save data onto data02.csv with logger2
    logger2.info("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", cpu, temperature, pressure, humidity, orientationx, orientationy, orientationz, magx, magy, magz, accx,accy,accz, gyrox, gyroy, gyroz, nt)

#3# third big function was implemented back to the mainloop (below),
#   it calculates the velocity of ISS, and the mass of the body that
#   attracts ISS (in this experiment Earth)

"IV. The main loop"
# rotate the Sense Hat so that the images and texts won't be the wrong way up
sh.rotation = 270

# time of conducting experiment (3 hours minus preparation time before mainloop minus one longest mainloop time; all calculated with tests)
ex_time = 10800 #in seconds

# telemetry data used for localization and other things
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   19035.20511872  .00002363  00000-0  44275-4 0  9994"
line2 = "2 25544  51.6436 302.5658 0005091 350.8118 155.5901 15.53232359154595"

iss = readtle(name, line1, line2)

# Earth radius
r_earth = 6371000 #in metres

# gravitational constant
G = 6.6740831*(10**(-11))

# number of measurements variable set (back) to 0
n = 0

# checking if there are files after tests or falsestarts to be deleted:
# so that they are clean and ready to be filled in a moment
if os.path.isdir(str(directory_path)+"./data01.csv"):
    os.rmdir(str(directory_path)+"./data01.csv")
if os.path.isdir(str(directory_path)+"./data02.csv"):
    os.rmdir(str(directory_path)+"./data02.csv")
if os.path.isdir(str(directory_path)+"./data03.csv"):
    os.rmdir(str(directory_path)+"./data03.csv")
if os.path.isdir(str(directory_path)+"./data04.csv"):
    os.rmdir(str(directory_path)+"./data04.csv")

# create a file savers, new files if they don't exist
#logger will contain information about actions and errors while running the programme
logzero.logfile(directory_path+"/data01.csv")
#logger2 will contain data collected with Raspberry Pi sensors
logger2 = setup_logger(name="logger2", logfile=str(directory_path)+"/data02.csv", level=logging.INFO)
#logger3 will contain data connected with ISS velocity calculation
logger3 = setup_logger(name="logger3", logfile=str(directory_path)+"/data03.csv", level=logging.INFO)
#logger4 will contain face_detection data
logger4 = setup_logger(name="logger4", logfile=str(directory_path)+"/data04.csv", level=logging.INFO)

#writing the first columns for data files
logger2.info("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", 'cpu', 'temperature', 'pressure', 'humidity', 'orientationx', 'orientationy', 'orientationz', 'magx', 'magy', 'magz', 'accx', 'accy','accz', 'gyrox', 'gyroy', 'gyroz', 'time')
logger3.info("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", 'Latitude', 'Longitude', 'Elevation','delta_time','delta_lat','delta_long','delta_elev','omega','v', 'M')
logger4.info("%s,%s", 'time','face_detected')

# starting values, frequency, measurements parameters
# when programme starts, variables are set
iss.compute()
a = float(iss.sublat)
b = float(iss.sublong)
c = float(iss.elevation)
d = datetime.now()

# setting the first time for different actions
first_iss = start_time + timedelta(seconds=60)   # first iss velocity calculation
first_iss2 = start_time + timedelta(seconds=60)  # first time after which the frequency of iss velocity calculation will change
first_faces = start_time + timedelta(seconds=60) # first time for 'detect_faces' with picamera
first_info = start_time + timedelta(seconds=60)  # first time for informative system

# setting measurements frequencies
frequency_iss = 60
frequency_faces = 60
frequency_info = 60

# introducing the team and welcoming visuals
sh.set_pixels(raspi_logo())
sleep(1)
loading_bar(0.05, white, '4444')
iss_anim(0.03)
sleep(1)
random_bar(0.03,1)
sh.set_pixels(polish_flag())
warsaw_flag()
sh.show_message("WARSAW PIONEERS TEAM", text_colour=yellow,back_colour=red, scroll_speed=0.25)
sh.clear()

# informing about mainloop start
logger.info("The mainloop has just started")

# main loop
while (now_time < start_time + timedelta(seconds=ex_time)):

    # checking if the joystick was pressed
    try:
    for event in sh.stick.get_events():
        if event.action == "pressed":
        # Check which direction
            if event.direction == "middle":
                iss_anim(0.01)
    except Exception as e:
        logger.info("In joystick section: "+str(e))

    # checking if it's time for changing the iss velocity frequency
    if now_time > first_iss2:
        frequency_iss = 2100 #35 mins
        first_iss2 = now_time + timdelta(seconds=2700) #45 min time when again frequency_iss will be 2100s

    # checking if it's time to calculate the iss velocity
    if now_time > first_iss:
        try:
            loading_bar(0.01, 'random', '4444')
            iss.compute()
            if a != float(iss.sublat) or b != float(iss.sublong) or c != iss.elevation:
                # first situation (coming over the 180 degree meridian)
                if b > 3 and iss.sublat < -3:
                    #calculating changes in angular displacement
                    delta_a = float(iss.sublat) - a
                    delta_b = float(iss.sublong) + b
                    delta_c = iss.elevation - c
                # second situation (normal)
                else:
                    #calculating changes in angular displacement
                    delta_a = float(iss.sublat) - a
                    delta_b = float(iss.sublong) - b
                    delta_c = iss.elevation - c

                # calculating change in time
                delta_time = datetime.now() - d
                delta_time = int(delta_time.seconds)
                #print("DELTA TIME: " + str(delta_time))

                # Omega as angular velocity (sum of angular displacement vectors)
                omega = ((math.sqrt(delta_a**2 + delta_b**2)) / (delta_time))
                #print("Omega       :" + str(omega)+"rad/s")

                # Calculating velocity based on omega and distance between ISS and Earth centre
                v = omega * (((iss.elevation+c)/2)+r_earth)
                #print("Velocity    :" + str(v)+'ms/s')

                # Calculating Earth Mass based on comparison of
                # the centripetal force and force of gravity
                M = (v**2 * (r_earth + (iss.elevation+c)/2))/G
                #print("Earth Mass  :" + str(M)+'kg')

                # write data onto data03.csv (logger3)
                if delta_time < 2000: #checking whether it's time to change back again the frequency of velocity calculating part
                    logger3.info("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", a,b,c,delta_time,delta_a,delta_b,delta_c,omega,v, M)
                else:
                    frequency_iss = 30

                # d for calculating the delta_time between the measurements
                # update the variables, so that deltas can be found again
                iss.compute()
                a = float(iss.sublat)
                b = float(iss.sublong)
                c = float(iss.elevation)
                d = datetime.now()

            # set new measurement time
            first_iss = datetime.now() + timedelta(seconds=frequency_iss)

        except Exception as e:
            logger.info("In velocity calculator :" +str(e))

    # checking whether it's time for detecting faces
    if now_time > first_faces:
        try:
            sh.clear()
            loading_bar(0.03, 'random', '5555') # visual stuff
            sh.show_message('detecting faces', scroll_speed=0.07)
            faces_detected = detect_faces()     # calling outside function
            if faces_detected > 0:
                sh.set_pixels(face())
                sleep(1)
                sh.show_message("Found face", scroll_speed=0.04, text_colour = white, back_colour = blue)
                sh.clear()
                # writing data onto data04.csv
                logger4.info(datetime.now(), True)
                # if faces are detected the frequency of face test is changed
                frequency_faces = 15  #seconds
            else:
                # writing data onto data04.csv
                logger4.info(datetime.now(), False)
                # if faces are not detected, the next measurement will be in 10mins
                frequency_faces = 600 #seconds = 10minutes

            # updating first_faces so that detect_faces will be called in face_frequency time
            first_faces = now_time + timedelta(seconds=face_frequency)

        except Exception as e:
            logger.info("In face detection section: "+str(e))

    try:
        get_data()
    except Exception as e:
        logger.info("In get_data: "+str(e))
    try:
        r = randint(0,4)
        if r == 0:
            gradient(0.02,30)
        if r == 1:
            random_bar(0.02, 'random')
        if r == 2:
            loading_bar(0.02, 'random', '5555')
    except Exception as e:
        logger.info("In loop visuals: " +str(e))

    # experiment time left
    # print("Time left: " + str(start_time + timedelta(seconds=ex_time) - now_time))

    # update current time and number of measurements
    n+=1
    now_time = datetime.now()

# the experiment ends
sh.clear()
logger.info("Experiment has ended, total measurements: "+str(n))
