"""
=======
ECVProc
=======
ECVProc is Python module for processing ECVPro output files.


=========== ======================================================
 functions
=========== ======================================================
cv_read      Read Capacitance-Voltage data from *.CV file 
iv_read      Read Current-Voltage data from *.IV file
ep_read      Read measured doping profile from *.EP file
log_read     Read data from ecv log file
lin_fit      Returns linear fit for measured 1/C^2 and calculated Doping level 

=========== ======================================================

"""


from ecvproc import cv_read, iv_read, ep_read, log_read, lin_fit