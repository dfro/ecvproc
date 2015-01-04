# -*- coding: utf-8 -*-

import numpy as np
import linecache
import math

#Constants
eps0 = 8.854187817e-14 #F*cm^-1
q = 1.602176462e-19 #C

def cv_read(cv_file, model):
    """ Read Capacitance-Voltage data from *.CV file
    
    Parameters
    ----------
    cv_file : str, path to the CV file
    model : str, 'Cp' or 'Cs' - model for calculation capacitance
    
    Returns
    -------
    Cp, Cs : float, capacitance in uF/cm^2
    voltage : float, voltage in V
    
    """
    
    #get area, size and frequency from line 12 of CV file
    #example: 0 0.106 2327.1 443.02 0
    #(?) (area) (first frequency in rad) (second freq) (?)
    cv_properties = linecache.getline(cv_file, 12)
    cv_properties = cv_properties.split()
    area = float(cv_properties[1])
    freq = float(cv_properties[2])

    #build voltage array from values in line 15
    #example: -0.64142 0.008 19 100
    #(start voltage) (step) (number of negative steps) (total number of steps)
    voltage_range = linecache.getline(cv_file, 15)
    voltage_range = voltage_range.split()
    vstart = float(voltage_range[0])
    vstep = float(voltage_range[1])
    nneg = float(voltage_range[2])
    npos = float(voltage_range[3]) - nneg
    voltage = np.concatenate((
                              np.linspace(vstart, vstart-vstep*nneg, nneg+1),
                              np.linspace(vstart, vstart+vstep*npos, npos,
                                          endpoint=False)
                              ),
                             axis = 0)

    # Y is complex admittance (Gp+Bp*i)
    cnv = dict.fromkeys([0], lambda x: complex(*eval(x)))
    Y = np.genfromtxt(cv_file, converters=cnv, 
                      delimiter=25, skip_header=15)
    #sort array
    array = np.vstack((voltage, Y.real, Y.imag)).T
    array = array[array[:,0].argsort(0)]
    voltage = array[:,0]
    Y.real = array[:,1]
    Y.imag = array[:,2]

    Diss = Y.real/Y.imag
    Cp = 1e6*Y.imag/(freq*area)
    Cs = Cp*(1+Diss**2)

    if model == 'Cp':
        return Cp, voltage
    if model == 'Cs':
        return Cs, voltage

def iv_read(iv_file):
    """ Read Current-Voltage data from *.IV file
    
    Parameters
    ----------
    cv_file : str, path to the IV file
    
    Returns
    -------
    current : float, current in mA/cm^2
    voltage : float, voltage in V
    
    """
    
    #build voltage array from values in line 15
    #example: -0.64142 0.008 19 100
    #(start voltage) (step) (number of negative steps) (total number of steps)
    voltage_range = linecache.getline(iv_file, 15)
    voltage_range = voltage_range.split()
    vstart = float(voltage_range[0])
    vstep = float(voltage_range[1])
    nneg = float(voltage_range[2])
    npos = float(voltage_range[3]) - nneg
    voltage = np.concatenate((
                              np.linspace(vstart, vstart-vstep*nneg, nneg+1),
                              np.linspace(vstart, vstart+vstep*npos, npos,
                                          endpoint=False)
                              ),
                             axis = 0)

    # read current 
    current = np.genfromtxt(iv_file, skip_header=15)
    
    #sort array
    array = np.vstack((voltage, current)).T
    array = array[array[:,0].argsort(0)]
    voltage = array[:,0]
    current = array[:,1]

    return current, voltage

def ep_read(ep_file):
    """ Read measured doping profile from *.EP file
    
    Parameters
    ----------
    ep_file : str, path to the EP file

    Returns
    -------
    doping : float, doping level in cm^-3
    depth : float, depletion width + etched width in um
    
    """
    data=np.genfromtxt(ep_file, skip_header=13, names=['depth', 'doping'])
    
    doping=data['doping']
    depth=data['depth']
    
    return doping, depth
	
def log_read(log_file, *field_name):
    """ Read data from ecv log file
    
    Parameters
    ----------
    log_file : str, path to log file
    field_name : tuple of str, name of field from the list
        list=('No', 'Lmp', 'MC', 'V-etch', 'I-etch', 
              'V-meas', 'I-meas', 'Dis', 'FBP', 'Wr',
              'Wd', 'X', 'N', 'F1', 'F2', 'Amp', 'dV')
    
    Returns
    -------
    data : tuple of nparrays
   
    """

    dtype=np.dtype([('No', '>i4'), ('Lmp', '>i4'), ('MC', '|S8'), 
           ('V-etch', '>f4'), ('I-etch', '>f4'), ('V-meas', '>f4'),
           ('I-meas', '>f4'), ('Dis', '>f4'), ('FBP', '>f4'), 
           ('Wr', '>f4'), ('Wd', '>f4'), ('X', '>f4'), ('N', '>f4'), 
           ('F1', '>f4'), ('F2', '>f4'), ('Amp', '>f4'), ('dV', '>f4')])
            
    f=open(log_file, 'r')
    exclude=['Spot', 'Value','Freq.', 'Dis.', 'C', 'G','Rs', 
             'dC/dV', 'FBP', 'Depl.', 'N', 'No.', 'ECVpro',
             'ID:', 'Description:', 'Saved', 'Spot:', 'Etch',
             'Ring:', 'Recipe:', 'Electrolyte:', 'Pot:', 
             'Contact', 'ECVision']

    data=[]
    
    for line in f:
        if not [s for s in line.split() if s in exclude] \
        and not line.split()==[]:
            if not 'F1=' in line:
                data.append(tuple(line.split())+
                            (mp['F1'], mp['F2'], mp['Amp'], mp['dV']))
            else:
                mp=dict(s.split('=') for s in line.split(', '))
            
    data = np.vstack(np.array(data, dtype=dtype))
    
    return tuple(np.reshape(data[i], -1) for i in field_name)
        
def lin_fit(capacitance, voltage, vmin=None, vmax=None, eps=15.15):
    """ Returns linear fit for measured 1/C^2 and calculated Doping level 
    
    Parameters
    ----------
    capacitance : float, capacitance in uF/cm^2
    voltage : float, voltage in V
    vmin, vmax : float, range for linear fitting
    eps : float, dielectric constant (default is for InAs)
    
    Returns
    -------
    cap_fit: ndarray, 1/C^2 in cm^4/uF^2
    volt_fit: ndarray, voltage in V
    doping: float, calculated doping level in cm^-3
    
    """
   
    if not vmin: vmin=min(voltage)
    if not vmax: vmax=max(voltage)
    volt_in_rage, cap_in_rage = [],[]
    for i in range(voltage.size):
        if voltage[i] >=vmin and voltage[i] <=vmax:
            volt_in_rage.append(voltage[i])
            cap_in_rage.append(capacitance[i])
     
            
    volt_in_rage = np.asarray(volt_in_rage)
    cap_in_rage = np.asarray(cap_in_rage)
    coeff = np.polyfit(volt_in_rage, 1/cap_in_rage**2, 1)
    
    # volt_fit contain two point
    # first 1/C^2-->0, second is based on maximum capacitance
    b = max(1/capacitance**2)
    k = 10**round(math.log10(b))
    volt_fit = [-coeff[1]/coeff[0], 
               (k*math.ceil(b/k)-coeff[1])/coeff[0]
               ]
    volt_fit = np.array(volt_fit)
    cap_fit = np.polyval(coeff, volt_fit)

    #doping calculation
    doping = -1e-12*2/(coeff[0]*q*eps*eps0) #cm^-3
    return cap_fit, volt_fit, doping