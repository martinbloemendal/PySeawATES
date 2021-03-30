'''
Agent classes and functions for the Flopy/NetLogo coupled sandbox model.
These functions provide the main interface between NetLogo and Python
'''

import numpy as np
import pandas as pd
import numexpr as ne
import logging
from grid_functions import PyGrid, PyGridAxi
import flopy.mt3d as mt3

#Variables (e.g. geographic coordinates) to be scaled between NetLogo and Python/SEAWAT.
#A scale factor of 10 indicates the NetLogo variables are scaled down by 10 relative
#to the Python/SEAWAT variables (allows for a smaller NL world to increase performance)
global scale_variables, scale_factor, scale_x, scale_y, offset_x, offset_y
scale_variables = ['plot_xmin','plot_xmax','plot_ymin','plot_ymax',
                   'building_xmin','building_xmax','building_ymin','building_ymax', 'Rth']

# scale_factor = 4.75
# scale_x = 4.741844
# scale_y = 4.76656
# offset_x = 6013
# offset_y = 568.9
   
scale_factor = 1
scale_x = 1
scale_y = 1
offset_x = 0
offset_y = 0


class PyAgent(object):
    '''
    Super class for Python agent objects, do not instantiate directly
    '''
    
    def __init__(self):
        '''
        Initial properties common to all agents
        '''
        
        self.breed = ''
        self.who = 0
        self.xcor = 0
        self.ycor = 0
        self.localized = False
       

class PyWell(PyAgent):
    '''
    Python well object class
    '''
        
    def __init__(self, external_attributes=None):
        '''
        Initialize a Python well object
        
        :param external_attributes: dict of attributes to be assigned to the object (generated
                                    from Excel or NetLogo)
        '''
        
        super(PyWell, self).__init__()
        
        self.breed = 'well'
        self.L = 0
        self.R = 0
        self.C = 0
        self.Q = 0
        self.localized = True
        self.Tmodflow = 0
        self.Hmodflow = 0
        self.Tqmin_c = 0
        self.Tqmin_h = 0
        self.Tqmax_c = 0
        self.Tqmax_h = 0
        self.setpointyear = 0
        self.flow = 0
        self.T_inj_assigned = 0
        self.S_modflow = 0
        self.T_modflow = 0
        self.start_idx = 0
        self.stop_idx = 0
        self.TRE = 0
        self.TRE_ly = 0
        self.Q_BuMod = 0
        self.T_BuMod = 0

        
        if external_attributes:
            for attrib, value in external_attributes.items():
                setattr(self, attrib, value)    
        
        
    #@profile
    def calc_setpoints(self, T_series, year, nYr=5):
        '''
        Calculate cooling and heating setpoints which balance storage over a given period
        for a given well object
        
        Based on adaptSetPoints.m by Theo Olsthoorn as implemented in mfLab
        
        :param T_series: pandas series of daily temperatures, indexed by day
        :param year: int - year for which to calculate the setpoints
        :param nYr: int - number of years over which the storage should be balanced
        '''
        
        Tqmax_c = self.Tqmax_c #Temporary variables to enable the use of numexpr functions
        Tqmax_h = self.Tqmax_h
        
        #Partial temperature series over which the setpoints should be computed
        calibration_temp = T_series[str(year - nYr):str(year - 1)]
        
        T0_c = 0
        T0_h = 20
        
        tqmc1 = self.Tqmax_c
        tqmc2 = T0_c
        tqmc = 0.5 * (tqmc1 + tqmc2)
         
        while tqmc1 - tqmc2 > 0.01:
     
            QdCooling = self.QdMax * ne.evaluate('calibration_temp > tqmc') * \
                np.minimum(1, self.QfracMin + (1-self.QfracMin) * \
                           ne.evaluate('(calibration_temp-tqmc) / (Tqmax_c-tqmc)'))
            QyCooling = np.sum(QdCooling) / nYr
     
            if QyCooling < self.Qy:
                tqmc1 = tqmc
                tqmc = 0.5 * (tqmc1 + tqmc2)
            else:
                tqmc2 = tqmc
                tqmc = 0.5 * (tqmc1 + tqmc2)
     
        self.Tqmin_c = tqmc
        
        tqmh1 = self.Tqmax_h
        tqmh2 = T0_h
        tqmh = 0.5 * (tqmh1 + tqmh2)
     
        while abs(tqmh1 - tqmh2) > 0.01:
            QdHeating = self.QdMax * ne.evaluate('calibration_temp < tqmh') * \
                np.minimum(1, self.QfracMin + (1-self.QfracMin) * \
                           ne.evaluate('(calibration_temp-tqmh) / (Tqmax_h-tqmh)'))
            QyHeating = np.sum(QdHeating) / nYr
            if QyHeating < self.Qy:
                tqmh1 = tqmh
                tqmh = 0.5 * (tqmh1 + tqmh2)
            else:
                tqmh2 = tqmh
                tqmh = 0.5 * (tqmh1 + tqmh2)
 
        self.Tqmin_h = tqmh
        self.setpointyear = year
                
                 

               
    #@profile   
    def calc_LRC(self, grid_obj):
        '''  Calculate the layer, row and column position of a well object within a grid object
        :param grid_obj: Python grid object in which the well is to be located       '''
         
        if grid_obj.__class__ == PyGridAxi:
        
            self.C = 0 #int(np.nonzero(grid_obj.XGR > self.xcor)[0][0]) - 1
            if self.IsMono == 1:
                self.R = 0 # int(len(grid_obj.YGR) - np.nonzero(grid_obj.YGR[::-1] > self.ycor)[0][0]) - 1
            if self.IsMono == 0:
                self.R = self.N * 2
 
            
            zgr_lay = grid_obj.botm[:, self.R, self.C]
            lay_id = np.arange(len(zgr_lay))
            self.zgr = np.insert(zgr_lay, 0, grid_obj.top[self.R, self.C]) #Vector of vert. grid coords at location of well
    
            self.zrange = np.array([self.ztop, self.ztop - self.FilterLength])
            self.stop_idx = np.where((self.zgr <= self.zrange[1]))[0][0]
            self.start_idx = np.where((self.zgr >= self.zrange[0]))[0][-1]
            
            #   lay_idx = np.s_[start_idx:stop_idx]
            self.L = lay_id[self.start_idx:self.stop_idx] #Vector of layer IDs
            
        
        if grid_obj.__class__ == PyGrid:
            
            self.L = 0 #For sandbox model
            self.R = int(len(grid_obj.YGR) - np.nonzero(grid_obj.YGR[::-1] > self.ycor)[0][0]) - 1
            self.C = int(np.nonzero(grid_obj.XGR > self.xcor)[0][0]) - 1
        
            zgr_lay = grid_obj.botm[:, self.R, self.C]
            lay_id = np.arange(len(zgr_lay))
            self.zgr = np.insert(zgr_lay, 0, grid_obj.top[self.R, self.C]) #Vector of vert. grid coords at location of well

            self.zrange = np.array([self.ztop, self.ztop - self.FilterLength])

            self.stop_idx = np.where((self.zgr <= self.zrange[1]))[0][0]
            self.start_idx = np.where((self.zgr >= self.zrange[0]))[0][-1]
        
        #   lay_idx = np.s_[start_idx:stop_idx]
            self.L = lay_id[self.start_idx:self.stop_idx] #Vector of layer IDs


    
       
def calc_flow(well_obj_list, run_length = 12, Qmult=1, imb=0, perlen=30, flowtype=1, excel_filename='well_data.xls', years=1, startwinter=1, temp_assigned = False):# temp_assigned = False):
    
    '''  Calculate the average daily flow of a well object, as required by
    the Modflow WEL package. Use once the setpoints have been computed
    :param T_period: pandas series of daily temperatures over which the flows
                     should be computed
    '''
#    flow = np.zeros((run_length, len(well_obj_list) ))

    PerPerYear= int(round(365/perlen, 0))
    
    for i in well_obj_list:
        i.flow = np.zeros(run_length)
 

    if flowtype == 0:
        PerPerSeason=PerPerYear/4
        start_winter=np.ones(years)
        start_summer=np.ones(years)
        for l in range(years):
            start_winter[l]=l*PerPerYear
            start_summer[l]=l*PerPerYear+PerPerYear/2
        for i in well_obj_list:
            for j in range(years):
                k=int(start_winter[j])
                for l in range(int(PerPerSeason)):
                    i.flow[k+l] = -i.Qy/perlen/PerPerSeason*(2*i.IsCold-1)
                k=int(start_summer[j])
                for l in range(int(PerPerSeason)):
                    i.flow[k+l] = i.Qy/perlen/PerPerSeason*(2*i.IsCold-1)
        
#        self.Q=self.Qy/perlen/3
        
    if flowtype == 1:
        SumSine = 0
        PPY=int(PerPerYear/2)
        for i in range(PPY):
            Sine = np.sin(3.1416*(i)/PPY)
            SumSine += Sine
        for i in well_obj_list:
            for j in range(run_length):
                if startwinter == 1:
                    i.flow[j] = round(np.cos(np.pi*j/PPY)/SumSine*(2*i.IsCold-1)*i.Qy/perlen, 0)
                else:
                    i.flow[j] = -round(np.cos(np.pi*j/PPY)/SumSine*(2*i.IsCold-1)*i.Qy/perlen, 0)
# Correct for imbalances THIS IS TURNED OF NOW
#        for i in range((well_obj_list[-1].SystemPair+1)):#np.arange(int(well_obj_list[-1].SystemPair + 1))):
#            PAIR = np.zeros(2)
#            l=0
#            for j in range(len(well_obj_list)):
#                if well_obj_list[j].SystemPair == i:
#                    PAIR[l] = j
#                    l = 1
#            for k in range(run_length):
#                if well_obj_list[int(PAIR[0])].flow[k] > 0:
#                    well_obj_list[int(PAIR[0])].flow[k] = - well_obj_list[int(PAIR[1])].flow[k]
#                if well_obj_list[int(PAIR[1])].flow[k] > 0:
#                    well_obj_list[int(PAIR[1])].flow[k] = - well_obj_list[int(PAIR[0])].flow[k]                  
 
                
    if flowtype == 2:  
#        print 'setpoints not working yet'
   
        nYr=5
        date.year = well_obj_list[0].YrStart
        for i in well_obj_list:
                    if date.year > i.setpointyear:
                        #If the year has changed: recalculate setpoints for the new year
                        i.calc_setpoints(T_series, date.year, nYr=nYr) 
                    
             
                    
        Tqmin_c = self.Tqmin_c #Temporary variables for numexpr functions
        Tqmax_c = self.Tqmax_c
        Tqmin_h = self.Tqmin_h
        Tqmax_h = self.Tqmax_h
    
        QdCooling = self.QdMax * ne.evaluate('T_period > Tqmin_c') * \
            np.minimum(1, self.QfracMin + (1-self.QfracMin) * \
                       ne.evaluate('(T_period-Tqmin_c) / (Tqmax_c-Tqmin_c)'))
 
        QdHeating = self.QdMax * ne.evaluate('T_period < Tqmin_h') * \
            np.minimum(1, self.QfracMin + (1-self.QfracMin) * \
                       ne.evaluate('(T_period-Tqmin_h) / (Tqmax_h-Tqmin_h)'))
         
        if self.IsCold == 1:
            Qday = QdHeating - QdCooling
        else:
            Qday = QdCooling - QdHeating
        
        self.Q = np.mean(Qday)

        if (self.Q < 0 and self.IsCold == 1): #Summer cold
            self.Q = self.Q * (1 + imb)
        elif (self.Q > 0 and self.IsCold == 0): #Summer warm
            self.Q = self.Q * (1 + imb)
        elif (self.Q < 0 and self.IsCold == 0): #Winter warm
            self.Q = self.Q * (1 - imb)
        elif (self.Q > 0 and self.IsCold == 1): #Winter cold
            self.Q = self.Q * (1 - imb)
    
            

        self.Q = self.Q * self.Qmult      


    if flowtype == 3:
        ''' supply total flow / period in "FLOWS  sheets of excle file '''
        cols = np.arange (len(well_obj_list))
        excel_flow = pd.read_excel(excel_filename, 'FLOWS', usecols=cols)
        for i in well_obj_list:
            for j in range(run_length):
                i.flow[j] = int(excel_flow.loc[j,i.N]/perlen)
        
        if temp_assigned == True:
            for i in well_obj_list:
                i.T_inj_assigned = np.zeros(run_length)
            excel_temp = pd.read_excel(excel_filename, 'TEMPS', usecols=cols)
            for i in well_obj_list:
                for j in range(run_length):
                    i.T_inj_assigned[j] = excel_temp.loc[j,i.N]
                    
            
    for i in well_obj_list:
        for j in range(run_length):
            if i.IsCold == 0:  #WARM WELL
                if imb < 0:    #imbalance towards cooling
                    if i.flow[j] < 0:  #water extracted from warm well
                        i.flow[j] = - imb * i.flow[j] 
                    else:
                        i.flow[j] = i.flow[j] #heat injection during cooling of the building
                else: #Imbalance towards heating
                    if i.flow[j] > 0: #heat injected
                        i.flow[j] = imb * i.flow[j]
                    else:
                        i.flow[j] = i.flow[j]  #heat extraction
            else: ##COLD WELL
                if imb < 0:    #imbalance towards cooling
                    if i.flow[j] > 0:  #water injected to cold well
                        i.flow[j] = -imb * i.flow[j] 
                    else:
                        i.flow[j] = i.flow[j] #water extracted from cold well
                else: #Imbalance towards heating
                    if i.flow[j] < 0: #cold water extraction
                        i.flow[j] = imb * i.flow[j]
                    else:
                        i.flow[j]= i.flow[j]  #Cold water infiltration
                
class PySystem(PyAgent):
    '''
    Python ATES system object class
    '''
        
    def __init__(self, external_attributes=None):
        '''
        Initialize a Python ATES system object
        
        :param external_attributes: dict of attributes to be assigned to the object (generated
                                    from Excel or NetLogo)
        '''
        
        super(PySystem, self).__init__()
        
        if external_attributes:
            for attrib, value in external_attributes.items():
                setattr(self, attrib, value)     

        self.breed = 'system'
        

    #@profile     
  
        
def createobjfromExcel(objclass, filename, sheetname, cols):
    '''
    Create a list of Python agent objects based on data from an Excel configuration file
    
    :param objclass: Python object class to be instantiated
    :param filename: .xls to be opened
    :param sheetname: sheet containing the object data
    :param cols: column indices containing the data to be read - ensure the columns
                 match the netlogo_init_attribs list to maintain consistency between
                 Excel, Python and NetLogo
    :returns: list of Python agent objects, instantiated using the Excel attributes
    '''
    
    excel_df = pd.read_excel(filename, sheetname, usecols=cols)
    obj_list = []
     
    for i in range(len(excel_df)):
        obj_attributes = dict(excel_df.iloc[i])
        newobj = objclass(obj_attributes)
        obj_list.append(newobj)   

    return obj_list

    
def create_conc_list(well_obj_list, attrib='T_inj'):
    '''
    Output a species concentration array as required for MT3DMS. Default attribute is temperature
     
    :param well_obj_list: list of Python well objects
    :param attrib: Python object attribute corresponding to the requested concentration
    :returns: array of concentrations, formatted for the MT3DMS SSM package
    '''
    itype = mt3.Mt3dSsm.itype_dict()
    ssm_data = {}
    ssmlist = []
    if isinstance(attrib,str):
         for i in well_obj_list:
            n_layers = len(i.L)
            if n_layers == 1:
                ssmlist.append([i.L, i.R, i.C, getattr(i, attrib), itype['WEL']])
            else:
                for k in range(n_layers):
                    ssmlist.append([i.L[k], i.R, i.C, getattr(i, attrib), itype['WEL']])
                    
                              
    elif isinstance(attrib,list):
        for i in well_obj_list:
            n_layers = len(i.L)
            if n_layers == 1:
                ssmlist.append([i.L, i.R, i.C, getattr(i, attrib[0]), itype['WEL'], \
                                getattr(i, attrib[0]), getattr(i, attrib[1])])
            else:
                for k in range(n_layers):
                    ssmlist.append([i.L[k], i.R, i.C, getattr(i, attrib[0]), itype['WEL'], \
                                    getattr(i, attrib[0]), getattr(i, attrib[1])])     
       
    
    ssm_data[0] = ssmlist

    return ssm_data
 
    
 
def create_LRCQ_list(well_obj_list, grid_obj):
    '''
    Format a list of layer/row/column positions and flows
     
    :param well_obj_list: list of Python well objects
    :returns: LRCQ list, formatted for the Modflow WEL package
    '''
     
#     LRCQ_list = {}
#     LRCQ_list[0] = [[i.L[0], i.R, i.C, i.Q] for i in well_obj_list]
#     print LRCQ_list 
    LRCQ_dict = {}
    LRCQ_list = []
    for i in well_obj_list:
        n_layers = len(i.L)
        if n_layers == 1:
            LRCQ_list.append([i.L, i.R, i.C, i.Q])
        else:
            trans_vec = -np.diff(i.zgr[i.start_idx:i.stop_idx+1])*grid_obj.HK[:,i.R,i.C][i.start_idx:i.stop_idx]
            q_vec = i.Q*trans_vec/np.sum(trans_vec)
            for k in range(n_layers):
                LRCQ_list.append([i.L[k], i.R, i.C, q_vec[k]])
            
    LRCQ_dict[0] = LRCQ_list
    
    return LRCQ_dict


        
def RecoveryEfficiencyW(x, well_obj_list,sys_obj_list, rl, T_bt,nW,lenly,years):
    rl = len(x)
    #x = Run_output_fin
    for j in range(nW):
       # a = 4+len(sys_obj_list)
        
        #Eout = sum(abs((x.loc[:,'W'+str(j)+'_T_mf_out']-T_bt) * x.loc[:,'W'+str(j)+'_Vin'])) #'W'+str(j)+'_Vin'   Vout * T_mf_out
        #Ein =  sum(abs((x.loc[:,'W'+str(j)+'_T_sys_in']-T_bt)* x.loc[:,'W'+str(j)+'_Vout'])) #                       Vin * T_sys_in
        
        #well_obj_list[j].TRE = abs(Eout/Ein)
        
        
        Eout = sum(abs((x.loc[-1:(((lenly)*(years-1))-1),'W'+str(j)+'_T_mf_out']-T_bt) * x.loc[-1:(((lenly)*(years-1))-1),'W'+str(j)+'_Vin'])) #'W'+str(j)+'_Vin'   Vout * T_mf_out
        Ein =  sum(abs((x.loc[-1:(((lenly)*(years-1))-1),'W'+str(j)+'_T_sys_in']-T_bt)* x.loc[-1:(((lenly)*(years-1))-1),'W'+str(j)+'_Vout'])) #                       Vin * T_sys_in
        
        well_obj_list[j].TRE = abs(Eout/Ein)
        
        Eout_xx = sum(abs((x.loc[rl-lenly:rl,'W'+str(j)+'_T_mf_out']-T_bt) * x.loc[rl-lenly:rl,'W'+str(j)+'_Vin']))
        Ein_xx =  sum(abs((x.loc[rl-lenly:rl,'W'+str(j)+'_T_sys_in']-T_bt)* x.loc[rl-lenly:rl,'W'+str(j)+'_Vout']))
        
        well_obj_list[j].TRE_ly = abs(Eout_xx / Ein_xx)
        
        Eout_yy = sum(abs((x.loc[-1:(lenly-1),'W'+str(j)+'_T_mf_out']-T_bt) * x.loc[-1:(lenly-1),'W'+str(j)+'_Vin']))
        Ein_yy =  sum(abs((x.loc[-1:(lenly-1),'W'+str(j)+'_T_sys_in']-T_bt)* x.loc[-1:(lenly-1),'W'+str(j)+'_Vout']))
        
        well_obj_list[j].TRE_fy = abs(Eout_yy / Ein_yy)
        
        
    return well_obj_list

def RecoveryEfficiencyW_dens(x, well_obj_list,sys_obj_list, rl, T_bt,nW,lenly,years):
    rl = len(x)
    #x = Run_output_fin
    for j in range(nW):
       # a = 4+len(sys_obj_list)
        
        #Eout = sum(abs((x.loc[:,'W'+str(j)+'_T_mf_out']-T_bt) * x.loc[:,'W'+str(j)+'_Vin'])) #'W'+str(j)+'_Vin'   Vout * T_mf_out
        #Ein =  sum(abs((x.loc[:,'W'+str(j)+'_T_sys_in']-T_bt)* x.loc[:,'W'+str(j)+'_Vout'])) #                       Vin * T_sys_in
        
        #well_obj_list[j].TRE = abs(Eout/Ein)
        
        
        Eout = sum(abs((x.loc[-1:(((lenly)*(years-1))-1),'W'+str(j)+'_T_mf_out']-T_bt) * x.loc[-1:(((lenly)*(years-1))-1),'W'+str(j)+'_Vin'] * x.loc[-1:(((lenly)*(years-1))-1),'Dens_water_out'])) #'W'+str(j)+'_Vin'   Vout * T_mf_out
        Ein =  sum(abs((x.loc[-1:(((lenly)*(years-1))-1),'W'+str(j)+'_T_sys_in']-T_bt)* x.loc[-1:(((lenly)*(years-1))-1),'W'+str(j)+'_Vout'] * x.loc[-1:(((lenly)*(years-1))-1),'Dens_water_in'])) #                       Vin * T_sys_in
        
        well_obj_list[j].TRE_D = abs(Eout/Ein)
        
        Eout_xx = sum(abs((x.loc[rl-lenly:rl,'W'+str(j)+'_T_mf_out']-T_bt) * x.loc[rl-lenly:rl,'W'+str(j)+'_Vin'] * x.loc[rl-lenly:rl,'Dens_water_out'] ))
        Ein_xx =  sum(abs((x.loc[rl-lenly:rl,'W'+str(j)+'_T_sys_in']-T_bt)* x.loc[rl-lenly:rl,'W'+str(j)+'_Vout'] * x.loc[rl-lenly:rl,'Dens_water_in']))
        
        well_obj_list[j].TRE_D_ly = abs(Eout_xx / Ein_xx)
        
        Eout_yy = sum(abs((x.loc[-1:(lenly-1),'W'+str(j)+'_T_mf_out']-T_bt) * x.loc[-1:(lenly-1),'W'+str(j)+'_Vin'] * x.loc[-1:(lenly-1),'Dens_water_out']))
        Ein_yy =  sum(abs((x.loc[-1:(lenly-1),'W'+str(j)+'_T_sys_in']-T_bt)* x.loc[-1:(lenly-1),'W'+str(j)+'_Vout'] * x.loc[-1:(lenly-1),'Dens_water_in']))
        
        well_obj_list[j].TRE_D_fy = abs(Eout_yy / Ein_yy)
        
        
    return well_obj_list

