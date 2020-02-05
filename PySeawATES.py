'''
    PySeawATES
    You are using the pyhton code developed to simulate 
    Aquifer Thermal Eneregy Storage (ATES) systems in MODFLOW/MT3D-MS/SEAWAT
    This code is developed at Delft University of Technology and 
    KWR water research institute Various researchers have contributed to 
    key elements of this code: dr. Martin Bloemendal, dr.Marc Jaxa-Rozen, 
    prof.dr.Theo Olsthoorn, Stijn Beernink. If you have any questions or 
    remarks please contact:
        Martin BLoemendal: j.m.bloemendal@tudelft.nl
        Stijn Beernink: stijn.beernink@kwrwater.nl
    The authors take no responsibility for any damage the may follow from 
    using or implementing (the results produced by) this model infrastructure
'''

#%%==============================================================================
''' [A] all imports, paths, main settings etc.'''
import os
import sys
import seaborn as sns
os.chdir('D:\Dropbox\Py\ATES_SWRv4_model_KWR')                                  # Set the proper working directory (WD)
import shutil 
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import flopy #print("flopy version = {}".format(flopy.__version__))
import flopy.modflow as mf
import flopy.mt3d as mt3
import flopy.seawat as swt
import flopy.utils.binaryfile as bf
import grid_functions as gf  
from agent_functions import PyWell, PySystem, createobjfromExcel, create_conc_list,\
                            create_LRCQ_list, calc_flow, RecoveryEfficiencyW, RecoveryEfficiencyW_dens#,  

'''Set name for run''' 
swtexe_name = 'swt_v4.exe'  
excel_filename = 'well_data3D.xlsx'                                             # in this XLS you set your basis ATES and subsurface characteristics. set to 3D/AXI accordingly in l58
naam = 'test'
case = 'v1' 
name = naam + case
dirs = ['./output/'+name]                                                       # make sure to create the 'output' folder in WD

'''grid settings '''
AXI = 0                                                                         # axial symmetric grid or not. | 1=AXI, 0 = 3D |
LIN = 1                                                                         # Linear (1) or logarithmic (0) cell sizes buildup around the center of the modelgrid 
ICBUND_Up = -1                                                                  # TOP:: -1 = boundary has constant temperature (NO Flow, YES Temperature boundary), 1 = NO Flow, No Tempertature boundary
ICBUND_Down = -1                                                                # BOTTOM: -1 = boundary has constant temperature (NO Flow, YES Temperature boundary), 1 = NO Flow, No Tempertature boundary
OutsideAirBound = 0                                                             # 1 = ON, 0=OFF, if ON: the temperature boundary at model top is adjusted following outside air temperature 
steady = True                                                                   # Switch for steady-state/transient Modflow simulation

'''main time Parameters'''
perlen = 30                                                                     # (DAYS)
years = 2                                                                       # set the minimum #years that a system will run
nper = 1                                                                        # Number of Modflow stress periods per coupled simulation period
nstp = 1                                                                        # Number of Modflow time steps per stress period
timprs = np.array([perlen])
nprs = len(timprs)
startwinter = 1                                                                 # simulation starts in winter (1) or summer (0). for distribution of flows of wells and surface temperature

'''Model flow/temp input'''
flowtype = 1                                                                    # 0 = block-function over 4 seasons, 1 = Qy distributed following a cosine, 2 = setpoints according to climate (provide climate in xls), 3 predefined in Xls sheet FLOWS, provide total well discharges for timestep in XLS,
imb = 1                                                                         # 1= balance. imbalance <0 --> more heating (cold well grows). the smallest well gets the volume identified in the XLS. well towards where the imbalance is defined is multiplied with the imbalance factor. (so a imb smaller than one has the opposite effect)
temp_assigned = False                                                           # if True, excel input temperature sheet 'TEMP' are being used as input, if False, constant Temp
gwflow_x = -0.                                                                  # Groundwater flowvelocity in x direction [m/y]
gwflow_y = -0.                                                                  # Groundwater flowvelocity in y direction [m/y] 
WaterDensON = 1                                                                 # Calculation of parameters standard =1

'''Load Excel data file'''                                                      # inistialize Well object, grid object, layers, flows etc
init_well_attribs = ['IsCold','xcor','ycor','parent','SystemPair','YrStart',    # Initial well/system attributes to be passed from the Excel file to the Python objects, Name of attributes to read from Excel file into Python objects
                     'T_inj', 'QdMax','Qy','QfracMin','Rth','Tqmax_c','Tqmax_h','ztop','FilterLength', 'IsMono', 'who', 'N']
init_system_attribs = ['plot_xmin','plot_ymin','plot_xmax','plot_ymax',         # Not used in this example but needed for ATES agents. Create lists of Python objects from Excel file - column indices must be set manually so that the names match init_*_attribs (cols have to be listed in increasing order, but doesn't have to respect order of init_well_attribs)
                       'building_xmin','building_ymin','building_xmax','building_ymax', 'xcor','ycor'] 
sys_obj_list = createobjfromExcel(PySystem,excel_filename, sheetname='Systems',cols='A:BJ') # not used for individual systems. but can be used when simulating multiple ATES systems
well_obj_list = createobjfromExcel(PyWell,excel_filename, sheetname='Wells',cols='A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,AA,AB,AC,AD') #creates well object from  data in xls file
form_obj_list = createobjfromExcel(PySystem,excel_filename, sheetname = 'Layers', cols = 'A:J') # creates subsurface/soil properties following the lithology given in the layer table in xls.
mon_obj_list = createobjfromExcel(PySystem,excel_filename, sheetname = 'Monitoring', cols = 'A:D') # creates list of monitoring points in your grid where you can track the temperature
nmon = len(mon_obj_list)
nform = len(form_obj_list)
nW=len(well_obj_list)                                                           # numer of wells
T_amb = form_obj_list[0].s1                                                     # Ambient Temperature of the subsurface

#%%============================================================================
''' [B] Detailed MODFLOW/geohydr. inputs'''                                         # Default conductivities and porosity - assumed constant for  model, can be handled as arrays like temperature/head
''' basic parameter settings'''
PEFF = form_obj_list[0].por                                                     # porosity [-]
al = 0.5                                                                        # dispersivity [m]
trpt = 0.05                                                                     # transv. dispersivity [m]
trpv = 0.005                                                                    # trans dispersivity [m]
ss = 6e-4                                                                       # specific storage [m^-1]
rho_s = 2640.                                                                   # density solids [kg/m3]
rho_f = 1000.                                                                   # density fluids [kg/m3]
rho_b = rho_s * (1-PEFF) + rho_f * PEFF                                         # bulk density [kg/m3]

'''thermal properties'''
Cp_s = 710                                                                      # specific. heatcapacity [J/kg/K]
Cp_f = 4183.                                                                    # specific. heatcapacity [J/kg/K]
Cw = Cp_f * rho_f                                                               # volumetric heatcapacity [J/m3/K]
Cs = Cp_s * rho_s                                                               # volumetric heatcapacity [J/m3/K]
Caq =  Cs * (1-PEFF) + Cw * PEFF                                                # volumetric aqufier heatcapacity
kT_s = 2.                                                                       # thermal conductivity  sandy solids [W/m/K]
kT_clay = 1.7                                                                   # thermal conductivity  clayy solids [W/m/K]
kT_f = 0.58                                                                     # thermal conductivity  water [W/m/K]
kT_aq = kT_s * (1-PEFF) + kT_f * PEFF                                           # thermal conductivity  aquifer bulk
kT_aqt = kT_clay * (1 - PEFF) + kT_f * PEFF                                     # thermal conductivity  aquitard bulk
Tdif_aq = kT_aq / (PEFF * rho_f * Cp_f) * 24 * 3600                             # thermal diffusivity, deals with conduction in dsp package
Tdif_aqt = kT_aqt / (PEFF * rho_f * Cp_f) * 24 * 3600 
Kdist=Cp_s / (rho_f * Cp_f)                                                     # thermal distribution coeff, deals with thermal retardation in rct package

'''Seawat parameters'''
crho_ref = T_amb
visc_ref = 0.00002394*10**(248.37/(crho_ref+133.15))                            # reference viscosity (VOSS 1982)

Tmax=0                                                                          # To calculate an approximation of the Temperature|density relation, the minimum(Tmax) and maximum density (4 C)
Tmin=T_amb                                                                      # please note: seawat uses a linear approximation of this relation
for i in range(len(well_obj_list)):
    if well_obj_list[i].T_inj > Tmax:
        Tmax = well_obj_list[i].T_inj
    if well_obj_list[i].T_inj < Tmin:
        Tmin = well_obj_list[i].T_inj
denseref = 1000 - (((Tmin - 4)**2) / 207)                                       # reference density
densemin = 1000 - (((Tmax - 4)**2) / 207)                                       # minimum density (maximum temperature)
drhodT = (denseref - densemin) / (Tmin - Tmax)                                  # linear relationship between T and p

'''run period info'''
rl = int(round (365 * years / perlen, 0))                                       # run length
Run_info = pd.DataFrame()                                                       # create run info dataframe
Run_info.loc[0,'Period'] = 0
Run_info.loc[0,'Day'] = 0
for i in range(rl):    
    Run_info.loc[i,'Period'] = int(i*1)
    Run_info.loc[i,'Day'] = (i+1)*(perlen)
Run_output = Run_info.copy()                                                    # output dataframe

'''Make grid'''                                                                 # Make the grid + assign properties, initial and boundary conditions  Initialize grid object
dmin = 5    #horizontal                                                         # smallest cel size at well [m]
dmin_bound = 100                                                                # total distance from well with 'dmin' sized cells [m]
dmax = 200                                                                      # largest cell size at model boundary [m]
aroundAll = 500                                                                 # normal=1500 [m] size of grid around well.
nstep = 15                                                                      # minimum number of steps that gridfunctions must add  to acquire the dmax size requiremen [-]
grid_extents = None                                                             # example:[-300,300,-300,300] #set model boundaries [m]
dz = 2.5    #vertical                                                           # vertical gridlayer thickness [m]  important to syncronize with layer tops in xls file!

IsMono=well_obj_list[0].IsMono                                                  # mono or doublet for setting up axigrid
zbot = form_obj_list[nform-1].zBot                                              # bottom of the modelgrid [m]
ztop = form_obj_list[0].zTop                                                    # top of the modelgrid [m]

if AXI == 1:
    grid_obj = gf.PyGridAxi()                                                   # for axi-sym, geo-propterties are automatically adjusted see grid_functions for details
    if LIN == 0:                                                                # Axisymmetrical grid with logarithmic increasing cell sizes from well 
        grid_obj.make_grid_axi_Log(well_obj_list, dmin=dmin, dmax=dmax, dz=dz,dmin_bound=dmin_bound,
                           nstep=int(nstep), ztop=ztop, zbot=zbot, aroundAll=aroundAll, grid_extents=grid_extents, IsMono=IsMono,ICBUND_Up=ICBUND_Up, ICBUND_Down=ICBUND_Down)
    else:                                                                       # Axisymmetrical grid with a zone with constant cell size at well, logarithmic increasing cell sizes from LIN boundary
        grid_obj.make_grid_axi_Lin(well_obj_list, dmin=dmin, dmax=dmax, dz=dz,dmin_bound=dmin_bound,
                           nstep=int(nstep), ztop=ztop, zbot=zbot, aroundAll=aroundAll, grid_extents=grid_extents, IsMono=IsMono,ICBUND_Up=ICBUND_Up, ICBUND_Down=ICBUND_Down)
else:
    grid_obj = gf.PyGrid()  
    if LIN ==0:                                                                 # 3D grid with logarithmic increasing cell sizes from well(s) 
        grid_obj.make_grid_Log(well_obj_list, dmin=dmin, dmax=dmax, dz=dz,dmin_bound=dmin_bound,
                               nstep=int(nstep), ztop=ztop, zbot=zbot, aroundAll=aroundAll, grid_extents=grid_extents, IsMono=IsMono,ICBUND_Up=ICBUND_Up, ICBUND_Down=ICBUND_Down)
    else:                                                                       # 3D grid with a constant cell size basis around the well(s). logarithmic increasing cell sizes from set boundary.
        grid_obj.make_grid_Lin(well_obj_list, dmin=dmin, dmax=dmax, dz=dz,dmin_bound=dmin_bound,
                               nstep=int(nstep), ztop=ztop, zbot=zbot, aroundAll=aroundAll, grid_extents=grid_extents, IsMono=IsMono,ICBUND_Up=ICBUND_Up, ICBUND_Down=ICBUND_Down)

'''set geo properties for each gridcell'''
gf.set_geo_prop(grid_obj, form_obj_list, gwflow_x=gwflow_x, gwflow_y=gwflow_y, dz=dz, AqL=2, Tdif_aqt=Tdif_aqt, Tdif_aq=Tdif_aq, rho_b=rho_b,rho_f=rho_f,Cp_f=Cp_f,Kdist=Kdist)
if OutsideAirBound == 1:
    gf.set_Tboundary(grid_obj,  perlen=perlen, run_length=rl, Tmin=5, Tmax=20, startwinter = startwinter)  # sets varying temperature at surface level. 

'''monitoring Layer, Row, Column number in grid'''
mon_LRC_list = gf.init_monitoring(grid_obj, mon_obj_list, dz, nmon, AXI, IsMono)
RES = np.zeros((rl,len(mon_LRC_list)))                                          # results file for monitoring points

#'''plot grid'''
#if str(grid_obj.__class__)[-5:] == "Axi'>":                            
#    xp, yp = np.meshgrid(grid_obj.sumdelr,grid_obj.botm)
#    plt.scatter(xp,yp,s=0.001,c='red')
#    plt.show()
#else:
#    xp, yp = np.meshgrid(grid_obj.XGR,grid_obj.YGR)
#    plt.scatter(xp,yp,s=0.001,c='red')
#    plt.show()                               
#                
'''initialize well properties for first timestep''' 
for i in well_obj_list: 
    i.calc_LRC(grid_obj)                                                        # Locate each well in the simulation grid

itype = mt3.Mt3dSsm.itype_dict()
well_LRCQ_list = {}
well_LRCQ_list[0] = [[0, 0, 0, 0]]
ssm_data = {}
ssm_data[0] = [[0, 0, 0, 0, itype['WEL']]]
laytyp = np.zeros([grid_obj.nlay])                                              # All confined layers

'''calculate flows '''
calc_flow(well_obj_list, perlen=perlen, flowtype=flowtype, excel_filename=excel_filename, Qmult=1, imb=imb, run_length=rl, years=years, startwinter=startwinter, temp_assigned = temp_assigned) # calculates flows during simulation

'''  MODFLOW INPUT files'''                                                     # set all the fixed conditions that don't change while running etc.
ml = mf.Modflow(name, version='mf2005', exe_name=swtexe_name, model_ws=dirs[0]) 
discret = mf.ModflowDis(ml, nrow=int(grid_obj.nrow), ncol=int(grid_obj.ncol), nlay=grid_obj.nlay,
                 delr=grid_obj.delr, delc=grid_obj.delc, laycbd=0., top=grid_obj.top, 
                 botm=grid_obj.botm, nper=nper, perlen=perlen, nstp=nstp, steady=steady)

lpf = mf.ModflowLpf(ml, hk=grid_obj.HK, vka=grid_obj.VK, ss=grid_obj.ss,ipakcb=53, sy=0.15, laytyp=laytyp, layavg=0.) 
#%%============================================================================
''' [C] Iterate the model''' 
start_time = time.time() 
for period in range(rl):
    if OutsideAirBound == 1:
        grid_obj.temp[0,:,:]= grid_obj.SurfaceT[period]                         # sets temperature at surface level according to time of year
    
    if well_obj_list:                                                           # Create well and temperature lists following Modflow/MT3DMS format. Each timestep, flows and infiltration temp are assigned to the wells
        for i in well_obj_list:                     
            i.Q = i.flow[period]
        if temp_assigned == True:
           i.T_inj = i.T_inj_assigned[period]
            
    well_LRCQ_list = create_LRCQ_list(well_obj_list, grid_obj)                  # Set the settings of the wells for that timestep
    ssm_data = create_conc_list(well_obj_list, attrib='T_inj') 
    
    '''Initialize MODFLOW Packages'''
    bas = mf.ModflowBas(ml, ibound=grid_obj.IBOUND, strt=grid_obj.head)         # Basemodel Modflow
    wel = mf.ModflowWel(ml, stress_period_data=well_LRCQ_list)                  # Put in Wells
    words = ['head','drawdown','budget', 'phead', 'pbudget']                
    save_head_every = 1
    #oc = mf.ModflowOc(ml)                                                       # Output control package class --> moved (p3.7 iso p3.6)                   
    pcg = mf.ModflowPcg(ml, mxiter=200, iter1=200, npcond=1,                    # Preconditioned Conjugate-Gradient Package  --> solves the finite differences equations
                        hclose=0.001, rclose=0.001, relax=1.0, nbpol=0)
    ml.write_input()
    
    '''Initialize MT3DMS packages'''
    mt = mt3.Mt3dms(name, 'nam_mt3dms',exe_name =swtexe_name, modflowmodel=ml, model_ws=dirs[0])
    adv = mt3.Mt3dAdv(mt, mixelm=0, percel=0.8, nadvfd=1, #Particle based methods
                      nplane=0,mxpart=250000, dceps=1e-4, 
                      npl=5, nph=8, npmin=1, npmax=16)
    btn = mt3.Mt3dBtn(mt, cinact=-100., icbund=grid_obj.ICBUND, prsity=grid_obj.PEFF,
                      sconc=grid_obj.temp, #sconc2=grid_obj.salinity,
                      ifmtcn=-1, chkmas=False, nprobs=0, nprmas=1, dt0=0, ttsmult=1,
                      ttsmax=1., ncomp=1, mcomp=1, nprs=nprs, timprs=timprs, mxstrn=9999)
    dsp = mt3.Mt3dDsp(mt, al=al, trpt=trpt, trpv=trpv, multiDiff=True, dmcoef = grid_obj.Tdif) 
    rct = mt3.Mt3dRct(mt, isothm=1, ireact=0, igetsc=0, rhob=grid_obj.rho_b, sp1=grid_obj.Kdist)
    gcg = mt3.Mt3dGcg(mt, mxiter=50, iter1=50, isolve=1, cclose=1e-9, iprgcg=0)
    ssm = mt3.Mt3dSsm(mt, stress_period_data=ssm_data)
    mt.write_input()
        
    '''Initialize SEAWAT packages'''
    mswtf = swt.Seawat(name, 'nam_swt',exe_name=swtexe_name, modflowmodel=ml, mt3dmodel=mt,
                       model_ws=dirs[0])

    vdf = swt.SeawatVdf(mswtf, mtdnconc=-1, nsrhoeos=1, nswtcpl=1, iwtable=0, densemin=0, densemax=0, denseref=denseref,
                        mtrhospec=1,denseslp=drhodT,crhoref=crho_ref)
    vsc = swt.SeawatVsc(mswtf, mt3dmuflg=-1, viscmin=0.0, viscmax=0.0, viscref=visc_ref, #Viscref must be set to the reference viscosity at T=12 --> 
                        nsmueos=1, mtmutempspec=1, mutempopt=1, amucoeff=(2.39e-5, 10, 248.4, 133.2),  # temp is used to calc visc. according to Eq. 18 langevin et al 2008
                        mtmuspec=2, dmudc=1.923e-06, cmuref=0.0,                # solute influence on viscocity
                        invisc=-1, visc=-1, extension='vsc')
    mswtf.write_input()
    oc = mf.ModflowOc(mswtf)
    m = mswtf.run_model(silent=True)                                            # or silent = True // #Run SEAWAT
    
    '''Copy Modflow/MT3DMS output to new files so they wont be overwritten in next timestep.'''
    shutil.copyfile(os.path.join(dirs[0], name+'.hds'),
                    os.path.join(dirs[0], name+str(period)+'.hds'))
    shutil.copyfile(os.path.join(dirs[0], 'MT3D001.UCN'),
                    os.path.join(dirs[0], name+str(period)+'S1'+'.UCN'))

    '''Create head & concentrations file object and read head & concentrations arrays for next simulation period'''
    h_obj = bf.HeadFile(os.path.join(dirs[0], name+str(period)+'.hds'))
    grid_obj.head = h_obj.get_data(totim=perlen)
    t_obj = bf.UcnFile(os.path.join(dirs[0], name+str(period)+'S1'+'.UCN'))
    grid_obj.temp = t_obj.get_data(totim=perlen)                

    if well_obj_list:
        for i in well_obj_list:                                                 # Update each active Python well object with the temperature and head at its grid location
            i.H_modflow = grid_obj.head[i.L[-1],i.R,i.C]
            i.T_modflow = np.average(grid_obj.temp[i.start_idx:i.stop_idx,i.R,i.C])  #the average of all the cells of the injection well! (start.idx and stop.idx) 
   
    '''Save temp monitoring pointsdata to results array'''
    for m in range(len(mon_LRC_list)):
        RES[period,m] = grid_obj.temp[int(mon_LRC_list[m,0]),int(mon_LRC_list[m,1]),int(mon_LRC_list[m,2])]    
    
    '''save the info the the Run_output file'''
    for j in range(len(well_obj_list)):
        if well_obj_list[j].Q >0:
            Run_output.loc[period,'W'+str(j)+'_Vin'] =  0
            Run_output.loc[period,'W'+str(j)+'_Vout'] =  well_obj_list[j].Q*perlen
        else:
            Run_output.loc[period,'W'+str(j)+'_Vin'] = well_obj_list[j].Q*perlen
            Run_output.loc[period,'W'+str(j)+'_Vout'] = 0  
            
        Run_output.loc[period,'W'+str(j)+'_T_sys_in'] = well_obj_list[j].T_inj
        Run_output.loc[period,'W'+str(j)+'_T_mf_out'] = well_obj_list[j].T_modflow
        Run_output.loc[period,'Dens_water_in'] = denseref + drhodT * well_obj_list[j].T_inj
        Run_output.loc[period,'Dens_water_out'] = denseref + drhodT * well_obj_list[j].T_modflow
    
    '''Density water influenced by Temperature'''                               # Update parameters based on rho_f per gridcell
    if WaterDensON ==1:
        grid_obj.rho_f = (1000 - ((grid_obj.temp-4)**2 / 207))                  # calculate updated water density for each cell       
        grid_obj.Kdist = Cp_s/(grid_obj.rho_f*Cp_f)                             # calculate updated Kdist for each cell (with updated rho_f
        for j in range (len(form_obj_list)):                                    # calculate updated Tdif for each layer (or Aqt or Aq)
            for k in range(int(form_obj_list[j].lbot - form_obj_list[j].ltop)):
                if form_obj_list[j].type == 'aquitard':
                    grid_obj.Tdif[k,:,:] =  kT_aqt / (PEFF*grid_obj.rho_f[k,:,:]*Cp_f) * 24 * 3600
                else:
                    grid_obj.Tdif[k,:,:] = kT_aq / (PEFF*grid_obj.rho_f[k,:,:]*Cp_f) * 24 * 3600 
   
    print (str(period)+' (of '+str(rl-1)+')') 
    h_obj.file.close()
    t_obj.file.close()

    elapsed_time = time.time() - start_time
    proxy = elapsed_time/60 * rl/(period+1) - elapsed_time/60
    print ('run time =', int(elapsed_time/60),' min')
    print ('expected remaining run time =', int(proxy),' min')


#Run_output_fin = Run_output.iloc[4:,]                                          # to correct for storage cycles, evaluation starts at month 4 (switch of the seasons
#Run_output_fin = Run_output_fin.reset_index()

end_time = time.time()
elapsed_time = (end_time- start_time) / 60
print ('Simulation completed, Runtime= '+str(round(elapsed_time,1))+' min') 

'''save run_output file'''
Run_output.to_csv(os.path.join(dirs[0])+'/Run_output__'+name+'.csv')
#Run_output_fin.to_csv(os.path.join(dirs[0])+'/Run_output_fin__'+name+'.csv')
#%%
''' [D] Post processing'''
lenly= int(rl/years)                                                            # length of last year (depends on the perlen), $$$ hier ging wat fout.
RecoveryEfficiencyW(Run_output,well_obj_list,sys_obj_list,rl,T_amb,nW,lenly,years)  # Recovery efficiency according to definition by Bloemendal&Hartog 2018
print(well_obj_list[0].TRE)                                                     # The Recovery Efficiency of a well
print(well_obj_list[0].TRE_ly)                                                  # The Recovery Efficiency of the last year of a well
print(well_obj_list[0].TRE_fy)                                                  # The Recovery Efficiency of the first year of a well

#RecoveryEfficiencyW_dens(Run_output,well_obj_list,sys_obj_list,rl,T_amb,nW,lenly,years) # correct recovery efficiency for changes in density. can give small changes at higher temperatures
#print(well_obj_list[0].TRE_D)                                                  
#print(well_obj_list[0].TRE_D_ly) 
#print(well_obj_list[0].TRE_D_fy) 

'''Geometric properties - analysis'''                                           # Check 'Bloemendal & Hartog 2018'
L_screen = (well_obj_list[0].stop_idx - well_obj_list[0].start_idx) * dz
yearlength = int(rl/years) 
V = sum(RES[rl-yearlength:rl,0])
RtoRth = (PEFF*Cw/Caq)**0.5

def AV_ratio(L,V):
    R_th = RtoRth* np.sqrt(V/(np.pi*L*PEFF))                                    # simplifiquation
    A_V = 2/L + 2/R_th
    return A_V
    
def L_Rth(L,V):
    L_Rth = L / (RtoRth* np.sqrt(V/(np.pi*L*PEFF)))
    return(L_Rth)
    
print('Efficiency= '+str(well_obj_list[0].TRE))
print('Efficiency LY= '+str(well_obj_list[0].TRE_ly))
a = AV_ratio(L_screen,V)
print('The A/V:',str(AV_ratio(L_screen,V)))
L_Rth(L_screen,V)
print('The L/Rth:',str(L_Rth(L_screen,V)))
print('Total Vin:',str(V))
print('ScreenLength:',str(L_screen))
print('Runtime= '+str(round(elapsed_time,1))+' min')

'''initialize contourlevels, plot boundaries, wells, layers, etc.'''
SY=0                                                                            # start year of simulation if applicable
Hlevel = np.linspace(-2,2, 100)                                                 # Head levels
Tlevel = np.linspace(Tmin-1, Tmax+1, 100)                                       # Temperature levels
T2level_w= [T_amb-0.5, T_amb+0.5, 25]                                           # levels for temperature contours
colors = ['r','b','k', 'g', 'c', 'y','grey', 'violet', 'dodgerblue','firebrick','coral','yellow', 'lightgreen', 'orange','firebrick','cyan']
if AXI == 0:
    X,Z=grid_obj.sumdelr + grid_obj.XGR[0] , grid_obj.botm[:,0,0] 
    X,Y=grid_obj.sumdelr + grid_obj.XGR[0] , grid_obj.YGR[0] - grid_obj.sumdelc 
else:
    X,Z=grid_obj.sumdelr, grid_obj.botm[:,0,0]

Z1 = int(np.around(np.average(well_obj_list[0].L)))
for i in range(len(grid_obj.sumdelr)):
    if grid_obj.sumdelr[i] < 300:                                               # right boundary of contour plots in m from well loaction
        a = i
    ResCol= int(a+1)

days = np.zeros(rl)
for i in range(rl):
    days[i] = perlen*(i+1)

OB = int(dmin_bound/dmin)*1.5                                                   # OutsideBox --> plot window / selection of UCN files
Wellb = np.zeros((len(well_obj_list),4))                                        # Define wells
for x in range(len(well_obj_list)):
    Wellb[x, 0]= well_obj_list[x].C - OB                                        # Left
    Wellb[x, 1]= well_obj_list[x].C + OB                                        # Right
    Wellb[x, 2]= well_obj_list[x].R - OB                                        # Down
    Wellb[x, 3]= well_obj_list[x].R + OB                                        # Up
Wellbmin =  Wellb.min(axis=0)
Wellbmax =  Wellb.max(axis=0)
Left  = int(Wellbmin[0])
Right = int(Wellbmax[1])
Down  = int(Wellbmin[2])
Up    = int(Wellbmax[3])
xw = [0,1]                                                                      # Define filter locations 
welltops = []                                                                   # creating a undefined list
wellbots = []
for i in range(nW):
    welltops.append(well_obj_list[i].ztop * np.ones(len(xw)))                   
    wellbots.append((well_obj_list[i].ztop - well_obj_list[i].FilterLength) * np.ones(len(xw)))

n_aqt = 0                                                                       # Define aquitards
n_aq = 0
if AXI == 1:
    xa = np.ones(len(X[0:ResCol]))
else:
    xa = np.ones(len(X[Left:Right]))
aqttops = []
aqtbots = []
aq_tops = []
aq_bots = []
aqtform_l = []
aqform_l = []

for i in range(nform):                                                          # Assigning which layers are aquitards and which aquifers
    if form_obj_list[i].type == 'aquitard':
        n_aqt += 1
        aqtform_l.append(i)
    else:
        n_aq += 1
        aqform_l.append(i)
for i in range(n_aqt):                                                          # Assigning the thicknesses of the aquifers and between which depth(z) they range
    j = aqtform_l[i]
    aqttops.append(form_obj_list[j].zTop * xa)
    aqtbots.append(form_obj_list[j].zBot * xa)
for i in range(n_aq):
    j = aqform_l[i]
    aq_tops.append(form_obj_list[j].zTop * xa)
    aq_bots.append(form_obj_list[j].zBot * xa)

#%%==============================================================================
''' [E] PLOT the figures'''
'''Temperature around the well''' 
plots = [rl-6,rl-1]                                                             # plot the times indicated in this array
for i in range(len(plots)):
    t=int(plots[i])
    year = SY + int(((t+1) * perlen) / 365)
    month = int((((t+1) * perlen) - (year * 365)) / 30 +1  )
    textstr = 'YY-MM= ' + str(int( year)) +'-'+str(month)
    
    h_obj = bf.HeadFile(os.path.join(dirs[0], name+str(t)+'.hds'))
    head = h_obj.get_data(totim=perlen)
    h_obj.close() 
    t_obj = bf.UcnFile(os.path.join(dirs[0], name+str(t)+'S1'+'.UCN'))
    temp1 = t_obj.get_data(totim=perlen)
    t_obj.close() 
    if AXI == 0:
        fig, (ax0, ax1)  = plt.subplots(ncols=2, sharey=True)
        im0 = ax0.contour(X[Left:Right], Y[Down:Up], temp1[Z1,Down:Up,Left:Right], 
                              colors=('black'), linestyles=('dotted'), linewidths=1,levels = T2level_w)#,label=True)
        im0 = ax0.contourf(X[Left:Right], Y[Down:Up], temp1[Z1,Down:Up,Left:Right], cmap='seismic', levels = Tlevel) #
        ax0.set_title('Temperature distribution [C] top of aquifer '+str((Z1)*dz)+' (m) depth ',fontsize=10)
        im0 = ax0.contourf(X[Left:Right], Y[Down:Up], temp1[int(np.average(well_obj_list[0].L)-1),Down:Up,Left:Right], cmap='seismic', levels = Tlevel) #
        fig.colorbar(im0, ax=ax0, format = '%1.0f')
        im1 = ax1.contourf(X[Left:Right], Y[Down:Up], head[max(well_obj_list[0].L-1),Down:Up,Left:Right], cmap='BrBG', levels = Hlevel) #
        ax1.set_title('Heads [m] in aquifer '+str((Z1)*dz)+' (m) depth ',fontsize=10)
        fig.colorbar(im1, ax=ax1, format = '%1.0f')
        for i in range(len(well_obj_list)):
            if well_obj_list[i].type == 'warm':
                ax0.plot(well_obj_list[i].xcor,well_obj_list[i].ycor,'o',c='k' ,markeredgecolor='k')
                ax1.plot(well_obj_list[i].xcor,well_obj_list[i].ycor,'o',c='k' ,markeredgecolor='k')
            else:
                ax0.plot(well_obj_list[i].xcor,well_obj_list[i].ycor, 'o', c='k',markeredgecolor='k')
                ax1.plot(well_obj_list[i].xcor,well_obj_list[i].ycor,'o',  c='k' ,markeredgecolor='k')
        for i in range(nmon):
            ax0.plot(mon_obj_list[i].x,mon_obj_list[i].y, 'o', c=colors[i+nW], label= str(mon_obj_list[i].Mon+1))
        props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        ax0.text(0.05, 0.975, textstr, transform=ax0.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax0.set_ylabel('y-coor [m]')
        ax0.set_xlabel('x-coor [m]')
        ax1.set_xlabel('x-coor [m]')
        plt.show()
        #plt.savefig(name + 'tempcontour-plan.png', dpi=300,bbox_inches='tight')
        
        fig, (ax0)  = plt.subplots(nrows=1)
        im0 = ax0.contourf(X[Left:Right], Z[:], temp1[:,well_obj_list[1].R,Left:Right], cmap='seismic', levels = Tlevel) #
        ax0.set_title('Cross section E-W: Temperature at some wells[C]',fontsize=10)
        cs = ax0.contour(X[Left:Right], Z[:],temp1[:,well_obj_list[1].R,Left:Right], colors=('black'), linestyles=('dotted'), linewidths=1,levels = T2level_w)
        plt.clabel(cs, fontsize = 9, inline = 1, color = 'k', fmt = '%1.1f')
        fig.colorbar(im0, ax=ax0, format = '%1.0f')
        ax0.set_ylabel('aquifer depth [m]')
        ax0.set_xlabel('x [m]')
        for i in range(n_aqt):
            ax0.plot(X[Left:Right], aqttops[i][:], color = 'grey', alpha = 0.6)
            ax0.plot(X[Left:Right], aqtbots[i][:], color = 'grey', alpha = 0.6)
            ax0.fill_between(X[Left:Right], aqttops[i][:], aqtbots[i][:], color = 'grey', alpha = 0.6)
        for i in range(nW):
            ax0.plot(well_obj_list[i].xcor + xw, welltops[i], color = 'k')  
            ax0.plot(well_obj_list[i].xcor + xw, wellbots[i], color = 'k')  
            ax0.fill_between(well_obj_list[i].xcor + xw, welltops[i], wellbots[i], color = 'k')
        props = dict(boxstyle='round', facecolor='w', alpha=0.5)
        ax0.text(0.05, 0.95, textstr, transform=ax0.transAxes, fontsize=10, verticalalignment='top', bbox=props)# place a text box in upper left in axes coords
        plt.show() 
#        fig.savefig(case_name+'Temp_vertical_warm-'+textstr+'.png', dpi=300) 

    else:
        if IsMono == 1:
            fig, (ax0)  = plt.subplots(nrows=1, sharex=True) 
            im0 = ax0.contour(X[0:ResCol], Z, temp1[:,0,0:ResCol],
                              colors=('black'), linestyles=('dotted'), linewidths=1,levels = T2level_w)#,label=True)
            ax0.clabel(im0, im0.levels,fmt = '%1.0f')#,fmt=fmt)#, fmt=fmt)
            im0 = ax0.contourf(X[0:ResCol], Z, temp1[:,0,0:ResCol],cmap='seismic', levels = Tlevel) #X[0:ResCol], Z, 
            fig.colorbar(im0, ax=ax0, format = '%1.0f')
            for i in range(nW):
                ax0.plot(xw, welltops[i], color = 'k') 
                ax0.plot(xw, wellbots[i], color = 'k')
                ax0.fill_between(xw, welltops[i], wellbots[i], color = 'k')
            for i in range(n_aqt):
                ax0.plot(X[0:ResCol], aqttops[i], color = 'grey', alpha = 0.6)
                ax0.plot(X[0:ResCol], aqtbots[i], color = 'grey', alpha = 0.6)
                ax0.fill_between(X[0:ResCol], aqttops[i], aqtbots[i], color = 'grey', alpha = 0.6)
            ax0.set_title('Temperature [C] of '+well_obj_list[0].type+'-well')#, day:'+str((t) * perlen))
            ax0.set_ylabel('aquifer depth [m]')
            eff = round(well_obj_list[0].TRE_ly, 2)
            props = dict(boxstyle='round', facecolor='w', alpha=0.5)
            ax0.text(0.05, 0.95, textstr, transform=ax0.transAxes, fontsize=10, verticalalignment='top', bbox=props)
            plt.xlabel('distance from well location [m]')
            plt.show()
#            fig.savefig('warmWell '+str(name)+'.png', dpi=300) 
            
        if IsMono == 0:
            fig, (ax2, ax3)  = plt.subplots(nrows=2, sharex=True) 
            im2 = ax2.contour(X[0:ResCol], Z, temp1[:,0,0:ResCol],
                              colors=('black'), linestyles=('dotted'), linewidths=1,levels = T2level_w)
            ax2.clabel(im2, im2.levels,fmt = '%1.1f')
            im2 = ax2.contourf(X[0:ResCol], Z, temp1[:,0,0:ResCol],cmap='seismic', levels = Tlevel) 
            fig.colorbar(im2, ax=ax2, format = '%1.0f')
            for i in range(nW):
                ax2.plot(xw, welltops[i], color = 'k') 
                ax2.plot(xw, wellbots[i], color = 'k')
                ax2.fill_between(xw, welltops[i], wellbots[i], color = 'k')
            for i in range(n_aqt):
                ax2.plot(X[0:ResCol], aqttops[i], color = 'grey', alpha = 0.6)
                ax2.plot(X[0:ResCol], aqtbots[i], color = 'grey', alpha = 0.6)
                ax2.fill_between(X[0:ResCol], aqttops[i], aqtbots[i], color = 'grey', alpha = 0.6)
            ax2.set_title('Temperature [C] of '+well_obj_list[0].type+'-well')
            ax2.set_ylabel('aquifer depth [m]')
            plt.xlabel('distance from well location [m]')
            props = dict(boxstyle='round', facecolor='w', alpha=0.5)
            ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)
            plt.show()
            im3 = ax3.contour(X[0:ResCol], Z, temp1[:,2,0:ResCol],
                              colors=('black'), linestyles=('dotted'), linewidths=1,levels = T2level_w)
            ax3.clabel(im3, im3.levels,fmt = '%1.1f'), 
            im3 = ax3.contourf(X[0:ResCol], Z, temp1[:,2,0:ResCol],cmap='seismic', levels = Tlevel) 
            fig.colorbar(im3, ax=ax3, format = '%1.0f')
            for i in range(nW):
                ax3.plot(xw, welltops[i], color = 'k') 
                ax3.plot(xw, wellbots[i], color = 'k')
                ax3.fill_between(xw, welltops[i], wellbots[i], color = 'k')
            for i in range(n_aqt):
                ax3.plot(X[0:ResCol], aqttops[i], color = 'grey', alpha = 0.6)
                ax3.plot(X[0:ResCol], aqtbots[i], color = 'grey', alpha = 0.6)
                ax3.fill_between(X[0:ResCol], aqttops[i], aqtbots[i], color = 'grey', alpha = 0.6)
            ax3.set_title('Temperature [C] of '+well_obj_list[1].type+'-well')
            ax3.set_ylabel('aquifer depth [m]')
            plt.xlabel('distance from well location [m]')
            props = dict(boxstyle='round', facecolor='w', alpha=0.5)
            ax3.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)
            plt.show()
#            fig.savefig(str(name)+'.png', dpi=300) 

''' Temperatures and discharge'''
fig, (ax0, ax1)  = plt.subplots(nrows=2, sharex=True)
ax0.set_title('Discharge of wells',fontsize=10)
for i in range(nW):
    im0 = ax0.plot(days,Run_output.loc[:,'W'+str(i)+'_Vin'] + Run_output.loc[:,'W'+str(i)+'_Vout'], color=colors[i], label=well_obj_list[i].type),
ax0.set_ylabel('well discharge [m3/month]',fontsize=10)
ax0.set_facecolor('lightgrey')
plt.grid(True, 'major', lw=1, c='w')
plt.show()   
plt.legend() 

ax1.set_title('Temperature  wells [C]',fontsize=10)
for i in range(nW):
    im1 = ax1.plot(days,Run_output.loc[:,'W'+str(i)+'_T_mf_out'], color=colors[i], label=well_obj_list[i].type),
ax1.set_ylabel('Temperature filter screens [C]',fontsize=10) 
ax1.set_facecolor('lightgrey')
plt.grid(True, 'major', lw=1, c='w') 
plt.show() 
plt.legend() 
#fig.savefig(str(name) + '_well_Q&T.png', dpi=300)

''' monitoring points '''
fig, (ax2)  = plt.subplots(nrows=1, sharex=True)
ax2.set_title('Temperatures monitoring points',fontsize=10) 
ax2.set_xlabel('time [d]',fontsize=10)
for i in range(nmon):
    im2 = ax2.plot(days,RES[:,i], color=colors[i+nW], label= str(mon_obj_list[i].Mon+1))
ax2.set_ylabel('Temperature monitoring point [C]',fontsize=10)  
ax2.set_facecolor('lightgrey')
plt.grid(True, 'major', lw=1, c='w')
plt.show()   
plt.legend() 
#fig.savefig(str(name) + '_Temp_monitoring_points.png', dpi=300)

'''SOIL PROFILE'''                                                              # Just the soil profile without temperature distribution
if AXI == 1:
    fig, (ax0)  = plt.subplots(nrows=1, sharex=True)
    for i in range(n_aqt):
        ax0.plot(X[0:ResCol], aqttops[i], color = 'grey', alpha = 0.6)
        ax0.plot(X[0:ResCol], aqtbots[i], color = 'grey', alpha = 0.6)
        ax0.fill_between(X[0:ResCol], aqttops[i], aqtbots[i], color = 'grey', alpha = 0.6)
    for i in range(n_aq):
        ax0.plot(X[0:ResCol], aq_tops[i], color = 'yellow', alpha = 0.6)
        ax0.plot(X[0:ResCol], aq_bots[i], color = 'yellow', alpha = 0.6)
        ax0.fill_between(X[0:ResCol], aq_tops[i], aq_bots[i], color = 'yellow', alpha = 0.6)
    for i in range(nW):
        ax0.plot(xw, welltops[i], color = 'k')  
        ax0.plot(xw, wellbots[i], color = 'k')  
        ax0.fill_between(xw, welltops[i], wellbots[i], color = 'k')
    dots=list()
    for i in range(nmon):
        dot=ax0.plot(mon_obj_list[i].x, mon_obj_list[i].z, 'o-',markeredgecolor = 'k',color = colors[i], label= 'Obs'+str(mon_obj_list[i].Mon)+' R='+str(round(mon_obj_list[i].x,0))+', Depth='+str(mon_obj_list[i].z))
        dots= dots+dot
    ax0.set_ylabel('Depth [m]')
    ax0.set_title(label = 'Soil Profile from input file:' + excel_filename)
    legend_color1 = mpatches.Patch(color='yellow', label='Aquifer')
    legend_color2 = mpatches.Patch(color='grey', label='Aquitard')
    ax0.legend(title = 'Soil types', frameon = True, loc = 'best',  handles=[legend_color1, legend_color2])
    plt.show()
else:
    fig, (ax0)  = plt.subplots(nrows=1, sharex=True)
    for i in range(n_aqt):
        ax0.plot(X[Left:Right], aqttops[i][:], color = 'grey', alpha = 0.6)
        ax0.plot(X[Left:Right], aqtbots[i][:], color = 'grey', alpha = 0.6)
        ax0.fill_between(X[Left:Right], aqttops[i][:], aqtbots[i][:], color = 'grey', alpha = 0.6)
    for i in range(n_aq):
        ax0.plot(X[Left:Right], aq_tops[i][:], color = 'yellow', alpha = 0.6)
        ax0.plot(X[Left:Right], aq_bots[i][:], color = 'yellow', alpha = 0.6)
        ax0.fill_between(X[Left:Right], aq_tops[i][:], aq_bots[i][:], color = 'yellow', alpha = 0.6)
    for i in range(nW):
        ax0.plot(well_obj_list[i].xcor + xw, welltops[i], color = 'k')  
        ax0.plot(well_obj_list[i].xcor + xw, wellbots[i], color = 'k')  
        ax0.fill_between(well_obj_list[i].xcor + xw, welltops[i], wellbots[i], color = 'k')
    dots=list()
    for i in range(nmon):
        dot=ax0.plot(mon_obj_list[i].x, mon_obj_list[i].z, 'o-',markeredgecolor = 'k',color = colors[i], label= 'Obs'+str(mon_obj_list[i].Mon)+' R='+str(round(mon_obj_list[i].x,0))+', Depth='+str(mon_obj_list[i].z))
        dots= dots+dot
    ax0.set_ylabel('Depth [m]')
    ax0.set_title(label = 'Soil Profile from input file:' + excel_filename)
    legend_color1 = mpatches.Patch(color='yellow', label='Aquifer')
    legend_color2 = mpatches.Patch(color='grey', label='Aquitard')
    ax0.legend(title = 'Soil types', frameon = True, loc = 'best',  handles=[legend_color1, legend_color2])
    plt.show()
    #fig.savefig(str(name) + '_soilProfile.png', dpi=300)