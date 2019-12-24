 
'''
Grid methods and functions for the Flopy/NetLogo coupled sandbox model
'''
 
import numpy as np
from scipy import interpolate
 
 
class PyGrid(object):
    '''
    Instantiate a Python grid object
    '''
     
    def __init__(self):
     
        self.XGR = []
        self.YGR = []
        self.ncol = 0
        self.nrow = 0
        self.nlay = 0
        self.delr = []
        self.delc = []
        self.IBOUND = []
        self.ICBUND = []
        self.head = np.array([np.ones((1,1))])
        self.temp = np.array([np.ones((1,1))])
        self.salinity = np.array([np.ones((1,1))])
#        self.HK
#        self.VK
#        self.PEFF

        
    def make_grid_Lin(self, well_obj_list, ztop, zbot, aroundAll=500, dmin=5, dmax=20, dz=5,dmin_bound =50, nstep=2, grid_extents=None, IsMono=0,ICBUND_Up=-1,ICBUND_Down=-1):
        '''
        Update the properties of a grid object - based on makegrid.m by Ruben Calje, 
        Theo Olsthoorn and Mark van der Valk as implemented in mfLab
         
        :param well_obj_list: list of Python well objects
        :param aroundAll: extra grid allowance around the "bounding box" of well coordinates
        :param dmin: target for minimum grid cell size
        :param dmax: target for maximum grid cell size
        :param nstep: refinement factor for grid cells around wells
        :param grid_extents: list of coordinates in the format [min_x, max_x, min_y, max_y]
                             If omitted, grid extents are calculated dynamically based on well coordinates
                             and aroundAll parameter
        '''
        
        '''
        THIS GRID FUNCTION MAKES AND LINEAR GRID INSIDE WITH A LOGARITHMIC LARGE GRID SIZE AROUND IT
        '''
        wells_xy = np.array([[i.xcor, i.ycor] for i in well_obj_list]) 
         
        xw = np.ceil(wells_xy[:,0] / dmin) * dmin
        yw = np.ceil(wells_xy[:,1] / dmin) * dmin
         
        if grid_extents: 
            min_x = grid_extents[0]
            max_x = grid_extents[1]
            min_y = grid_extents[2]
            max_y = grid_extents[3]
        else:
            min_x = np.min(xw - dmin_bound)
            max_x = np.max(xw + dmin_bound)
            min_y = np.min(yw - dmin_bound)
            max_y = np.max(yw + dmin_bound)      
         
        XGR = np.arange(min_x, max_x, dmin)
        YGR = np.arange(min_y, max_y, dmin)
        
        #PREPEND & APPEND
        dx = np.logspace(np.log10(dmin), np.log10(dmax),nstep)
        dxtot = max(np.cumsum(dx))
        
        if dxtot < aroundAll: #number of cells around the small sized grid
            extracells = int((aroundAll - dxtot)/dmax) +1
        else:
            extracells = 0
        
        extracellsL = [dmax]*extracells
        
        dx_TOT = np.append(dx,extracellsL) #All cells that need to be added outside the linear part. 
        dx_TOT = np.cumsum(dx_TOT)
        dy_TOT=dx_TOT
        
        xprep = np.fliplr([(dx_TOT*-1) +  min(XGR)])[0]
        xapp = dx_TOT + max(XGR)

        yprep = np.fliplr([(dy_TOT*-1) +  min(YGR)])[0]
        yapp = dy_TOT + max(YGR)
        
        XGR = np.append(xprep,XGR)
        XGR = np.append(XGR,xapp)
        YGR = np.append(yprep,YGR)
        YGR = np.append(YGR,yapp)      

#        d = np.cumsum(np.append(dx[0] / 2, dx[1:len(dx)]))
#        L = d[-1]
#        subgrid = np.append(-d[::-1], d)
#         
#        for iW in range(len(wells_xy)):
#            XGR = XGR[(XGR < wells_xy[iW,0] - L) | (XGR > wells_xy[iW,0] + L)]
#            YGR = YGR[(YGR < wells_xy[iW,1] - L) | (YGR > wells_xy[iW,1] + L)]
#             
#        Nx = len(XGR);
#        Ny = len(YGR);
#        Ns = len(subgrid);
#        Nw = len(wells_xy);
#         
#        XGR = np.append(XGR, np.zeros(Nw*Ns))
#        YGR = np.append(YGR, np.zeros(Nw*Ns))
#         
#        for iW in range(len(wells_xy)):
#            XGR[Nx + iW*Ns + np.arange(0,Ns)] = wells_xy[iW,0] + subgrid;
#            YGR[Ny + iW*Ns + np.arange(0,Ns)] = wells_xy[iW,1] + subgrid;
         
        #XGR, YGR: 1D arrays of cell coordinates (respectively columns and rows)
        self.XGR = XGR#cleangrid(np.unique(np.around(XGR*100)/100), dmin)
        self.YGR = YGR[::-1]#cleangrid(np.unique(np.around(YGR*100)/100), dmin)[::-1]

        self.ncol = len(self.XGR) - 1 #Number of grid columns
        self.delr = np.diff(self.XGR) #Width of each column
        self.nrow = len(self.YGR) - 1 #Number of grid rows
        self.delc = -np.diff(self.YGR) #Height of each row

        self.top = ztop * np.ones([self.nrow, self.ncol])
        botm_range = np.arange(zbot, ztop, dz)[::-1]
        botm_2d = np.ones([self.nrow, self.ncol])
        self.botm = botm_2d*botm_range[:, None, None]
        self.nlay = len(botm_range)
        self.sumdelr = np.cumsum(self.delr)
        self.sumdelc = np.cumsum(self.delc)

        self.IBOUND, self.ICBUND = boundaries(self) #Create grid boundaries

        if ICBUND_Up == 0 and ICBUND_Down == 0:
            self.ICBUND[-1,:,:] = 1# -1 # The upper boundary; comment to have no flow boundary
            self.ICBUND[0,:,:] = 1
        
        if ICBUND_Up == -1 and ICBUND_Down == 0:
            self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
            self.ICBUND[0,:,:] = 1
        
        if ICBUND_Up == 0 and ICBUND_Down == -1:
            self.ICBUND[-1,:,:] = 1# -1 # The upper boundary; comment to have no flow boundary
            self.ICBUND[0,:,:] = -1    
        
        else:
            self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
            self.ICBUND[0,:,:] = -1
  


    def make_grid_Log(self, well_obj_list, ztop, zbot, aroundAll=500, dmin=5,dmin_bound =50, dmax=20, dz=5, nstep=2, grid_extents=None, IsMono=0,ICBUND_Up=-1,ICBUND_Down=-1):
        '''
        Update the properties of a grid object - based on makegrid.m by Ruben Calje, 
        Theo Olsthoorn and Mark van der Valk as implemented in mfLab
         
        :param well_obj_list: list of Python well objects
        :param aroundAll: extra grid allowance around the "bounding box" of well coordinates
        :param dmin: target for minimum grid cell size
        :param dmax: target for maximum grid cell size
        :param nstep: refinement factor for grid cells around wells
        :param grid_extents: list of coordinates in the format [min_x, max_x, min_y, max_y]
                             If omitted, grid extents are calculated dynamically based on well coordinates
                             and aroundAll parameter
        '''
         
        wells_xy = np.array([[i.xcor, i.ycor] for i in well_obj_list]) 
         
        xw = np.ceil(wells_xy[:,0] / dmin) * dmin
        yw = np.ceil(wells_xy[:,1] / dmin) * dmin
         
        if grid_extents: 
            min_x = grid_extents[0]
            max_x = grid_extents[1]
            min_y = grid_extents[2]
            max_y = grid_extents[3]
        else:
            min_x = np.min(xw - aroundAll)
            max_x = np.max(xw + aroundAll)
            min_y = np.min(yw - aroundAll)
            max_y = np.max(yw + aroundAll)      
         
        XGR = np.arange(min_x, max_x + dmin, dmax)
        YGR = np.arange(min_y, max_y + dmin, dmax)
         
        dx = np.logspace(np.log10(dmin), np.log10(dmax),nstep)
        d = np.cumsum(np.append(dx[0] / 2, dx[1:len(dx)]))
        L = d[-1]
        subgrid = np.append(-d[::-1], d)
         
        for iW in range(len(wells_xy)):
            XGR = XGR[(XGR < wells_xy[iW,0] - L) | (XGR > wells_xy[iW,0] + L)]
            YGR = YGR[(YGR < wells_xy[iW,1] - L) | (YGR > wells_xy[iW,1] + L)]
             
        Nx = len(XGR);
        Ny = len(YGR);
        Ns = len(subgrid);
        Nw = len(wells_xy);
         
        XGR = np.append(XGR, np.zeros(Nw*Ns))
        YGR = np.append(YGR, np.zeros(Nw*Ns))
         
        for iW in range(len(wells_xy)):
            XGR[Nx + iW*Ns + np.arange(0,Ns)] = wells_xy[iW,0] + subgrid;
            YGR[Ny + iW*Ns + np.arange(0,Ns)] = wells_xy[iW,1] + subgrid;
         
        #XGR, YGR: 1D arrays of cell coordinates (respectively columns and rows)
        self.XGR = cleangrid(np.unique(np.around(XGR*100)/100), dmin)
        self.YGR = cleangrid(np.unique(np.around(YGR*100)/100), dmin)[::-1]

        self.ncol = len(self.XGR) - 1 #Number of grid columns
        self.delr = np.diff(self.XGR) #Width of each column
        self.nrow = len(self.YGR) - 1 #Number of grid rows
        self.delc = -np.diff(self.YGR) #Height of each row

        self.top = ztop * np.ones([self.nrow, self.ncol])
        botm_range = np.arange(zbot, ztop, dz)[::-1]
        botm_2d = np.ones([self.nrow, self.ncol])
        self.botm = botm_2d*botm_range[:, None, None]
        self.nlay = len(botm_range)
        self.sumdelr = np.cumsum(self.delr)
        self.sumdelc = np.cumsum(self.delc)

        self.IBOUND, self.ICBUND = boundaries(self) #Create grid boundaries

        if ICBUND_Up == 0 and ICBUND_Down == 0:
            self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
            self.ICBUND[0,:,:] = 0
        
        if ICBUND_Up == -1 and ICBUND_Down == 0:
            self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
            self.ICBUND[0,:,:] = 0
        
        if ICBUND_Up == 0 and ICBUND_Down == -1:
            self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
            self.ICBUND[0,:,:] = -1    
        
        else:
            self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
            self.ICBUND[0,:,:] = -1
        #-1  # The Upper boundary; comment to have no flow boundary   
      

       
class PyGridAxi(object):
    '''
    Instantiate a Python axisymmetric grid object
    '''
     
    def __init__(self):
     
#        self.XGR = []
#        self.YGR = []
        self.ncol = 0
        self.nrow = 0
        self.nlay = 0
        self.delr = []
        self.delc = []
        self.IBOUND = []
        self.ICBUND = []
        self.head = np.array([np.ones((1,1))])
        self.temp = np.array([np.ones((1,1))])
        self.salinity = np.array([np.ones((1,1))])
#        self.IsMono
#        self.HK
#        self.VK
#        self.PEFF

    def make_grid_axi_Lin(self, well_obj_list, ztop, zbot, aroundAll=500, dmin=1, dmax=200, dz=5,dmin_bound =50, nstep=20, grid_extents=None, IsMono=1,ICBUND_Up=-1,ICBUND_Down=-1):
        
        if IsMono == 1:
            
        #Axisymmetric grid for a Monowell
            self.nrow = 1
            
            lincells = [dmin]*int(dmin_bound/dmin)
                        
            startsize=np.log10(dmin)
            endsize=np.log10(dmax)
            for j in range(1000):    # iterate to number of columns required to meet model extend. stating at minimum indicated number of steps 'nstep'
                delr = np.logspace(startsize,endsize,num=(nstep+j),base=10.)
                self.delr = np.append(lincells,delr)
                self.sumdelr = np.cumsum(self.delr)
                self.sumdelc = np.cumsum(self.delc)
                if (self.sumdelr[-1]-self.delr[-1])>=aroundAll:   # last column is boundary, so the real modelling environment needs to be inside the boundary
                    break
        
            self.ncol = nstep+j+len(lincells)    
            self.Lx = np.sum(self.delr)
            self.delc=np.ones(1)
            self.top = ztop * np.ones([self.nrow, self.ncol])
            botm_range = np.arange(zbot, ztop, dz)[::-1]
            botm_2d = np.ones([self.nrow, self.ncol])
            self.botm = botm_2d*botm_range[:, None, None]
            self.nlay = len(botm_range)
            
            ib = np.ones((self.nlay, self.nrow, self.ncol))
            ib[:,:,-1] = -1
#            ib[0,:,:]=-1
            self.IBOUND = ib
            self.ICBUND = self.IBOUND
            
            if ICBUND_Up == 0 and ICBUND_Down == 0:
                self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = 0
            
            if ICBUND_Up == -1 and ICBUND_Down == 0:
                self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = 0
            
            if ICBUND_Up == 0 and ICBUND_Down == -1:
                self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = -1    
            
            else:
                self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = -1
            #-1  # The Upper boundary; comment to have no flow boundary   


        if IsMono == 0:
            
        #Axisymmetric grid Doublet
            self.nrow = 3
        
            lincells = [dmin]*int(dmin_bound/dmin)
                        
            startsize=np.log10(dmin)
            endsize=np.log10(dmax)
            for j in range(1000):    # iterate to number of columns required to meet model extend. stating at minimum indicated number of steps 'nstep'
                delr = np.logspace(startsize,endsize,num=(nstep+j),base=10.)
                self.delr = np.append(lincells,delr)
                self.sumdelr = np.cumsum(self.delr)
                self.sumdelc = np.cumsum(self.delc)
                if (self.sumdelr[-1]-self.delr[-1])>=aroundAll:   # last column is boundary, so the real modelling environment needs to be inside the boundary
                    break
        
            self.ncol = nstep+j+len(lincells)
            self.Lx = np.sum(self.delr)
            self.delc=np.ones([3])
        
            self.top = ztop * np.ones([self.nrow, self.ncol])
            botm_range = np.arange(zbot, ztop, dz)[::-1]
            botm_2d = np.ones([self.nrow, self.ncol])
            self.botm = botm_2d*botm_range[:, None, None]
            self.nlay = len(botm_range)
        
            ib = np.ones((self.nlay, self.nrow, self.ncol))
            ib[:,:,-1] = -1
#            ib[0,:,:]=-1
            ib[:,1,:]=0.
            self.IBOUND = ib
            self.ICBUND = self.IBOUND
            
            if ICBUND_Up == 0 and ICBUND_Down == 0:
                self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = 0
            
            if ICBUND_Up == -1 and ICBUND_Down == 0:
                self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = 0
            
            if ICBUND_Up == 0 and ICBUND_Down == -1:
                self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = -1    
            
            else:
                self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = -1
            #-1  # The Upper boundary; comment to have no flow boundary   

    def make_grid_axi_Log(self, well_obj_list, ztop, zbot, aroundAll=500, dmin=1, dmax=200, dz=5,dmin_bound =50, nstep=20, grid_extents=None, IsMono=1,ICBUND_Up=-1,ICBUND_Down=-1):
        
        if IsMono == 1:
            
        #Axisymmetric grid for a Monowell
            self.nrow = 1
        
            startsize=np.log10(dmin)
            endsize=np.log10(dmax)
            for j in range(1000):    # iterate to number of columns required to meet model extend. stating at minimum indicated number of steps 'nstep'
                self.delr = np.logspace(startsize,endsize,num=(nstep+j),base=10.)
                self.sumdelr = np.cumsum(self.delr)
                self.sumdelc = np.cumsum(self.delc)
                if (self.sumdelr[-1]-self.delr[-1])>=aroundAll:   # last column is boundary, so the real modelling environment needs to inside the boundary
                    break
        
            self.ncol = nstep+j    
            self.Lx = np.sum(self.delr)
            self.delc=np.ones(1)
            self.top = ztop * np.ones([self.nrow, self.ncol])
            botm_range = np.arange(zbot, ztop, dz)[::-1]
            botm_2d = np.ones([self.nrow, self.ncol])
            self.botm = botm_2d*botm_range[:, None, None]
            self.nlay = len(botm_range)
            
            ib = np.ones((self.nlay, self.nrow, self.ncol))
            ib[:,:,-1] = -1
#            ib[0,:,:]=-1
            self.IBOUND = ib
            self.ICBUND = self.IBOUND
            
            if ICBUND_Up == 0 and ICBUND_Down == 0:
                self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = 0
            
            if ICBUND_Up == -1 and ICBUND_Down == 0:
                self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = 0
            
            if ICBUND_Up == 0 and ICBUND_Down == -1:
                self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = -1    
            
            else:
                self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = -1
            #-1  # The Upper boundary; comment to have no flow boundary   


        if IsMono == 0:
            
        #Axisymmetric grid Doublet
            self.nrow = 3
        
            startsize=np.log10(dmin)
            endsize=np.log10(dmax)
            for j in range(1000):
                self.delr = np.logspace(startsize,endsize,num=(nstep+j),base=10.)
                self.sumdelr = np.cumsum(self.delr)
                if self.sumdelr[-1]-self.delr[-1]>=aroundAll: 
                    break
        
            self.ncol = nstep+j   
            self.Lx = np.sum(self.delr)
            self.delc=np.ones([3])
        
            self.top = ztop * np.ones([self.nrow, self.ncol])
            botm_range = np.arange(zbot, ztop, dz)[::-1]
            botm_2d = np.ones([self.nrow, self.ncol])
            self.botm = botm_2d*botm_range[:, None, None]
            self.nlay = len(botm_range)
        
            ib = np.ones((self.nlay, self.nrow, self.ncol))
            ib[:,:,-1] = -1
#            ib[0,:,:]=-1
            ib[:,1,:]=0.
            self.IBOUND = ib
            self.ICBUND = self.IBOUND
            
            if ICBUND_Up == 0 and ICBUND_Down == 0:
                self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = 0
            
            if ICBUND_Up == -1 and ICBUND_Down == 0:
                self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = 0
            
            if ICBUND_Up == 0 and ICBUND_Down == -1:
                self.ICBUND[-1,:,:] = 0# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = -1    
            
            else:
                self.ICBUND[-1,:,:] = -1# -1 # The upper boundary; comment to have no flow boundary
                self.ICBUND[0,:,:] = -1
            #-1  # The Upper boundary; comment to have no flow boundary   

        
            
def set_geo_prop(grid_obj, form_obj_list,  gwflow_x=0, gwflow_y=0, dz=1, AqL=2, Tdif_aqt = 0.108370, Tdif_aq=0.108370,rho_b=2148,rho_f=1000,Cp_f=4183,Kdist = 0.00016973464021037533):
    grid_obj.init_grid = np.ones((grid_obj.nlay, grid_obj.nrow, grid_obj.ncol)) #Just to size the grid
    
    if grid_obj.__class__ == PyGridAxi:
# Axisymmetric geo properties
        init_axi_grid = grid_obj.init_grid
        for i in range(len(grid_obj.delr)):
            init_axi_grid[:,:,i] =  2 * np.pi * (grid_obj.sumdelr[i] - 0.5 * grid_obj.delr[i])

# Initial arrays for grid values
        grid_obj.head = 0 * grid_obj.init_grid #Zero initial head - no groundwater flow
        grid_obj.temp = 1 * grid_obj.init_grid #10 degrees
        grid_obj.salinity = 1. * grid_obj.init_grid #initial salinity            
        grid_obj.HK =  1 * grid_obj.init_grid
        grid_obj.VK = 1 * grid_obj.init_grid
        grid_obj.PEFF = 1 * grid_obj.init_grid
        grid_obj.ss = 1 * grid_obj.init_grid
        grid_obj.rho_b = 1 * grid_obj.init_grid
        grid_obj.Tdif = 1* grid_obj.init_grid
        grid_obj.Kdist = 1*grid_obj.init_grid
        grid_obj.rho_f = 1*grid_obj.init_grid
# Assign geo and initial properties to the grid.        
        for j in range (len(form_obj_list)):
            form_obj_list[j].lbot = form_obj_list[j].zBot / -dz
            if j == 0:
                form_obj_list[j].ltop = 0
            else:
                form_obj_list[j].ltop = form_obj_list[j-1].zBot / -dz
        
        for j in range (len(form_obj_list)):
            for k in range(int(form_obj_list[j].lbot - form_obj_list[j].ltop)):
                k += int(form_obj_list[j].ltop)
                grid_obj.HK[k,:,:] = init_axi_grid[k,:,:] * form_obj_list[j].Kh
                grid_obj.VK[k,:,:] = init_axi_grid[k,:,:] * form_obj_list[j].Kv
                grid_obj.PEFF[k,:,:] = init_axi_grid[k,:,:] * form_obj_list[j].por
                grid_obj.ss[k,:,:] = init_axi_grid[k,:,:] * form_obj_list[j].ss
                grid_obj.rho_b[k,:,:] = init_axi_grid[k,:,:] * rho_b
                if form_obj_list[j].type == 'aquitard':
                    grid_obj.Tdif[k,:,:] =  Tdif_aqt 
                else:
                    grid_obj.Tdif[k,:,:] = Tdif_aq 
                grid_obj.temp[k,:,:] =  form_obj_list[j].s1
                grid_obj.salinity[k,:,:] =  form_obj_list[j].s2
                grid_obj.Kdist = Kdist
                grid_obj.rho_f = rho_f
                
                '''aanvullen met update bij AXI !! '''
    if grid_obj.__class__ == PyGrid:
        
# Initial arrays for grid values        
        grid_obj.HK = 1 * grid_obj.init_grid
        grid_obj.VK = 1 * grid_obj.init_grid
        grid_obj.PEFF = 1 * grid_obj.init_grid 
        grid_obj.ss = 1 * grid_obj.init_grid
        grid_obj.head = 1 * grid_obj.init_grid
        grid_obj.temp = 1 * grid_obj.init_grid #10 degrees
        grid_obj.salinity = 1 * grid_obj.init_grid #initial salinity
        grid_obj.gwflow_x = gwflow_x
        grid_obj.gwflow_y = gwflow_y
        grid_obj.rho_b = 1 * grid_obj.init_grid
        grid_obj.Tdif = 1* grid_obj.init_grid
        grid_obj.Kdist = 1*grid_obj.init_grid
        grid_obj.rho_f = 1*grid_obj.init_grid
        
# Assign geo and initial properties to the grid.        
        for j in range (len(form_obj_list)):
            form_obj_list[j].lbot = form_obj_list[j].zBot / -dz
            if j == 0:
                form_obj_list[j].ltop = 0
            else:
                form_obj_list[j].ltop = form_obj_list[j-1].zBot / -dz
        
        for j in range (len(form_obj_list)):
            for k in range(int(form_obj_list[j].lbot - form_obj_list[j].ltop)):
                k += int(form_obj_list[j].ltop)
                grid_obj.HK[k,:,:] = form_obj_list[j].Kh
                grid_obj.VK[k,:,:] = form_obj_list[j].Kv
                grid_obj.PEFF[k,:,:] = form_obj_list[j].por
                grid_obj.ss[k,:,:] = form_obj_list[j].ss       
                grid_obj.temp[k,:,:] = grid_obj.init_grid[k,:,:] * form_obj_list[j].s1
                grid_obj.salinity[k,:,:] = grid_obj.init_grid[k,:,:] * form_obj_list[j].s2
       
#Set initial heads according to groundwater flow (based on mfLab Utrecht model)
        PEFF = form_obj_list[AqL].por
        HK = form_obj_list[AqL].Kh
        y_array = np.array([(grid_obj.YGR[:-1] - np.mean(grid_obj.YGR[:-1])) * PEFF * -grid_obj.gwflow_y / 365 / HK])
        y_tile = np.array([np.tile(y_array.T, (1, grid_obj.ncol))])
        x_array = (grid_obj.XGR[:-1] - np.mean(grid_obj.XGR[:-1])) * PEFF * -grid_obj.gwflow_x / 365 / HK
        y_tile += x_array
        grid_obj.head = np.tile(y_tile, (grid_obj.nlay, 1, 1))
        
        for j in range (len(form_obj_list)):
            for k in range(int(form_obj_list[j].lbot - form_obj_list[j].ltop)):
                if form_obj_list[j].type == 'aquitard':
                    grid_obj.Tdif[k,:,:] =  Tdif_aqt 
                else:
                    grid_obj.Tdif[k,:,:] = Tdif_aq
        
        grid_obj.temp[k,:,:] =  form_obj_list[j].s1
        grid_obj.salinity[k,:,:] =  form_obj_list[j].s2
        grid_obj.Kdist = Kdist
        grid_obj.rho_f = rho_f   

def set_Tboundary(grid_obj, perlen=30, run_length = 12, Tmin=5, Tmax=20, startwinter = 1):
#    SumSine = 0
    PerPerYear= int(round(365/perlen, 0))
    grid_obj.SurfaceT=np.zeros(run_length)
    PPY=int(PerPerYear/2)
    grid_obj.Tmax = Tmax
    grid_obj.Tmin = Tmin
#    for i in range(PPY):
#        Sine = np.sin(np.pi*(i)/PPY)
#        SumSine += Sine
    for i in range(run_length):
        if startwinter == 1:
            grid_obj.SurfaceT[i] = (-np.cos(np.pi * i/PPY)) * (Tmax - Tmin)/2 + (Tmax - Tmin)/2 + Tmin
        else:
            grid_obj.SurfaceT[i] = (np.cos(np.pi * i/PPY)) * (Tmax - Tmin)/2 + (Tmax - Tmin)/2 + Tmin
#    plt.plot(grid_obj.SurfaceT)        
    
def init_monitoring(grid_obj, mon_obj_list, dz, nmon, AXI, IsMono):
    mon_LRC_list = np.zeros((nmon,3))
    for i in range(nmon):
        mon_LRC_list[i,0] = int(mon_obj_list[i].z / -dz)
        #mon_LRC_list[i,2] = (mon_obj_list[i].x) #/...?)
        
        if AXI == 0:
            mon_LRC_list[i,2] = int(np.nonzero(grid_obj.XGR > mon_obj_list[i].x)[0][0]) - 1 
            mon_LRC_list[i,1] = int(len(grid_obj.YGR) - np.nonzero(grid_obj.YGR[::-1] > mon_obj_list[i].y)[0][0]) - 1
        else: 
            mon_LRC_list[i,1] = 0 
            
    if AXI == 1 and IsMono == 0:
        mon_LRC_list = np.concatenate((mon_LRC_list, mon_LRC_list), axis = 0)
        for i in range (nmon):
            mon_LRC_list[i+nmon,1] = 2
            aa=np.cumsum(grid_obj.delr)          #to fill in the xth cell (monitor points)
            mon_LRC_list[i,2] = int(len(aa) - sum(z> mon_obj_list[i].x for z in aa))             
        nmon = len(mon_LRC_list)
    
    return mon_LRC_list        
         
def cleangrid(XGR, dmin):
    '''
    Remove cells smaller than dmin in a grid object. 
     
    :param XGR: 1D array of grid coordinates
    :param dmin: float - target for minimum grid cell size
    :returns: updated 1D array of grid coordinates
    '''
     
    k=0
    while 1:
        Dx = np.diff(XGR);
        minDx = np.minimum(Dx[:len(Dx)-1], Dx[1:])
        minminDx = np.amin(minDx)
 
        if np.fmod(k, 2) == 0:
            imin = np.nonzero(minDx == minminDx)[0][0]
        else:
            imin = np.nonzero(minDx == minminDx)[0][-1]
 
        if minminDx < dmin:
            XGR = np.delete(XGR, imin+1)
            k += 1
        else:
            return XGR
 
 
def boundaries(grid_obj):
    '''
    Create boundary lists for a grid object. Configured to yield a boundary for heads and
    concentrations on the edges of the grid
     
    :param nrow: int - number of grid rows
    :param ncol: int - number of grid columns
    :returns: nested lists representing the 2D boundary arrays (as required for Modflow/MT3DMS)
    '''
     
    # ib = np.ones((nrow, ncol))
    # ib[0, :] = -1 #North boundary
    # ib[len(ib)-1, :] = -1 #South boundary
    # ib[:, 0] = -1 #West boundary
    # ib[:, np.shape(ib)[1]-1] = -1 #East boundary
    # IBOUND = [ib.tolist()]
     
    # ic = np.ones((nrow, ncol))
    # ic[0,:] = -1 #North boundary)
    # ic[len(ic)-1, :] = -1 #South boundary
    # ic[:,0] = -1 #West boundary
    # ic[:, np.shape(ic)[1]-1] = -1 #East boundary
    # ICBUND = [ic.tolist()]      
    
    ib = -np.ones((grid_obj.nlay, grid_obj.nrow, grid_obj.ncol))
    ib[:,1:-1,1:-1] = 1

    IBOUND = ib
    ICBUND = IBOUND


    return IBOUND, ICBUND  

 
def cell_areas(grid_obj):
    '''
    Calculate the area of cells in the grid
     
    :param grid_obj: Python grid object
    :returns: 2D array of cell areas
    '''
     
    Dx = np.diff(grid_obj.XGR)
    Dy = -np.diff(grid_obj.YGR)
     
    area_array = np.zeros((len(Dy), len(Dx)))
    for i in range(len(Dy)):
        for j in range(len(Dx)):
            area_array[i,j] = Dy[i]*Dx[j]
     
    return area_array


def cell_volumes(grid_obj):
    '''
    Calculate the area of cells in the grid
     
    :param grid_obj: Python grid object
    :returns: 2D array of cell areas
    '''
    

    Dx = np.diff(grid_obj.XGR)
    Dy = -np.diff(grid_obj.YGR)
     
    area_array = np.zeros((len(Dy), len(Dx)))
    for i in range(len(Dy)):
        for j in range(len(Dx)):
            area_array[i,j] = Dy[i]*Dx[j]
     
    #vol_array = np.zeros((grid_obj.nlay, grid_obj.nrow, grid_obj.ncol))

    zgr = np.insert(grid_obj.botm, 0, grid_obj.top, axis=0)
    dz = np.diff(zgr, axis=0)
    vol_array = dz*area_array

    return vol_array
     
 
def grid_interpolate(olddata_array, oldgrid_obj, newgrid_obj):
    '''
    Interpolates a 2D data array to match the coordinates of a new grid object
     
    :param olddata_array: 2D array to be interpolated
    :param oldgrid_obj: reference Python grid object (corresponding to the input array)
    :param newgrid_obj: new Python grid object (over which the array should be interpolated)
    :returns: interpolated 2D array
    '''
     
    orig_x = oldgrid_obj.XGR[1:]
    orig_y = oldgrid_obj.YGR[1:][::-1] #Reverse Y axis to match Modflow row scheme
     
    new_x = newgrid_obj.XGR[1:]
    new_y = newgrid_obj.YGR[1:][::-1]
     
    interp_obj = interpolate.interp2d(orig_x, orig_y, olddata_array[::-1], kind='linear')
    newdata_grid = interp_obj(new_x, new_y)[::-1]
     
    return newdata_grid
 
 
def subsurface_use(grid_obj, t_array, Tcrit=1, Tref=10, bounds=None):
    '''
    Calculate the fraction of subsurface area which is "in use" for energy storage -
    based on Analyze_Extr.m by Martin Bloemendal
     
    :param grid_obj: Python grid object
    :param t_array: 2D temperature array
    :param Tcrit: float - deltaT above which the temperature change is considered significant
    :param Tref: float - reference initial temperature of the subsurface
    :param bounds: list of grid extents to use for calculation (MinL, MaxL, MinR, MaxR, MinC, MaxC)
    :returns: float - fraction of the area of interest in which the deltaT is significant
    '''
           
    if bounds:
        MinL, MaxL, MinR, MaxR, MinC, MaxC = bounds
    else:
        MinL = 0
        MaxL = grid_obj.nlay
        MinR = 0
        MaxR = grid_obj.nrow
        MinC = 0
        MaxC = grid_obj.ncol
     
    total_vol = cell_volumes(grid_obj) 
    reference_vol = total_vol[MinL:MaxL,MinR:MaxR, MinC:MaxC]
    reference_temp = t_array[MinL:MaxL, MinR:MaxR, MinC:MaxC] - Tref
     
    area_w = (reference_temp > Tcrit) * reference_vol
    area_c = (reference_temp < -Tcrit) * reference_vol
    used_fraction = (np.sum(area_w)+np.sum(area_c)) / np.sum(reference_vol)
     
    return used_fraction

#def set_heterogeneity(grid_obj, DEF=0, tops=[0], bots=[0], Khs=10, Kvs=2, temp =12, salinity=1, ztop=0):
##    tops = np.zeros(len(form_obj_list))
##    bots = np.zeros(len(form_obj_list))
##    Khs = np.zeros(len(form_obj_list))
##    Kvs = np.zeros(len(form_obj_list))
##    salinity = np.zeros(len(form_obj_list))
##    temp = np.zeros(len(form_obj_list))
##    for j in range (len(form_obj_list)):
##        if j == 0:
##            tops[j] = ztop
##        else:
##            tops[j]= form_obj_list[j-1].zBot
##        bots[j] =  form_obj_list[j].zBot
##        Khs[j] =  form_obj_list[j].Khs  
##        Kvs[j] =  form_obj_list[j].Kvs  
##        salinity[j] =  form_obj_list[j].s2  
##        temp[j] =  form_obj_list[j].s1  
#    
#    dz=-grid_obj.botm[-1,-1,-1]/grid_obj.nlay
#    init_axi_grid = grid_obj.init_grid
#    for i in range(len(grid_obj.delr)):
#        init_axi_grid[:,:,i] =  2 * np.pi * (grid_obj.sumdelr[i] - 0.5 * grid_obj.delr[i])
#            
#    if grid_obj.__class__ == PyGridAxi:
#
#        if DEF == 1:
#            for j in range(len(tops)):
#                t = int(tops[j] / -dz)
#                l = int(bots[j] / -dz)
#                for m in range(int(l-t)):
#                    k=int(m+t)
#                    grid_obj.HK[k,:,:]= init_axi_grid[k,:,:] * Khs[j]
#                    grid_obj.VK[k,:,:]= init_axi_grid[k,:,:] * Kvs[j]
#                    grid_obj.temp[k,:,:]=  temp[j]
#                    grid_obj.salinity[k,:,:]=  salinity[j]
##                grid_obj.HK[t:l,:,:]= Khs[j] * init_axi_grid
##                grid_obj.VK[t:l,:,:]= Kvs[j] * init_axi_grid
#                     
#        if DEF == -1:
#            grid_obj.HK = grid_obj.HK
#            grid_obj.VK = grid_obj.VK
#        else:
#            grid_obj.HK = grid_obj.HK
#            grid_obj.VK = grid_obj.VK
#             
#             
#             
#    if grid_obj.__class__ == PyGrid:
#        if DEF == 1:
#            for j in range(len(tops)):
#                t = int(tops[j] / -dz)
#                l = int(bots[j] / -dz)
#            for m in range(int(l-t)):
#                k=int(m+t)
#                grid_obj.HK[k,:,:]= init_axi_grid[k,:,:] * Khs[j]
#                grid_obj.VK[k,:,:]= init_axi_grid[k,:,:] * Kvs[j]
#
#        if DEF == -1:
#            grid_obj.HK = grid_obj.HK
#            grid_obj.VK = grid_obj.VK                     
#        else:
#            grid_obj.HK = grid_obj.HK
#            grid_obj.VK = grid_obj.VK
