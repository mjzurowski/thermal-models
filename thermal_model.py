import numpy as np
import copy
from scipy.optimize import root # Equilibrium point finder

kb = 8.617333262E-5*1.6E-19 # J/K
n = 5 # thermal conduction power law
cw = 108 # J/K2/m3, tungsten heat capacity
Sig = 0.4E9 # W/K5/m3, electron-phonon coupling
fsc = 1 # SC heat capacity increase - setting this to one now...

def C_dict(c,l0=1E-2,a0=1E-4,T0=50E-3):
    """
    return a dictionary of the appropriate format to calculate the heat capacity
    c = [J/K] list of coefficients used to scale odd powers of temperatute
    l0 = [m] length of material measurement was made
    a0 = [m^2] area of material measurement was made
    T0 = [K] temperature where material was measured
    """ 
    _C = {
        "coeffs": c,
        "V0": l0*a0,
        "T0": T0
    }
    return _C

def G_dict(g,n,l0=1E-2,a0=1E-4,T0=50E-3):
    """
    return a dictionary of the appropriate format to calculate the conductance
    g = [W/K] conductance measured 
    n = [unitless] coupling type
    l0 = [m] length of material measurement was made
    a0 = [m^2] area of material measurement was made
    T0 = [K] temperature where material was measured
    """ 
    _G = {
        "G": g,
        "n": n,
        "A0": a0,
        "L0": l0,
        "V0": l0*a0,
        "T0": T0
    }
    return _G

def ConnectionName(c1,c2):
    """
    Get a unique name for a connection between two materials by ordering their names alphabetically
    """
    temp = [c1,c2]
    temp.sort()
    return temp[0]+","+temp[1]

class Connection:
    def __init__(self,comp1,comp2,g_dict):
        """
        A thermal link that exists between Components comp1 and comp2 defined by some conductance value and coupling
        comp1 = the first Component - it is assumed this is the material that you measure the conduction FROM
        comp2 = the second Component - it is assumed this is the material you measure the conduction TO
        g_dict = dictionary containing a measurement of the conductance at some temperature and geometry
        """
        self.name = ConnectionName(comp1.name,comp2.name) ## force the name of the connection to be organised by alphabetical order
        self.g = g_dict["G"]
        self.n = g_dict["n"]
        if self.n == 5: 
            ## Apply scaling correction for electron phonon coupling
            self.K = self.g/(self.n*pow(g_dict["T0"],self.n))*(comp1.len/g_dict["L0"])*(g_dict["A0"]/comp1.area)
        else: 
            ## Apply scaling correction for non electron phonon coupling
            self.K = self.g/(self.n*pow(g_dict["T0"],self.n))*(comp1.area*comp1.len/g_dict["V0"])
        self.c1 = comp1
        self.c2 = comp2

    def info(self):
        print("Connection "+self.name+" is between components "+self.c1.name+" and "+self.c2.name)
        print("Conductance is "+str(self.K)+" J/K with power law n="+str(self.n))

class Component:
    def __init__(self,name,T_i,len,area,c):
        """
        Initialise a component with a name and list of connections. 
        name = [string] label for the component
        T_i = [K] initial temperature
        len = [m] length of component
        area = [m^2] area of component
        c = [J/K] list of heat capacity constants of material. If it's not a number, it should be a dictionary that includes a list of coefficients and the temp, area, and length they were measured at
        """
        self.name = name
        self.T = T_i
        self.c = c
        self.len = len
        self.area = area
        self.connections = {}

    def C(self):
        if type(self.c)!=dict:
            ## if its a number instead of a dictionary, just return the number
            return self.c
        else:
            _C = 0
            coeffs = self.c["coeffs"]
            for i in range(0,len(coeffs)):
                _C+=coeffs[i]*pow(self.T/self.c["T0"],2*i+1)*self.len*self.area/self.c["V0"]
            return _C

    def info(self):
        print("Component "+self.name+" has a heat capacity of "+str(self.C())+" and the following connections:")
        for c in self.connections:
            self.connections[c][1].info()
            print(" ")

    def add_connection(self,comp,g_dict,other=False):
        """
        Add a connection between this Component and some other external component ("component") given some conductance dictionary
        comp = other component
        g_dict = conductance dictionary
        other = optional flag. If this is false, it assumes the conductance dictionary was constructed based on measurements FROM self. If true, the measurements are TO self.
        """
        if other:
            connect = Connection(comp,self,g_dict)

        else:
            connect = Connection(self,comp,g_dict)

        ## add the connection to this Component
        if connect.name not in self.connections:
            self.connections[connect.name] = [comp,connect]
        else:
            print("Connection to "+comp.name+" already exists:")

        ## add the connection to the other component
        if connect.name not in comp.connections:
            comp.connections[connect.name] = [self,connect]

    def remove_connection(self,comp):
        ## add code to remove a connection from the list in case something is added incorrectly
        name = ConnectionName(self.name, comp.name)
        self.connections.pop(name)
        comp.connections.pop(name)

    def P(self,external_comp,connect):
        """
        Power lost from this Component to external_comp via this connection
        Output units: [J/s]
        """
        return connect.K*(pow(self.T,connect.n) - pow(external_comp.T,connect.n))
    
    def dT(self,dt):
        """
        Change in temperature in some timestep dt for Component due to its connections
        Output units: [K]
        """
        if self.name=="bath":
            return 0 ## bath shouldn't change temp
        else:
            p_lost = 0
            for c in self.connections:
                comp = self.connections[c][0] ## get the component this connection is with
                connect = self.connections[c][1] ## get the information about the connection with this component
                p_lost+= self.P(comp,connect)
            return -dt*p_lost/self.C()

class TES(Component):
    def __init__(self,R_L,alpha,beta,Tc,R0,Rn,L,T_i,len,area,c):
        """
        TES is a specialised type of component that gains power via Joule heating
        name = [string] label for the component
        R_L = [Ohm] load resistance (parasitic + shunt)
        alpha = thermal sensitivity
        beta = current sensitivity
        Tc = [K] critical temperature
        R0 = percentage bias
        Rn = [Ohm] normal resistance
        L = [H] inductance of the TES
        len = [m] length of component
        area = [m^2] area of component
        c = list of heat capacity constants of material. If it's not a number, it should be a dictionary that includes a list of coefficients and the temp, area, and length they were measured at
        """
        super().__init__("TES",T_i,len,area,c) ## inherit attributes of the base class

        ## Add the TES specific ones
        self.R_L = R_L
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.Tc = Tc
        self.R0 = R0*Rn
        self.Rn = Rn

        self.I = None ## Initial state current assumed to be I0
        self.I0 = None

    def initialize_I0(self):
        ### Need to do a full thing to get the equilibrium state
        """
        Get the initial value of I0 assuming everything in equilibrium for a given temp and resistance
        """
        dummy_dt = 1
        thermal_load = self.dT(dummy_dt) ## dt value doesn't matter as it gets cancelled
        P_J0 = -(self.C()/dummy_dt)*thermal_load
        if P_J0 < 0:
            print("Initial settings aren't right, requesting an imaginary TES current... try again")
        else:
            self.I = np.sqrt(P_J0/self.R0)
            self.I0 = np.sqrt(P_J0/self.R0)
            print("setting current to "+str(self.I)+" A")

    def R(self):
        """
        Resistance of the TES, given its instantaneous temperature and current
        Output units: [Ohm]
        """
        if self.I0 == 0:
            x_tanh = (self.T - self.Tc)*self.alpha/self.Tc
        else:
            x_tanh = (self.T - self.Tc)*self.alpha/self.Tc + (self.I - self.I0)*self.beta/self.I0
        return 0.5*self.Rn*(1+np.tanh(x_tanh))

    def P_J(self):
        """
        Joule power heating on the TES due to the current passing through
        Output units: [J/s]
        """
        return self.I*self.I*self.R()

    def dI(self,dt):
        """
        Change in current over some period dt due to resistance
        Output units: [A]
        """
        return (dt/self.L)*(self.I0*self.R0 - self.R_L*(self.I - self.I0) - self.I*self.R())

    def dT_TES(self,dt):
        """
        Total change in TES temperature in a time step. Accounts for Joule heating and loss to connections.
        Output units: [K]
        """
        return self.P_J()*(dt/self.C()) + self.dT(dt)

    ############################################
    
    
class ThermalModel:
    def __init__(self, components):
        """
        A thermal model takes in a list of components with associated connections (one of which must be an absorber with the label 'abs') and a TES
        It then provides a list of coupled functions to calculate the temperature of all the components (and current/resistance of the TES)
        """
        self.components = {}
        for c in components:
            self.add_component(c)
            
        if "TES" not in self.components:
            print("You haven't included a TES... make sure you add one")
        if "abs" not in self.components:
            print("You haven't included an absorber with label abs... make sure you add one")

        self.tot_time = 0 ## total time elapsed
        self.initial_settings = None
        self.equilibrate()

    def print_connections(self):
        connects = []
        for comp in self.components:
            for c in self.components[comp].connections:
                connects.append(c)
        print(connects)

    def add_component(self,comp):
        self.components[comp.name] = comp

    def remove_component(self,comp):
        self.comp.pop(comp.name)

    def equilibration_func(self,T_s):
        """
        Function to get initial starting conditions. We assume TES is held at Tc and bath is at Tb and everything else is able to float.
        The root finding algorithm should then solve for all the temperature values for the other components so that their temp isn't changing
        This returns the list of dT values we are setting to 0, given this list of temps
        T_s = list of temperatures
        """
        dummy_dt = 1
        d_T = []
        i=0
        for c in self.components:
            if c not in ["TES","bath"]:
                self.components[c].T = T_s[i] ## pull the list of components we are going to fit to
                d_T.append(self.components[c].dT(dummy_dt)*1E6/dummy_dt)
                i+=1
        return d_T

    def find_roots(self):
        """
        Apply the root finding algorithm.
        """
        T_0 = []
        for c in self.components:
            if c not in ["TES","bath"]:
                T_0.append(self.components[c].T) ## get initial guess from components
        if len(T_0)!=0:
            ## If there are only a TES and bath in the system, we can skip the temperature calcs
            eq = root(self.equilibration_func,T_0,method = 'lm')
            T_s = eq.x
            i=0
            for c in self.components:
                if c not in ["TES","bath"]:
                    self.components[c].T = T_s[i] ## force it to the correct settings
                    i+=1
            return eq

    def equilibrate(self,_print=True):
        """
        Function to get initial starting conditions. We assume TES is held at Tc and bath is at Tb and everything else is able to float.
        The root finding algorithm should then solve for all the temperature values for the other components so that their temp isn't changing
        Once the temperatures have all been set, the TES current is then also calculated
        """
        print("Calculating temperatures for equilibrium:")
        
        ## force the root finding algorithm to keep going to find a better root
        eq = self.find_roots()
        count = 0
        while(np.sum(eq.fun**2) > 1e-11 and count < 10000):
            count += 1
            eq = self.find_roots()
            dT = {}
            for comp in self.components:
                if comp not in ["TES","bath"]:
                    dT[comp] = self.components[comp].dT(1E-6)
            for comp in self.components:
                if comp not in ["TES","bath"]:   
                    self.components[comp].T += dT[comp]


        if _print:
            for c in self.components:
                print(c+" temp: "+str(self.components[c].T)+" K")

        ## Calculate the initial current based on these temperatures
        self.components["TES"].initialize_I0()

        ## Save a deeop copy of the initial settings so we can easily "reset" things to equilibrium
        self.initial_settings = copy.deepcopy(self.components)

    def reset_model(self):
        """
        Reset the model to equilibrium conditions at any time
        """
        self.components = copy.deepcopy(self.initial_settings)
        self.tot_time = 0 ## total time elapsed

    def step(self,event,dt):
        """
        Iterate over a step in time dt and increment all appropriate values
        event is some function of time that represents some injection of power into the absorber
        event = [J] some function that deposits energy in the absorber
        dt = [s] time step
        """
        dT = {}
        self.tot_time+=dt

        ## calculate the change in temperature for everyone
        for comp in self.components:
            if comp == "TES":
                dT[comp] = self.components[comp].dT_TES(dt)
            elif comp == "abs":
                dT[comp] = event(self.tot_time)*(dt/self.components[comp].C()) + self.components[comp].dT(dt) 
            else:
                dT[comp] = self.components[comp].dT(dt)

        ## apply that temperature change
        for comp in self.components:
            self.components[comp].T += dT[comp]

        ## finally, also iterate the current in the TES
        self.components["TES"].I += self.components["TES"].dI(dt)
