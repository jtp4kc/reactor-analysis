'''
Created on Apr 25, 2017

@author: adjun_000
'''

import os
TEST_FOLDER = os.path.join("C:\\", "Users", "adjun_000", "Documents",
    "Grad School", "Grad School Research (PhD)", "Experiments",
    "20170517 Reactions over SrPO4")  # 0.060 g, methane, 9.6 m^2/g
TEST_FILE = "setup.txt"

# Constants
R = 8.3144621  # L*kPa / K*mol
T = 273.15 + 25  # K, ambient
P = 101.325  # kPa, ambient
Ts = 273.15  # K, standard
Ps = 101.325  # kPa, standard

class _SetupReader():

    def _load(self, filename):
        with open(filename, "r") as inputfile:
            holdon = False
            section = []
            for line in inputfile:
                nocomment = line.split("#")
                line = nocomment[0]
                line = line.strip()
                if line != "":
                    if holdon:
                        if line.lower() == "end":
                            self._multiline(section)
                            holdon = False
                            section = []
                        else:
                            section.append(line)
                    elif "=" in line and len(line.split("=")) == 2:
                        self._assign(line)
                    else:
                        hold = self._checkhold(line)
                        if hold:
                            holdon = True
                            section.append(line)
                        else:
                            segs = line.split()
                            if segs[0] == "new":
                                self._make(segs[1:])
                            elif segs[0] == "set":
                                self._set(segs[1:])
                            else:
                                self._handleline(line)
            if len(section) > 0:
                self._multiline(section)
            inputfile.close()

    def _checkhold(self, line):  # should multiline?
        return False
    def _assign(self, line):  # variable = value
        pass
    def _set(self, segs):  # set .....
        pass
    def _make(self, segs):  # new .....
        pass
    def _multiline(self, lines):  # handle block
        pass
    def _handleline(self, line):  # for anything else
        pass

class Reference(_SetupReader):

    def __init__(self):
        self.components = []
        self.vle = []
        self.focus = None

    def load(self, filename):
        if filename is not None:
            self._load(filename)

    def _checkhold(self, line):
        return False

    def _assign(self, line):
        pass

    def _set(self, segs):
        if segs[0].lower() == "vle":
            coeffs = []
            for seg in segs[2:]:
                coeffs.append(float(seg))
            self.vle.append((segs[1], coeffs))
            print("VLE for " + segs[1] + ": " + str(coeffs))
        elif segs[0].lower() == "focus":
            self.focus = segs[1:]
            print("Focus: " + str(self.focus))

    def _make(self, segs):
        pass

    def _multiline(self, lines):
        pass

    def _handleline(self, line):
        segs = line.split()
        if (len(segs) > 0):
            name = segs[0]
            name = name.replace("_", " ")
            rf = None
            mw = None
            s = None
            if len(segs) > 1:
                rf = float(segs[1])
            if len(segs) > 2:
                mw = float(segs[2])
            if len(segs) > 3:
                s = float(segs[3])
            self.components.append(Component(nm=name, rf=rf, mw=mw, s=s))

class Reaction(_SetupReader):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.components = []
        self.flows = []
        self.reactant = None
        self.standard = None
        self.inert = None
        self.rxndate = ""
        self.catalyst = ""
        self.catdate = ""
        self.surfarea = 0
        self.loading = 0
        self.datafile = ""
        self.dataext = ""
        self.folderout = "Results"
        self.focus = []
        self.disable = []
        self._reactantname = ""
        self._standardname = ""
        self._inertname = ""

    def do_test(self):
        self.components += [
            Component("meth", 1.0, 0.05, 1.00, 16, 1),
            Component("eth" , 2.0, 0.05, 1.00, 30, 2),
            Component("prop", 3.0, 0.05, 1.00, 44, 3),
            Component("but" , 4.0, 0.05, 1.00, 58, 4),
            Component("pent", 5.0, 0.05, 1.00, 72, 5),
            Component("hex" , 6.0, 0.05, 1.00, 86, 6),
            Component("hept", 7.0, 0.05, 1.00, 100, 7),
            Component("oct" , 8.0, 0.05, 1.00, 114, 8),
            Component("non", 9.0, 0.05, 1.00, 128, 9)]
        meth = self.components[0]
        ethn = self.components[1]
        inrt = Component("dec", 10.0, 0.05, 1.00, 142, 10)
        self.standard = meth
        self.reactant = ethn
        self.inert = inrt
        self.rxndate = "20171011"
        self.catalyst = "TEST"
        self.catdate = "20171011"
        self.loading = 0.050
        self.surfarea = 1.0
        self.flows += [Flow() for _ in range(4)]
        i = 0
        for flow in self.flows:
            flow.vle = [0.0001, 0.001, 0.01, 0.1, 1]
            flow.temp.setAsCelsius(240 + i * 20)
            flow.pres.set(1, "psi", True)
            flow.hours[0] = 10 * i
            flow.hours[1] = 10 * (i + 1)
            flow.sat_pres.set(i, "psi", True)
            flow.sat_temp.setAsCelsius(27)
            flow.npts = i
            flow.fillarea(self.components[2], 9000)
            flow.background(self.components[2], 7000, True)
            flow.background(self.components[3], 0.5, False)
            flow.add(ethn, 0.005 * 0.80 * 1.00, "mass")
            flow.add(inrt, 15 + 5 * i, "amb", None, False)
            flow.add(inrt, 25, "std", ethn, False)
            flow.add(meth, 10, "amb", None, True)
            i += 1
        self.flows[0].ignorecomp(self.components[4])
        self.flows[0].ignoretime(self.components[6], 2)
        self.focus = [self.components[2], self.components[3],
                      self.components[6], self.components[7]]
        self.disable = [self.components[6]]

    def load(self, filename, reference):
        if filename is None:
            self.do_test()
        else:
            self._load(filename)
            self.reactant = self._find(self._reactantname)
            print("Reactant: " + str(self.reactant))
            if self.reactant is not None:
                self.components.remove(self.reactant)
                self.components.insert(0, self.reactant)
            self.standard = self._find(self._standardname)
            print("Standard: " + str(self.standard))
            if self.standard is not None:
                self.components.remove(self.standard)
                self.components.append(self.standard)
            self.inert = self._find(self._inertname)
            print("Inert: " + str(self.inert))
            if self.inert is not None:
                self.components.remove(self.inert)
            for flow in self.flows:
                print flow
            self.apply_reference(reference)

    def _checkhold(self, line):
        segs = line.split()
        if len(segs) > 1 and segs[0] == "new" and segs[1].lower() == "flow":
            return True
        return False

    def _assign(self, line):
        segs = line.split("=")
        key = segs[0].strip().lower()
        val = segs[1].strip()
        found = True
        if key == "rxndate":
            self.rxndate = val
        elif key == "catalyst":
            self.catalyst = val
        elif key == "catdate":
            self.catdate = val
        elif key == "surfarea":
            self.surfarea = float(val)
        elif key == "loading":
            self.loading = float(val)
        elif key == "datafile":
            self.datafile = val
        elif key == "dataext":
            self.dataext = val
        elif key == "folderout":
            self.folderout = val
        else:
            print("Unrecognized key: " + key + " for value " + val)
            found = False
        if found:
            print(key + " = " + val)

    def _set(self, segs):
        if segs[0].lower() == "reactant":
            self._reactantname = segs[1]
        elif segs[0].lower() == "standard":
            self._standardname = segs[1]
        elif segs[0].lower() == "inert":
            self._inertname = segs[1]
        elif segs[0].lower() == "disabled":
            comp = self._find(segs[1])
            if comp is not None:
                print("Disable: " + comp.name)
                self.disable.append(comp)
        elif segs[0].lower() == "vle":
            comp = self._find(segs[1])
            if comp is not None:
                vle = []
                for seg in segs[1:]:
                    vle.append(float(seg))
                comp.vle = vle
                print("VLE for " + str(comp) + ": " + str(vle))
        elif segs[0].lower() == "focus":
            self.focus = []
            for seg in segs[1:]:
                comp = self._find(seg)
                if comp is not None:
                    self.focus.append(comp)
            print("Focus: " + str(self.focus))

    def _make(self, segs):
        if segs[0].lower() == "component":
            name = segs[1]
            name = name.replace("_", " ")
            rt = None
            rta = None
            rf = None
            mw = None
            s = None
            if len(segs) > 2:
                rt = float(segs[2])
            if len(segs) > 3:
                rta = float(segs[3])
            if len(segs) > 4:
                rf = float(segs[4])
            if len(segs) > 5:
                mw = float(segs[5])
            if len(segs) > 6:
                s = float(segs[6])
            ret = Component(nm=name, rt=rt, rta=rta, rf=rf, mw=mw, s=s)
            self.components.append(ret)

    def _multiline(self, lines):
        item = None
        for line in lines:
            segs = line.split()
            if segs[0] == "new" and segs[1].lower() == "flow":
                item = Flow()
                self.flows.append(item)
                for seg in segs[2:]:
                    keyval = seg.split("=")
                    key = keyval[0]
                    val = float(keyval[1])
                    if key == "temp":
                        item.temp.setAsCelsius(val)
                    elif key == "pres":
                        item.pres.set(val, "psi", True)
                    elif key == "start":
                        item.hours[0] = val
                    elif key == "end":
                        item.hours[1] = val
                    elif key == "satP":
                        item.sat_pres.set(val, "psi", True)
                    elif key == "satT":
                        item.sat_temp.setAsCelsius(val)
                    elif key == "npts":
                        item.npts = int(val)
                    elif key == "equil":
                        item.equil = int(val)
                    elif key == "label":
                        item.label = val
            elif segs[0] == "background":
                comp = self._find(segs[1])
                if comp is not None:
                    if len(segs) > 3 and segs[3] == "%conv":
                        item.background(comp, float(segs[2]))
                    else:
                        item.background(comp, float(segs[2]), True)
            elif segs[0] == "nopkarea":
                comp = self._find(segs[1])
                if comp is not None:
                    item.fillarea(comp, float(segs[2]))
            elif segs[0] == "ignore":
                comp = self._find(segs[1])
                if comp is not None:
                    item.ignorecomp(comp)
            elif segs[0] == "ignoreone":
                comp = self._find(segs[1])
                if comp is not None:
                    item.ignoretime(comp, float(segs[2]))
            elif segs[0] == "link":
                for flow in self.flows:
                    if segs[1] == flow.label:
                        item.link = flow
            else:
                comp = self._find(segs[0])
                if comp is not None:
                    if segs[1] == "mass":
                        grpm = float(segs[2])
                        dens = 1
                        frac = 1
                        if len(segs) > 3:
                            dens = float(segs[3])  # density
                        if len(segs) > 4:
                            frac = float(segs[4])  # fraction
                        item.add(comp, grpm, dens, frac, "mass")
                    elif segs[1] == "amb" or segs[1] == "std":
                        after = False
                        sat = None
                        if len(segs) > 3:
                            if segs[3] == "after":
                                after = True
                            elif segs[3] == "sat":
                                sat = self._find(segs[4])
                        item.add(comp, float(segs[2]), unit=segs[1], satur=sat,
                                 after=after)

    def _handleline(self, line):
        pass

    def _find(self, name):
        name = name.replace("_", " ").lower()
        for comp in self.components:
            if comp.name.lower() == name:
                return comp
        if self.inert is not None and self.inert.name.lower() == name:
            return self.inert
        if self.reactant is not None and self.reactant.name.lower() == name:
            return self.reactant
        if self.standard is not None and self.standard.name.lower() == name:
            return self.standard
        return None

    def apply_reference(self, reference):
        for comp in reference.components:
            mine = self._find(comp.name)
            if mine is not None:
                mine.resp_factor = comp.resp_factor
                mine.mol_weight = comp.mol_weight
                mine.stoich = comp.stoich
                # print("Apply reference info for " + str(mine))
        for vle in reference.vle:
            name = vle[0]
            mine = self._find(name)
            if mine is not None:
                mine.vle = vle[1]
                # print("Apply reference vle to " + str(mine))
        if reference.focus is not None:
            # print("Using reference for focus")
            self.focus = []
            for nm in reference.focus:
                self.focus.append(self._find(nm))
            # print(self.focus)

class Component:

    def __init__(self, nm="unkwn", rt=None, rta=None, rf=None, mw=None, s=None):
        self.name = nm
        self.ret_time = rt
        self.rt_adjust = rta
        self.resp_factor = rf
        self.mol_weight = mw
        self.stoich = s
        self.vle = None

    def __str__(self):
        return "Component_" + self.name

    def __repr__(self):
        return "Component_" + self.name

class Temp():

    convCK = 273.15
    convFR = 459.67

    def __init__(self, value=0, scale="K"):
        self.temp = self.convCK  # K
        if scale == "C":
            self.setAsCelsius(value)
        elif scale == "F":
            self.setAsFahrenheit(value)
        elif scale == "R":
            self.setAsRankine(value)
        else:
            self.setAsKelvin(value)

    def setAsKelvin(self, value):
        self.temp = value

    def getAsKelvin(self):
        return self.temp

    def setAsCelsius(self, value):
        self.temp = value + self.convCK

    def getAsCelsius(self):
        return self.temp - self.convCK

    def setAsFahrenheit(self, value):
        self.setAsCelsius((value - 32) * 5 / 9.0)

    def getAsFahrenheit(self):
        return self.getAsCelsius() * 9 / 5.0 + 32

    def setAsRankine(self, value):
        self.setAsFahrenheit(value - self.convFR)

    def getAsRankine(self):
        return self.getAsFahrenheit() + self.convFR

    def __str__(self):
        return str(self.temp) + " K"

class Pres():

    PA = "Pa"
    KPA = "kPa"
    ATM = "atm"
    PSI = "psi"
    BAR = "bar"
    MBAR = "mbar"
    MMHG = "mmHg"
    TORR = MMHG
    INHG = "inHg"
    MMH2O = "mmH2O"
    INH2O = "inH2O"

    convAtm = 101325.0  # Pa/atm
    convBar = 100000.0  # Pa/bar
    convTorr = convAtm / 760.0  # Pa/mmHg
    convPsi = convAtm / 14.6959  # Pa/psi
    convH2O = convTorr / 13.6087  # Pa/mmH2O
    inchmm = 25.4  # mm/in

    def __init__(self, value=0, unit="Pa", gauge=False):
        self.pres = self.convAtm  # Pa
        self.set(value, unit, gauge)

    def set(self, value, unit="Pa", gauge=False):
        convValue = 1
        if unit == self.PA:
            convValue = 1
        elif unit == self.KPA:
            convValue = 1000.0
        elif unit == self.ATM:
            convValue = self.convAtm
        elif unit == self.PSI:
            convValue = self.convPsi
        elif unit == self.BAR:
            convValue = self.convBar
        elif unit == self.MBAR:
            convValue = self.convBar / 1000.0
        elif unit == self.TORR:
            convValue = self.convTorr
        elif unit == self.INHG:
            convValue = self.convTorr * self.inchmm
        elif unit == self.MMH2O:
            convValue = self.convH2O
        elif unit == self.INH2O:
            convValue = self.convH2O * self.inchmm
        else:
            print("Unknown unit value.")
            return

        if gauge:
            value += self.convAtm / convValue
        self.pres = value * convValue

    def get(self, unit="Pa", gauge=False):
        convValue = 1
        if unit == self.PA:
            convValue = 1
        elif unit == self.KPA:
            convValue = 1000.0
        elif unit == self.ATM:
            convValue = self.convAtm
        elif unit == self.PSI:
            convValue = self.convPsi
        elif unit == self.BAR:
            convValue = self.convBar
        elif unit == self.MBAR:
            convValue = self.convBar / 1000.0
        elif unit == self.TORR:
            convValue = self.convTorr
        elif unit == self.INHG:
            convValue = self.convTorr * self.inchmm
        elif unit == self.MMH2O:
            convValue = self.convH2O
        elif unit == self.INH2O:
            convValue = self.convH2O * self.inchmm
        else:
            print("Unknown unit value.")
            return

        value = self.pres / convValue
        if gauge:
            value -= self.convAtm / convValue
        return value

    def __str__(self):
        return str(self.pres) + " Pa"

class Subflow():

    def __init__(self, component, val0=0, val1=1, val2=0, unit="std", satur=None):
        self.component = component
        self.vals = [val0, val1, val2]
        self.unit = unit
        self.satur = satur

class Flow():

    # PV = nRT,  n/V = P/RT
    convAmb = P / (R * T * 1000)  # mol/mL, ambient
    convStd = Ps / (R * Ts * 1000)  # mol/mL, standard

    def __init__(self):
        self.temp = Temp(25, "C")
        self.pres = Pres(0, "psi", True)
        self.hours = [0, 0]
        self.sat_temp = Temp(25, "C")
        self.sat_pres = Pres(0, "psi", True)
        self.npts = 0
        self.equil = 0
        self.stream = []
        self.post = []
        self.fill = []
        self.bkgd = []
        self.ignore = []
        self.ignoreone = []
        self.label = None
        self.link = None

    def add(self, component, rate, density=0, wtpercent=0, unit="std", satur=None,
                after=False):
        ''' Set satur to a component to specify what is being bubbled through
            Set after to true to state that a component is only introduced after
                the reactor
            component - the component object for this particular flow
            rate - the number value of the flow, per minute
            unit - what unit the flow is in, either "mass" (g, liq) or, if 
                volume (mL, vapor), specify "amb" or "std" for 25 C or 0 C 
                calibrated flow
        '''
        if unit.lower() == "std" or unit.lower() == "amb":
            if density > 0 or wtpercent > 0:
                print("Warning: unexpected values for gas flow " + component.name +
                      " at " + str(rate) + " mL/min")
        elif unit.lower() == "mass":
            if density <= 0:
                print("Warning: density not given for " + component.name + " at " +
                      str(rate) + " g/min")
            if wtpercent <= 0:
                wtpercent = 1
        else:
            print("Unknown unit for flowrate.")
            return

        subflow = Subflow(component, rate, density, wtpercent, unit, satur)

        if after:
            self.post.append(subflow)
        else:
            self.stream.append(subflow)

#         if satur is not None:
#             satp = self.sat_pres.get("kPa")
#             temp = self.sat_temp.getAsCelsius()
#             pres = 0  # kPa
#             for (coef, order) in zip(self.vle,
#                                      range(len(self.vle) - 1, -1, -1)):
#                 pres += coef * temp ** order
#             carp = satp - pres
#             mols = mol * pres / carp
#             self.stream.append((satur, mols))
#             self.reactorflow += mols
#             self.totalflow += mols

    def fillarea(self, component=None, val=0):
        self.fill.append((component, val))

    def background(self, component=None, val=0, area=False):
        self.bkgd.append((component, val, area))

    def ignorecomp(self, component):
        self.ignore.append(component)

    def ignoretime(self, component, index):
        self.ignoreone.append((component, index))

    def __str__(self):
        return "Flow_hrs" + str(self.hours)

    def __repr__(self):
        return "Flow_hrs" + str(self.hours)

if __name__ == '__main__':
    os.chdir(TEST_FOLDER)
    Reaction().load(TEST_FILE)












