'''
Created on Jun 14, 2017
Updated Jul 2018
Works with the files generated from the Agilent and Shimadzu GCs
@author: Tyler P
'''

import os
import math

import AnalyzeChromatogram as ac
import ReactorConditionsV7 as rc
import ReactorResultsV4 as rr
import PlotResults as pr
import numpy as np

TEST_MODE = False
# Reaction:
# 2 EtOH -> 2 AcH + 2 H2 -> CrH + H2O + 2 H2 -> BuOH + H2O || BD + 2 H2O + 2 H2
PATHS = [
#### Agilent GC and reactors
#    "20160808 Reaction over HAP",  # 0.060 g, octane, 35 m^2/g?
#    "20160901 Reaction over SiO2-MgO",  # 0.200 g, octane, 54.9 m^2/g
#    "20161114 Reactions over SiO2-MgO",  # 0.060 g, methane, 54.9 m^2/g
#    "20161219 Reactions over SiO2-MgO",  # blank run, methane
#    "20170313 Reactions over SiO2-MgO",  # 0.200 g, methane
#    "20170315 Reactions over SiO2-MgO",  # 0.200 g, methane
#    "20170410 Reactions over SiO2-MgO",  # 0.051 g, methane 88.1 m^2/g
#### Reactor #1, Shimadzu GC
#    "20170424 Reactions over HAP",  # 0.050 g, methane, 35 m^2/g
#    "20170426 Reactions over HAP",  # 0.061 g, methane, 35 m^2/g
#    "20170501 Reactions over SrPO4",  # 0.200 g, methane, 9.6 m^2/g
#    "20170509 Reactions over SrPO4",  # 0.200 g, methane, 9.6 m^2/g
#    "20170517 Reactions over SrPO4",  # 0.060 g, methane, 9.6 m^2/g
#    "20170530 Reactions over SrPO4",  # 0.060 g, methane, 9.6 m^2/g
#    "20170605 Background",  # methane
#    "20170612 Reactions over MgPO4",  # 0.062 g, methane, 21 m^2/g
#    "20170619 Reactions over CaPO4",  # 0.060 g, methane, 5.4 m^2/g
#    "20170626 Reactions over SrPO4 Sabra",  # 0.120 g, methane, 1.5 m^2/g
#    "20170704 Reactions over CaPO4 HiConv",  # 0.200 g, methane, 5.4 m^2/g
#    "20170710 Reactions over MgPO4-HiConv",  # 0.314 g, methane, 21 m^2/g
#    "20170717 CaPO4-HiConv2",  # 0.403 g, methane, 5.4 m^2/g
#    "20170724 CaPO4-600C",  # 0.410 g, methane, 60.5 m^2/g
#    "20170726 CaPO4-700C",  # 0.300 g, methane, 12.2 m^2/g
#    "20170731 CaPO4-800C",  # 0.304 g, methane, 12.3 m^2/g
#    "20170807 CaPO4-900C",  # 0.400 g, methane, 1.4 m^2/g
#    "20170809 SrPO4-600C",  # 0.106 g, methane, 39.8 m^2/g
#    "20170815 BTCP-Sabra",  # 0.302 g, methane, 5.2 m^2/g
#    "20170821 SrPO4-600C_3",  # 0.301 g, methane, 39.8 m^2/g
#    "20170823 SrPO4-700C",  # 0.394 g, methane, 12.8 m^2/g
#    "20170828 SrPO4-800C",  # 0.407 g, methane, 5.5 m^2/g
#    "20170830 SrPO4-900C",  # 0.410 g, methane, 4.2 m^2/g
#    "20170911 Background",  # methane
#    "20170913 CaPO4-900C",  # 0.410 g, methane, 1.4 m^2/g
#    "20170918_Co-N-C",  # 0.060 g, methane, ?.? m^2/g
#    "20170921_Co-N-C",  # 0.060 g, methane, ?.? m^2/g
#    "20170922_N-C",  # 0.059 g, methane, ?.? m^2/g
#    "20170925_Co-C",  # 0.035 g, methane, ?.? m^2/g
#    "20170927_Co-N-C-HCl",  # 0.060 g, methane, ?.? m^2/g
#    "20170929_Co-N-C",  # 0.035 g, methane, ?.? m^2/g
#### Reactor #2, Shimadzu GC
#    "20171005_SrPO4-RT",  # 0.119 g, methane, 9.4 m^2/g
#    "20171009_Background",  # methane
#    "20171011_SrPO4-900C",  # 0.348 g, methane, 4.2 m^2/g
# -temp divide
    # "20171015_SiO2-SrPO4",  # 0.102 g, methane, 6.0 m^2/g
    # "20171018_SiO2-SrCO3",  # 0.098 g, methane, ?.? m^2/g
    # "20171021_SiO2-MgO",  # 0.200 g, methane, 16.8 m^2/g
    # "20171024_SiO2-SrPO4_2",  # 0.211 g, methane, 6.0 m^2/g
    # "20171031_ZrO2-SrPO4",  # 0.206 g, methane, 11.9 m^2/g
    # "20171106_SiO2-SrO",  # 0.403 g, methane, 2.2 m^2/g need to look at unknown 1,3-BDO
    # "20171108_ZrO2",  # 0.200 g, methane, 10.4 m^2/g
    # "20171113_SrPO4-800C",  # 0.200 g, methane, ?.? m^2/g
    # "20171120_MgO",  # 0.200 g, methane, ?.? m^2/g
    # "20171128_SiO2",  # 0.200 g, methane, ?.? m^2/g
#    "20171204_Background",  # methane
#    "20171207_Background",  # methane
    # "20171211_SiO2",  # 0.200 g, methane, ?.? m^2/g
#### Reactor #3, Shimadzu GC
#    "20171215_ZrO2-SrPO4",  # 0.200 g, methane, ?.? m^2/g
#    "20171220_Background",  # methane
#    "20171228_SiO2-SrPO4_1to2",  # 0.200 g, methane, ?.? m^2/g
#    "20180115_SrCO3",  # 0.235 g, methane, ?.? m^2/g
#    "20180119_ResponseSeries",  # 0.000 g, methane, 0.0 m^2/g
#    "20180129_ZrO2-SrPO4_2to1",  # 0.103 g, methane, ?.? m^2/g
#    "20180206_ZrO2-SrPO4_1to2",  # 0.101 g, methane, ?.? m^2/g
#    "20180219_ZrO2",  # 0.150 g, methane, 10.4 m^2/g
#    "20180226_ZrO2",  # 0.101 g, methane, 10.4 m^2/g
#    "20180312_ZrO2",  # 0.050 g, methane, 10.4 m^2/g
#    "20180326_CaHAP",  # 0.060 g, methane, 10.4 m^2/g

#    "20180917_LargeRxtr_Blank",  # methane
    "20180921_CaHAP-Zack",  # 0.102 g, methane, 35 m^2/g
]
SETUPS = ["setup.txt"
#          , "setupBlank.txt"
#          , "setupRegeneration.txt"
#          , "setupDeactivation.txt"
          ]
FOLDER_BASE = os.path.join("C:\\", "Users", "adjun", "Documents",
    "School", "Reactions")
FOLDER_IN = [os.path.join(FOLDER_BASE, path) for path in PATHS]
GLOBAL_SETTINGS = os.path.join(FOLDER_BASE, "reference.txt")

def findval(nparray, value, extrainfo=False):
    ''' Find the index of a value in an array that is closest but still less
        than the specified value
        Useful for finding a bracket of values in an array, if the array is 
        inherently sorted (i.e. for plotting)
        @param nparray: 1-D numpy array
        @param value: the value to approach but not exceed when searching
        @param extrainfo: if True, return the sorted array, sorting array, and 
                            sorted index in a tuple in addition to the index
        @return index, an integer showing the location of the identified value,
                    if negative, the value is less than all entries in the array
    '''
    ind = np.argsort(nparray)
    x = nparray[ind]
    pos = -1
    for i in range(len(x)):
        if (x[i] > value):
            break
        pos = i
    if (pos == -1):
        index = -1
    else:
        index = ind[pos]
    if extrainfo:
        return (index, x, ind, pos)
    else:
        return index

class Compdata():

    def __init__(self, comp_list, component, data, raw, flow):
        self.comp = component
        for g in range(len(comp_list)):
            if component is comp_list[g]:
                break
        self.counts = data[:, g]
        self.raw = raw[:, :, g]
        self.rate = []
        self.satur = []
        self.after = []
        self.bkgd = 0
        self.bkgda = 0
        self.fill = 0
        self.ignore = False
        self.ignoreone = []

        std = []
        amb = []
        mass = []
        satur = []
#         self.component = component
#         self.vals = [val0, val1, val2]
#         self.unit = unit
#         self.satur = satur
        for rate in flow.stream:
            if rate.component is self.comp:
                if rate.unit.lower() == "mass":
                    mass.append(str(rate.vals[0]) + "*" + str(rate.vals[1]) + "*" +
                                str(rate.vals[2]))
        for rate in flow.post:
            if rate.component is self.comp:
                if rate.unit.lower() == "std":
                    std.append(str(rate.vals[0]))
                elif rate.unit.lower() == "amb":
                    amb.append(str(rate.vals[0]))
        self.rate = ["+".join(mass)]
        self.satur = [""]
        self.after = ["+".join(std)]
        if len(amb) > 0:
            if len(std) > 0:
                self.after.append("+")
            self.after += ["(", "+".join(amb), ")*", rc.Ts, "/", rc.T]

        for fill in flow.fill:
            if fill[0] is self.comp:
                self.fill += fill[1]
        for bkgd in flow.bkgd:
            if bkgd[0] is self.comp:
                if bkgd[2]:
                    self.bkgda += bkgd[1]
                else:
                    self.bkgd += bkgd[1]
        for ign in flow.ignore:
            if ign is self.comp:
                self.ignore = True
        for ign in flow.ignoreone:
            if ign[0] is self.comp:
                self.ignoreone.append(ign[1])

class Flowrate():

    def __init__(self, comps, flow, time, raw, counts):
        self.flow = flow
        self.hours = [(self.flow.hours[0] - 0.5, self.flow.hours[1] + 0.5)]
        ind0 = findval(time, flow.hours[0])
        ind1 = findval(time, flow.hours[1])
        if ind1 >= 0 and ind1 < len(time):
            ind1 += 1
        self.rng = (ind0, ind1)
        self.time = time[ind0:ind1]
        self.data = [Compdata(comps, c, counts[ind0:ind1, :], raw, flow)
                     for c in comps]
        self.std = self.data[-1]
        self.rxt = self.data[0]

        # determine flow values for the inert
        self.inertrate = []
        self.inertafter = []
        if rxn.inert is not None:
            std = 0
            amb = 0
            stda = 0
            amba = 0
            # self.satur = satur
            for rate in flow.stream:
                if rate.component is rxn.inert:
                    if rate.unit.lower() == "std":
                        std += rate.vals[0]
                    elif rate.unit.lower() == "amb":
                        amb += rate.vals[0]
            for rate in flow.post:
                if rate.component is rxn.inert:
                    if rate.unit.lower() == "std":
                        stda += rate.vals[0]
                    elif rate.unit.lower() == "amb":
                        amba += rate.vals[0]
            self.inertrate = [std, "+(", amb, ")*", rc.Ts, "/", rc.T]
            self.inertafter = [stda, "+(", amba, ")*", rc.Ts, "/", rc.T]

    def maketables(self, report):
        self.headerTbl = report.getNewDataTable()
        self.bkgdTbl = report.getNewDataTable()
        self.tableTbl = report.getNewDataTable()

    def excel(self, report, rxn, nmcells, mwcells, rfcells, rocells, sacell):
        header = self.headerTbl
        bkgd = self.bkgdTbl
        table = self.tableTbl
        after = report.getNewDataTable()
        header.add_space_above(1)
        after.add_space_above(1)

        header.add_first_col(["FLOW"] + 6 * [""])
        ratecl = header.add_datacol(["-rate--------", "="] + 5 * [""])
        header.add_datacol(["-------------", "sccm", "", "Reactant",
                            "Standard", "Inert", "total"])
        lst = ["-------------", "=", ""]
        if rxn.reactant is not None:
            lst += [rxn.reactant.name]
        else:
            lst += [""]
        if rxn.standard is not None:
            lst += [rxn.standard.name]
        else:
            lst += [""]
        if rxn.inert is not None:
            lst += [rxn.inert.name]
        else:
            lst += [""]
        lst += [""]
        mpcl = header.add_datacol(lst)
        vpcl = header.add_datacol(["-----------=", "% " + self.data[0].comp.name,
                            "Vaporizer"] + 4 * [""])
        header.add_datacol(["-temp--------", "{:0.0f}".
                            format(self.flow.temp.getAsKelvin()), "",
                            "g/min", "g/min", "sccm", ""])
        stcl = header.add_datacol(["-----------=", "K", "Saturator"] + 4 * [""])
        mscl = header.add_datacol(["-mass-------", "=", "", "deg C", "",
                                   "sccm", ""])
        afcl = header.add_datacol(["(w/o inert)--------------", "g/min", "After"] +
                           4 * [""])
        prcl = header.add_datacol(["", "=", "", "sccm", "sccm", "sccm", ""])
        header.add_datacol(["-----------=", "wt% std", "Moles:"] + 4 * [""])
        rxcl = header.add_datacol(["-rxnt---------", "=", "vaporizer"] +
                                  4 * [""])
        stur = header.add_datacol(["-----------=", "g/min", "saturator"] + 4 * [""])
        rctr = header.add_datacol(["-std----------", "=", ":reactor"] + 4 * [""])
        smpl = header.add_datacol(["-----------=", "g/min", "after"] +
                                  4 * [""])
        frcell = ratecl[1]
        conccell = mpcl[1]
        rctflow = rctr[3]
        stdflow = smpl[4]
        ratecl[1].setAsFormula([rctr[6], "*", rc.R, "*", rc.Ts, "/", rc.Ps,
                                "*1000"])
        rctmass = rxcl[1]
        rctmass.setAsFormula([smpl[3], "*", mwcells[1]])
        stdmass = rctr[1]
        stdmass.setAsFormula([smpl[4], "*", mwcells[0]])
        totmass = mscl[1]
        totmass.setAsFormula([rctmass, "+", stdmass])
        prcmass = prcl[1]
        prcmass.setAsFormula([stdmass, "/", totmass, "*100"])
        mpcl[1].setAsFormula([rctr[3], "/", rctr[6], "*100"])
        rxcl[6].setAsFormula(["sum(", rxcl[3], ":", rxcl[5], ")"])
        stur[6].setAsFormula(["sum(", stur[3], ":", stur[5], ")"])
        rctr[6].setAsFormula(["sum(", rctr[3], ":", rctr[5], ")"])
        rctr[5].setAsFormula([rxcl[5], "+", stur[5]])
        rctr[4].setAsFormula([rxcl[4], "+", stur[4]])
        rctr[3].setAsFormula([rxcl[3], "+", stur[3]])
        smpl[3].setAsFormula([afcl[3], "*", rc.Ps, "/", rc.R, "/", rc.Ts,
                              "/1000+", rctr[3]])
        smpl[4].setAsFormula([afcl[4], "*", rc.Ps, "/", rc.R, "/", rc.Ts,
                              "/1000+", rctr[4]])
        smpl[5].setAsFormula([afcl[5], "*", rc.Ps, "/", rc.R, "/", rc.Ts,
                              "/1000+", rctr[5]])
        smpl[6].setAsFormula(["sum(", smpl[3], ":", smpl[5], ")"])
        rxcl[3].setAsFormula([vpcl[3], "/", mwcells[1]])
        rxcl[4].setAsFormula([vpcl[4], "/", mwcells[0]])
        rxcl[5].setAsFormula([vpcl[5], "*", rc.Ps, "/", rc.R, "/", rc.Ts,
                              "/1000"])
        # todo: saturator
        # stur[3].setAsFormula([])
        # stur[4].setAsFormula([])
        # stur[5].setAsFormula([])
        stdflow.hard = True
        rctflow.hard = True
        vpcl[3].setAsFormula(self.rxt.rate)  # rxt
        afcl[3].setAsFormula(self.rxt.after)
        vpcl[4].setAsFormula(self.std.rate)  # std
        afcl[4].setAsFormula(self.std.after)
        vpcl[5].setAsFormula(self.inertrate)  # inert
        afcl[5].setAsFormula(self.inertafter)

        # background cells
        bkgdcells = []
        bkgd.add_first_col(["BKGD"] + [""] * 3)
        bkgd.add_datacol(["/----------=", "counts", "% conv", "mol / s"])
        def add_bkgd(name, nmcell, area, conv, noconv=False):
            column = bkgd.add_datacol([name, area, conv, "="])
            column[0].setAsFormula([nmcell])
            column[3]._refa = column[1]
            column[3]._ref = column[2]
            bkgdcells.append(column[3])
            if noconv:
                column[2].setAsText("")
                column[3].setAsText("")
            for cell in column:
                cell.hard = True
        add_bkgd(self.data[-1].comp.name, nmcells[0], self.std.bkgda,
                 self.std.bkgd, True)
        for g in range(0, len(self.data) - 1):
            add_bkgd(self.data[g].comp.name, nmcells[g + 1], self.data[g].bkgda,
                     self.data[g].bkgd, g == 0)
        bkgd.add_datacol(["", "counts", "% conv", "mol / s"])
        for g in range(2, len(self.data)):
            bkgdcells[g].setAsFormula(["IF(", bkgdcells[g]._ref,
                "<0,0,", bkgdcells[g]._ref, "/100*(" , rctflow, "/60)/",
                rocells[g], ")"])

        # time
        n = len(self.time)
        u = 3
        d = -1
        table.add_first_col(["DATA", "Index", "[#]"] +
                            [str(i) for i in range(self.rng[0], self.rng[1])] +
                            ["AVERAGE"])
        times = table.add_datacol(["", "Time", "[hr]"] + [str(t)
                                  for t in self.time] + [""])
        seps = table.add_plain_col("|")
        val = 0
        if self.flow.equil > 0 and self.flow.equil < n:
            val = self.flow.equil
        if self.flow.equil >= n:
            val = n - 1
        sub = n - self.flow.npts
        if self.flow.npts > 0 and self.flow.npts < n and sub > val:
            val = sub
        seps[u - 2].setAsText("-equil-")
        seps[u - 1].setAsText(val)
        seps[d].setAsFormula([n, "-", seps[u - 1]])
        avgcells = []
        def avg(cells, trip=None):
            if trip is None:
                trip = "1"
            lst = ["IFERROR(IF(", trip , ",AVERAGE(INDIRECT(ADDRESS(ROW(",
                   cells[d], ")-$", seps[d], ",COLUMN(", cells[d],
                   "))):", cells[d - 1], '),"-"),"?")']
            cells[d].setAsFormula(lst)
            avgcells.append(cells[d])
        def formatRT(rt):
            store = [""] * n
            for j in range(rt.shape[1]):
                if np.isnan(rt[2, j]):
                    continue
                i = int(round(rt[2, j])) - self.rng[0]
                if i < 0 or i >= n:
                    continue
                st = "{0:0.3f}".format(rt[1, j])
                if store[i] == "":
                    store[i] = st
                else:
                    store[i] += ", " + st
            return store
        # standard
        stdrt = table.add_datacol(["Standard:", "RT", "[mins]"] +
                                    ["="] * n)
        stdarea = table.add_datacol([self.std.comp.name, "Area", "[counts]"])
        stdarea[0].setAsFormula([nmcells[0]])
        avg(stdarea)
        store = formatRT(self.std.raw)
        for i in range(n):
            cell = stdrt[i + u]
            cell.setAsText(store[i])
            cell = stdarea[i + u]
            cell.setAsFormula(["{:0.0f}".format(self.std.counts[i]), "-",
                                bkgdcells[0]._refa])
        # reactant
        table.add_plain_col("|")
        rxtrt = table.add_datacol(["Reactant:", "RT", "[mins]"])
        areacells = table.add_datacol(["", "Area", "[counts]"])
        areacells[0].setAsFormula([nmcells[1]])
        store = formatRT(self.rxt.raw)
        for i in range(n):
            cell = rxtrt[i + u]
            cell.setAsText(store[i])
            cell = areacells[i + u]
            cell.setAsFormula(["{:0.0f}".format(self.rxt.counts[i]), "-",
                                bkgdcells[1]._refa])
        avg(areacells)
        masscells = table.add_datacol(["", "(mass ratio)",
            "[g r/g std]"] + ["="] * len(self.time))
        for cell, area, stda in zip(masscells[u:d],
                                    areacells[u:d], stdarea[u:d]):
            cell.setAsFormula(["(", area, "/", rfcells[1], ")/(",
                               stda, "/", rfcells[0], ")"])
        rctmole = table.add_datacol(["", "(moles)", "[mol/min]"] + ["="] * n)
        for cell, mass in zip(rctmole[u:d], masscells[u:d]):
            cell.setAsFormula([mass, "*(", mwcells[0], "/", mwcells[1], ")*",
                               stdflow])
        avg(rctmole)
        react = table.add_datacol(["", "Reacted", "[mol/s]"] + ["="] * n)
        ratecells = table.add_datacol(["", "Rate", "[mol/s*m^2]"] + ["="] * n)
        avg(ratecells)
        for rate, rct in zip(ratecells[u:d], react[u:d]):
            rate.setAsFormula([rct, "/", sacell])
        convcells = table.add_datacol(["", "Conv", "[%]"] + ["="] * n)
        for conv, rct in zip(convcells[u:d], react[u:d]):
            conv.setAsFormula([rct, "/(", rctflow, "/60)*100"])
        avg(convcells)
        # products
        consumed = []
        tblafter = []
        for g in range(1, len(self.data) - 1):
            data = self.data[g]
            nopkarea = data.fill
            table.add_plain_col("|")
            prdrt = table.add_datacol(["Product :", "RT", "[mins]"])
            areacells = table.add_datacol(["", "Raw Area", "[counts]"])
            areacells[0].setAsFormula([nmcells[g + 1]])
            areacell2 = table.add_datacol(["", "Area", "[counts]"])
            masscells = table.add_datacol(["", "Mass R", "[g i/g std]"])
            prdmass = table.add_datacol(["NoPkArea:", "Raw Moles", "[mol/min]"])
            moles = table.add_datacol([nopkarea, "Moles", "[mol/s]"])
            consm = table.add_datacol(["", "Consumed", "[mol/s]"])
            rates = table.add_datacol(["", "Rate", "[mol/s*m^2]"])
            select = table.add_datacol(["Include?:", "Selectivity", "[%]"])
            swchcells = table.add_datacol(["TRUE", "", "Use?"])
            if data.comp in rxn.focus:
                tblafter.append((data.comp, select, nmcells[g + 1]))
            nopk = moles[0]
            nopk.hard = True
            trip = swchcells[0]
            trip.hard = True
            deny = False
            if np.isnan(data.counts).all():
                # component not present at this flowrate
                deny = True
            elif np.isnan(data.counts).any():
                # component questionable, investigate
                num = 0
                h = (n + 1) / 2  # always rounds up, less stringent
                secondhalffail = True
                for i in range(n):
                    if np.isnan(data.counts[i]):
                        num += 1
                    elif i >= h:
                        secondhalffail = False
                if secondhalffail or num > h:
                    deny = True
            if data.ignore:
                deny = True
            if data.comp in rxn.disable:
                deny = True
            store = formatRT(data.raw)
            for i in range(u, u + n):
                rtcell = prdrt[i]
                stda = stdarea[i]
                rct = react[i]
                swch = swchcells[i]
                area = areacells[i]
                are2 = areacell2[i]
                mass = masscells[i]
                prdm = prdmass[i]
                mole = moles[i]
                cnsm = consm[i]
                rate = rates[i]
                sel = select[i]
                swch.setAsFormula(["IF(", are2, "<=0,FALSE,TRUE)"])
                rtcell.setAsText(store[i - u])
                val = data.counts[i - u]
                if np.isnan(data.counts[i - u]):
                    val = 0
                    area.setAsText("")
                else:
                    area.setAsText("{:0.0f}".format(val))
                are2.setAsFormula(["IF(", area, ">0, ", area, ", ",
                                   nopk, ")-", bkgdcells[g + 1]._refa])
                mass.setAsFormula(["(", are2, "/", rfcells[g + 1],
                        ")/(", stda, "/", rfcells[0], ")"])
                prdm.setAsFormula([mass, "*(", mwcells[0], "/", mwcells[g + 1],
                                   ")*", stdflow])
                mole.setAsFormula([prdm, "/60-", bkgdcells[g + 1]])
                cnsm.setAsFormula(["IF(", swch, ", IF(", trip, ",", mole, "*",
                                   rocells[g + 1], ', 0), 0)'])
                rate.setAsFormula(["IF(", swch, ",", mole, "/", sacell,
                                   ', "")'])
                sel.setAsFormula(["IF(", swch, ",", cnsm, "/", rct, "*100",
                                  ', "")'])
                for t in data.ignoreone:
                    if t == i - u:
                        swch.setAsText("FALSE")
            consumed.append(consm)
            if nopkarea > 0:
                deny = False
            if deny:
                trip.setAsText("FALSE")
            avg(areacells, trip)
            avg(rates, trip)
            avg(select, trip)
        for i in range(u, len(react) + d):
            rct = react[i]
            toadd = []
            for g in range(len(consumed)):
                if g > 0:
                    toadd.append("+")
                toadd.append(consumed[g][i])
            if len(consumed) == 0:
                toadd.append("0")
            rct.setAsFormula(toadd)

        # the table after
        template = [""] * (n + u)
        timshift = after.add_first_col(template)
        rxtrate = after.add_datacol(template)
        rxtconv = after.add_datacol(template)
        firstcol = None
        lastcol = None
        for c in rxn.focus:
            if c is not self.rxt and c is not self.std:
                prdsel = after.add_datacol(template)
                if firstcol is None:
                    firstcol = prdsel
                lastcol = prdsel
                notfound = True
                for (comp, sel, nm) in tblafter:
                    if comp is c:
                        notfound = False
                        for i in range(1, len(timshift)):
                            p = prdsel[i]
                            prod = sel[i]
                            p.setAsFormula([prod])
                        prdsel[1].setAsFormula([nm])
                        break
                if notfound:
                    if c is not None:
                        prdsel[1].setAsText(c.name)
        others = after.add_datacol(template)
        others[1].setAsText("Others")
        others[2].setAsText("[%]")
        if firstcol is not None:
            for i in range(3, len(timshift)):
                others[i].setAsFormula(["100-SUM(", firstcol[i], ":",
                       lastcol[i], ")"])
        for i in range(len(timshift)):
            tim = timshift[i]
            time = times[i]
            if i >= u:
                tim.setAsFormula([time, "-", timshift[0]])
            else:
                tim.setAsFormula([time])
        timshift[0].setAsFormula([times[u]])
        for i in range(1, len(timshift)):
            r = rxtrate[i]
            rate = ratecells[i]
            r.setAsFormula([rate])
        for i in range(1, len(timshift)):
            c = rxtconv[i]
            conv = convcells[i]
            c.setAsFormula([conv])

        return avgcells, rctflow, frcell, conccell


def add_xls_header(report, header, props, components, rxn):
    nmcells = []
    mwcells = []
    rfcells = []
    rocells = []
    name = rxn.catalyst
    if rxn.catdate != "":
        name += " (" + rxn.catdate + ")"
    cells = header.add_headings(["Catalyst", name, "", "Loading",
                                 "{:0.3f}".format(rxn.loading), "g", "",
                                 "SA", "{:0.2f}".format(rxn.surfarea),
                                 "m^2/g", "", "Area", "=", "m^2", "",
                                 "Rxn Date:", rxn.rxndate])
    loadcell = cells[4]
    loadcell.hard = True
    sacell = cells[12]
    sacell.hard = True
    sacell.setAsFormula([loadcell, "*", rxn.surfarea])

    props.add_first_col(["Properties:"] + [""] * 3)
    props.add_datacol(["", "RF", "MW", "Rxt:P"])
    for g in range(-1, len(components) - 1):
        ratio = 0
        if components[0].stoich is not None and components[g].stoich is not None:
            ratio = (1.0 * components[g].stoich / components[0].stoich)
        cells = props.add_datacol([components[g].name,
            components[g].resp_factor, components[g].mol_weight, ratio])
        for cell in cells:
            cell.hard = True
        nmcells.append(cells[0])
        rfcells.append(cells[1])
        mwcells.append(cells[2])
        rocells.append(cells[3])
    props.add_datacol(["", "RF", "MW", "Rxt:P"])

    return nmcells, mwcells, rfcells, rocells, loadcell, sacell

def add_xls_summary(quick, summary, rxn, comps, flows, avgcells,
                    rxtcells, flowcells, conccells, loadcell):
    # quick summary
    quick.add_first_col(["Catalyst", ""] + [rxn.catalyst] * len(flows))
    quick.add_datacol(["Synth Date", ""] +
                      [rxn.catdate] * len(flows))
    quick.add_datacol(["SurfArea", "[m2 g-1]"] + [
                       "{:0.3f}".format(rxn.surfarea)] * len(flows))
    quick.add_datacol(["Loading", "[g]"] + ["{:0.3f}".format(rxn.loading)] *
                      len(flows))
    quick.add_datacol(["Temp", "[K]"] +
                          [flow.temp.getAsKelvin() for flow in flows])
    fr = quick.add_datacol(["Flowrate", "[sccm]"] +
                          ["tbd" for flow in flows])
    cn = quick.add_datacol(["Conc", "[%]"] + ["tbd" for flow in flows])
    pr = quick.add_datacol(["P_" + comps[0].name.capitalize(), "[kPa]"] +
            ["tbd*" + str(flow.pres.get("kPa")) for flow in flows])
    conv = quick.add_datacol([comps[0].name.capitalize(), "[% conv]"])
    rate = quick.add_datacol(["", "[mol s-1 m-2]"])
    for i in range(len(flows)):
        fr[i + 2].setAsFormula(["round(", flowcells[i], ",0)"])
        cn[i + 2].setAsFormula(["round(", conccells[i], ",1)"])
        pr[i + 2].setAsFormula(["round(", conccells[i], "/100*" +
                                str(flows[i].pres.get("kPa")), ", 2)"])
    for i in range(len(avgcells)):
        avg = avgcells[i]
        conv[i + 2].setAsFormula(["ROUND(", avg[4], ",2)"])
        rate[i + 2].setAsFormula([avg[3]])
    first = "Selectivity [%]"
    grab = None
    last = None
    for comp in rxn.focus:
        nm = ""
        if comp is not None:
            nm = comp.name
        sel = quick.add_datacol([first, nm])
        if first != "":
            grab = sel
        first = ""
        last = sel
        g = -1;
        for i in range(len(comps)):
            if comp is comps[i]:
                g = i
                break
        if g > 0:
            for i in range(len(avgcells)):
                avg = avgcells[i]
                if avg[g * 3 + 4] is not None:
                    sel[i + 2].setAsFormula(["IFERROR(ROUND(", avg[g * 3 + 4],
                                              ",1)", ",\"-\")"])
    if grab is not None and last is not None:
        sig = quick.add_datacol(["", "Others"])
        for i in range(len(avgcells)):
            sig[i + 2].setAsFormula(["100-SUM(", grab[i + 2], ":", last[i + 2],
                                      ")"])
    quick.add_plain_row()

    # full summary
    summary.add_first_col(["Temp", "[K]"] +
                          [flow.temp.getAsKelvin() for flow in flows])
    fr = summary.add_datacol(["Flowrate", "[sccm]"] +
                          ["tbd" for flow in flows])
    cn = summary.add_datacol(["Conc", "[%]"] + ["tbd" for flow in flows])
    pr = summary.add_datacol(["P_" + comps[0].name.capitalize(), "[kPa]"] +
            ["tbd*" + str(flow.pres.get("kPa")) for flow in flows])
    conv = summary.add_datacol([comps[0].name.capitalize(), "[% conv]"])
    cnv2 = summary.add_datacol(["", "RF conv"])
    rate = summary.add_datacol(["", "[mol s-1 m-2]"])
    for i in range(len(flows)):
        fr[i + 2].setAsFormula([flowcells[i]])
        cn[i + 2].setAsFormula([conccells[i]])
        pr[i + 2].setAsFormula([conccells[i], "/100*" +
                                str(flows[i].pres.get("kPa"))])
    for i in range(len(avgcells)):
        avg = avgcells[i]
        conv[i + 2].setAsFormula(["ROUND(", avg[4], ",2)"])
        cnv2[i + 2].setAsFormula(["ROUND((1-", avg[2], "/", rxtcells[i], ")*100,2)"])
        rate[i + 2].setAsFormula([avg[3]])
    first = "Selectivity [%]"
    grab = None
    last = None
    for g in range(1, len(comps) - 1):
        sel = summary.add_datacol([first, comps[g].name])
        if first != "":
            grab = sel
        first = ""
        last = sel
        for i in range(len(avgcells)):
            avg = avgcells[i]
            if avg[g * 3 + 4] is not None:
                sel[i + 2].setAsFormula(["IFERROR(ROUND(", avg[g * 3 + 4],
                                          ",1)", ",\"-\")"])
    if grab is not None and last is not None:
        sig = summary.add_datacol(["Sum", "[%]"])
        for i in range(len(avgcells)):
            sig[i + 2].setAsFormula(["SUM(", grab[i + 2], ":", last[i + 2],
                                      ")"])
    first = "Rates [mol/s*m^2]"
    for g in range(1, len(comps) - 1):
        rate = summary.add_datacol([first, comps[g].name])
        first = ""
        for i in range(len(avgcells)):
            avg = avgcells[i]
            if avg[g * 3 + 3] is not None:
                rate[i + 2].setAsFormula([avg[g * 3 + 3]])
    flow = summary.add_datacol(["F", "[mol hr-1]"])
    whfr = summary.add_datacol(["W/F", "[g hr mol-1]"])
    for i in range(len(rxtcells)):
        whfr[i + 2].setAsFormula([loadcell, "/", flow[i + 2]])
    for i in range(len(rxtcells)):
        flow[i + 2].setAsFormula([rxtcells[i], "*60"])
    summary.add_plain_row()

class PeakList():

    def __init__(self):
        self.times = None
        self.peak_master = []

        self.ret_times = None
        self.unassigned = None
        self.assigned = None

    def set_times(self, times):
        self.times = times

    def get_time(self, index):
        if self.times is None:
            raise ValueError("No times exist for this peak list")
        if len(self.times) == 0:
            return None
        return self.times[index]

    def add_peak(self, index, rt, area, leading=None, tailing=None):
        while index >= len(self.peak_master):
            self.peak_master.append([])
        self.peak_master[index].append(PeakList.Peak(leading, rt, tailing, area))

    def get_peaks(self, index):
        if index >= len(self.peak_master):
            return None
        return self.peak_master[index]

    def get_peaks_when(self, t0, t1):
        if self.times is None:
            raise ValueError("No times exist for this peak list")
        if len(self.times) == 0:
            return None
        if t0 > t1:
            t1, t0 = t0, t1

        pl = PeakList()
        # check max/min
        if t0 > np.max(self.times):
            return pl
        if t1 < np.min(self.times):
            return pl

        i0 = -1
        i1 = -1
        for i in xrange(len(self.times)):
            if i0 < 0 and t0 >= self.times[i]:
                i0 = i
            if t1 <= self.times[i]:
                i1 = i
            else:
                break
        if i0 == -1:
            i0 = 0
        if i1 == -1:
            i1 = len(self.times)

        pl.times = self.times[i0:i1]
        pl.peak_master = self.peak_master[i0:i1]
        return pl

    def sort(self):
        for ind in xrange(len(self.peak_master)):
            sublist = self.peak_master[ind]
            temp = []
            for _ in xrange(len(sublist)):
                peak = sublist.pop()
                r = 0
                found = False
                for r, p in enumerate(temp):
                    if peak.rt < p.rt:
                        found = True
                        break
                if not found:
                    r += 1
                temp.insert(r, peak)
            sublist.extend(temp)

    def get_retention_times(self):
        return self.ret_times

    def get_assigned_peaks(self, rt_index):
        if self.assigned is None:
            return None
        node = self.assigned[rt_index]
        pl = PeakList()
        pl.times = self.times
        for i, p in zip(node.ind, node.peaks):
            while i >= len(pl.peak_master):
                pl.peak_master.append([])
            pl.peak_master[i].append(p)
        return pl

    def get_unassigned_peaks(self):
        pl = PeakList()
        pl.times = self.times
        pl.peak_master = self.unassigned
        return pl

    def sort_retention_times(self):
        pass

    def group_retention_times(self):
        self.unassigned = [list() for _ in range(len(self.peak_master))]
        nodes = []
        for ind in xrange(len(self.peak_master)):
            sublist = self.peak_master[ind]
            if ind == 0:
                nodes = [PeakList.Node(p, 0) for p in sublist]
            else:
                for n in nodes:
                    n.test(sublist)
                for p in sublist:
                    if len(p.nodes) == 0:
                        node = PeakList.Node(p, ind)
                        add = False
                        for i, n in enumerate(nodes):
                            if node.avg_rt < n.avg_rt:
                                nodes.insert(i, node)
                                add = True
                                break
                        if not add:
                            nodes.append(node)
                    elif len(p.nodes) == 1:
                        p.nodes[0].apply(p, ind)
                    else:  # what happens if two peaks have the same node listed?
                        diffs = []
                        for n in p.nodes:
                            diffs.append(abs(p.area - n.peaks[-1].area))
                        mind = min(diffs)
                        for i, d in enumerate(diffs):
                            if mind == d:
                                p.nodes[i].apply(p, ind)
                                break
                to_remove = []
                for n in nodes:
                    if n.age > 3 and len(n.peaks) <= 3:
                        to_remove.append(n)
                for n in to_remove:
                    nodes.remove(n)
                    # print "Removing node {0:0.0f} , with {1:0.0f} peaks".format(n.id, len(n.peaks))
                    for p, i in zip(n.peaks, n.ind):
                        self.unassigned[i].append(p)
                        # print "\tunassigned[{0:0.0f}] has {1:0.0f} peaks".format(i, len(self.unassigned[i]))
                    n.peaks = []
                    n.ind = []

            for p in sublist:
                p.nodes = []
            for n in nodes:
                n.reset()
        self.assigned = nodes
        self.ret_times = []
        for n in nodes:
            self.ret_times.append(n.avg_rt)

    def size(self):
        return len(self.peak_master)

    def num_peaks(self):
        ret = 0
        for sub in self.peak_master:
            ret += len(sub)
        return ret

    class Peak():
        def __init__(self, lead, rt, tail, area):
            self.rt = rt
            self.area = area
            self.leading = lead
            self.tailing = tail

            self.nodes = []

    class Node():
        count = -1;

        def __init__(self, start=None, ind=None):
            PeakList.Node.count += 1
            self.id = PeakList.Node.count
            self.age = 0
            self.peaks = []
            self.ind = []
            self.conflict = None

            self.avg_rt = 0
            if start is not None:
                self.start(start, ind)

        def reset(self):
            self.conflict = None

        def start(self, peak, ind=None):
            self.peaks.append(peak)
            self.avg_rt = peak.rt
            self.age = 0
            if ind is not None:
                self.ind.append(ind)
            else:
                self.ind.append(0)

        def apply(self, peak, ind):
            self.peaks.append(peak)
            self.ind.append(ind)
            num = len(self.peaks)
            self.avg_rt = self.avg_rt * (num - 1.0) / num + peak.rt * 1.0 / num
            self.age = 0

        def test(self, sublist):
            rt = self.avg_rt
            for peak in sublist:
                diff = peak.rt - rt
                # hard limit a diff of +/- 3 minutes
                if -3 < diff and diff < 3:
                    last = self.peaks[-1]
                    if last.tailing is not None and last.leading is not None:
                        if last.leading < peak.rt and peak.rt < last.tailing:
                            peak.nodes.append(self)
                    else:
                        if rt - 0.35 < peak.rt and peak.rt < rt + 0.35:
                            peak.nodes.append(self)
            self.age += 1

class ComponentData():

    def __init__(self, component):
        self.comp = component
        self.name = "Others"
        self.rt = 0
        self.adj = 0
        if self.comp is not None:
            self.name = component.value
            if "rt" in component.attr:
                self.rt = float(component.attr["rt"])
            if "adj" in component.attr:
                self.adj = float(component.attr["adj"])
        self.peaks = []

    def test_peak(self, ret, area, index, time):
        add = False
        if self.comp is None:
            add = True
        else:
            up = self.rt + self.adj
            dn = self.rt - self.adj
            if dn <= ret and ret <= up:
                add = True
        if add:
            self.peaks.append((index, time, ret, area))
        return add

    def num_peaks(self):
        return len(self.peaks)

    def get_peaks(self, ref_time, hr_start=0, hr_end=np.Inf):
        sub_list = []
        for (i, t, r, a) in self.peaks:
            diff = (t - ref_time).total_seconds() / 60.0 / 60.0
            if hr_start <= diff and diff <= hr_end:
                sub_list.append((i, t, r, a))
        return sub_list

    def get_all_peaks(self):
        ret = []
        ret.extend(self.peaks)
        return ret


def import_file_data(datafile, components, agilent):
    datafiles = ac.read_datafiles(datafile, agilent)

    comp_list = []
    for comp in components.subcontent:
        if not "rt" in comp.attr:
            continue
        i = 0
        for i in xrange(len(comp_list)):
            if comp.attr["rt"] < comp_list[i].attr["rt"]:
                break
        comp_list.insert(i, comp)

    comp_data = []
    for comp in comp_list:
        comp_data.append(ComponentData(comp))
    comp_data.append(ComponentData(None))

    # ac.OVERRIDE_BASELINE = True
    # ac.OVERRIDE_PEAKS = True
    # ac.OVERRIDE_SELECTION = True

    pl = PeakList()
    plots = []
    times = []
    tot = len(datafiles)
    for ind, df in enumerate(datafiles):
        print("Processing {0} of {1}: {2}".format(ind + 1, tot, df.name))
        peaks, cg_plot = ac.process_datafile(df, skip=True)
        if cg_plot is not None:
            plots.append(cg_plot)

        time = df.datetime
        times.append(time)
        for p in peaks:
            ret = p.retention
            area = p.area
            pl.add_peak(ind, ret, area, p.leading, p.tailing)
            for cd in comp_data:
                add = cd.test_peak(ret, area, ind, time)
                if add:
                    break
    pl.set_times(times)

    ref_time = datafiles[0].datetime
    return comp_data, ref_time, plots, pl

def analyze(rxn):
    folder = os.path.join(rxn.info.attr["folderout"], "")
    if not os.path.isdir(folder):
        os.mkdir(folder)

    filenm = None
    agilent = False
    if "shimadzu" in rxn.info.attr:
        filenm = rxn.info.attr["shimadzu"]
    if "agilent" in rxn.info.attr:
        filenm = rxn.info.attr["agilent"]
        agilent = True

    if filenm is not None:
        comp_data, ref_time, plots, peak_list = import_file_data(filenm, rxn.comps, agilent)
        os.chdir(folder)
        if len(plots) > 0:
            pr.plot_several("chromatograms", plots)

        num_peaks = 0
        for cd in comp_data:
            num_peaks += cd.num_peaks()

        i = 0
        times = np.zeros(num_peaks)
        rettimes = []
        for cd in comp_data:
            ln = cd.num_peaks()
            if ln > 0:
                vals = np.zeros(num_peaks) * np.NaN
                peaks = cd.get_all_peaks()
                for j in xrange(0, ln):
                    (_, t, r, _) = peaks[j]
                    times[i + j] = (t - ref_time).total_seconds() / 60.0 / 60.0
                    vals[i + j] = r
                rettimes.append(vals)
                i += ln
        plotdata = pr.PlotData(times, rettimes, xlabel="Reaction Time (hr)",
                               ylabel="Retention Time (min)")
        plotdata.dots = True
        plotdata.lines = False
        plotsetup = pr.PlotSetup([plotdata], "retention times")
        pr.plot_results(plotsetup)

        pr.pyplot.clf()
        n_peaks = peak_list.num_peaks()
        peak_list.sort()
        peak_list.group_retention_times()
        times = np.zeros(n_peaks)
        values = []
        unlist = peak_list.get_unassigned_peaks()
        ret_times = peak_list.get_retention_times()
        i = 0  # index so far along list of peaks
        if ret_times is not None:
            for r, rt in enumerate(ret_times):
                comp = peak_list.get_assigned_peaks(r)
                ln = comp.num_peaks()
                if ln > 0:
                    vals = np.zeros(num_peaks) * np.NaN
                    j = 0
                    for ind in xrange(comp.size()):
                        t = comp.get_time(ind)
                        for pk in comp.get_peaks(ind):
                            times[i + j] = (t - ref_time).total_seconds() / 60.0 / 60.0
                            vals[i + j] = pk.rt
                            j += 1
                    values.append(vals)
                    i += ln
        ln = unlist.num_peaks()
        if ln > 0:
            vals = np.zeros(num_peaks) * np.NaN
            j = 0
            for ind in xrange(unlist.size()):
                t = unlist.get_time(ind)
                for pk in unlist.get_peaks(ind):
                    times[i + j] = (t - ref_time).total_seconds() / 60.0 / 60.0
                    vals[i + j] = pk.rt
                    j += 1
            values.append(vals)
        plotdata = pr.PlotData(times, values, xlabel="Reaction Time (hr)",
                               ylabel="Retention Time (min)")
        plotdata.dots = True
        plotdata.lines = False
        plotsetup = pr.PlotSetup([plotdata], "retention times plus")
        pr.plot_results(plotsetup)

if __name__ == '__main__':
    for fold_in in FOLDER_IN:
        ref = rc.Reference()
        ref.load(GLOBAL_SETTINGS)
        for setup in SETUPS:
            os.chdir(fold_in)
            rxn = rc.Details()
            rxn.load(setup)
            analyze(rxn)

























