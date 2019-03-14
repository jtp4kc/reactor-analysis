'''
Created on Mar 5, 2019

@author: Tyler P.
'''

import os

PATH = os.path.join("C:/", "Users", "adjun_000", "Documents", "Grad School",
                    "Grad School Research (PhD)", "Experiments", "Reactions")
def_file = "analysis_settings.txt"
definitions = None
config_files = [
    ("", "20190205_5MgOonSiO2-IWI", "details.txt"),
    ]

# Constants
R = 8.3144621  # L*kPa / K*mol
T = 273.15 + 25  # K, ambient
P = 101.325  # kPa, ambient
Ts = 273.15  # K, standard
Ps = 101.325  # kPa, standard

# Test Flags
TEST_CONFIG = False
TEST_PEAKS = False
TEST_OUTPUT = False

class Peaks():
    def init(self, rxn_data):
        pass

class Output():
    def save_checkpoint(self, setup, rxn_data, peaks):
        pass

    def read_checkpoint(self):
        return None, None

    def save_analysis(self, setup, rxn_data, peaks):
        pass

class Content():

    def __init__(self):
        self.name = "content"
        self.attr = dict()
        self.subcontent = []
        self.flags = []
        self.comment = ""

class Subcontent():

    def __init__(self):
        self.value = "subcontent"
        self.attr = dict()
        self.flags = []
        self.comment = ""

flag_ch = "~"
attr_ch = "="
cmnt_ch = "#"
hidn_ch = "_"

class _SetupFile():

    def __init__(self):
        self.header = []
        self.attr = dict()
        self.content = []
        self.flags = []

        self._load_content = None

    def _load(self, filename):
        with open(filename, "r") as inputfile:
            lino = 0
            header = True
            for line in inputfile:
                tline = line.strip()
                if lino == 0:
                    if tline != "Reaction Conditions v2":
                        print("Warning: file version not recognized- " + filename)
                    lino = 1
                    continue
                if header and tline.startswith(cmnt_ch):
                    head = tline.lstrip(cmnt_ch)
                    self.header.append(head)
                else:
                    header = False
                if not header and tline != "":
                    if tline.startswith(cmnt_ch):
                        comment = tline[1:]
                        self._comment(comment)
                    else:
                        comment = None
                        testcomment = tline.split(cmnt_ch)
                        if len(testcomment) > 1:
                            tline = testcomment[0].strip()
                            comment = cmnt_ch.join(testcomment[1:])
                        if attr_ch in tline:
                            parts = tline.split(attr_ch)
                            tparts = [p.strip() for p in parts]
                            tline = attr_ch.join(tparts)
                        read = tline.split()
                        keys = []
                        values = []
                        entry = None
                        flags = []
                        pno = 0
                        lastop = None
                        for piece in read:
                            if attr_ch in piece:
                                keyval = piece.split(attr_ch)
                                keys.append(keyval[0])
                                values.append(keyval[1])
                                lastop = "attr"
                            elif piece.startswith(flag_ch):
                                flags.append(piece[1:])
                                lastop = "flag"
                            elif pno == 0:
                                entry = piece
                                lastop = "entry"
                            else:
                                if lastop == "attr":
                                    values[-1] += " " + piece
                                if lastop == "flag":
                                    flags[-1] += " " + piece
                                if lastop == "entry":
                                    entry += " " + piece
                            pno += 1
                        if line.lstrip() != line:
                            self._sub(entry, keys, values, flags, comment)
                        else:
                            self._main(entry, keys, values, flags, comment)
                lino += 1
            self._endoffile()
            inputfile.close()

    def _write(self, filename, append=False):
        access = "w"
        if append:
            access = "a"
        with open(filename, access) as outfile:
            outfile.write("Reaction Conditions v2\n")
            for line in self.header:
                out = line
                if not line.startswith(cmnt_ch):
                    out = cmnt_ch + line
                outfile.write(out + "\n")
            for flag in self.flags:
                out = flag
                if not out.startswith(flag_ch):
                    out = flag_ch + out
                outfile.write(out + "\n")
            for key in self.attr:
                if key.startswith(hidn_ch):
                    continue
                outfile.write("{0:9s}{1}{2:1s}".format(key, attr_ch, str(self.attr[key])))
                if ("_" + key) in self.attr:
                    val = self.attr[hidn_ch + key]
                    if not val.startswith(cmnt_ch):
                        val = cmnt_ch + val
                    outfile.write("  " + val)
                outfile.write("\n")
            outfile.write("\n")
            for cn in self.content:
                out = str(cn.name)
                if cn.comment is not None:
                    out += "  " + str(cn.comment)
                outfile.write(out + "\n")
                for flag in cn.flags:
                    out = flag
                    if not out.startswith(flag_ch):
                        out = flag_ch + out
                    outfile.write("\t" + out + "\n")
                for key in cn.attr:
                    if key.startswith(hidn_ch):
                        continue
                    outfile.write("\t{0:9s}={1:1s}".format(key, str(cn.attr[key])))
                    if (hidn_ch + key) in cn.attr:
                        val = str(cn.attr[hidn_ch + key])
                        if not val.startswith(cmnt_ch):
                            val = cmnt_ch + val
                        outfile.write("  " + val)
                    outfile.write("\n")
                for sc in cn.subcontent:
                    line = "\t" + sc.value
                    if len(sc.attr) != 0:
                        for key in sc.attr:
                            if key.startswith(hidn_ch):
                                continue
                            line += " \t" + "{0:1s}{1}{2:1s}".format(key, attr_ch, str(sc.attr[key]))
                    if len(sc.flags) != 0:
                        line += " \t" + " \t".join([flag_ch + str(flag) for flag in sc.flags])
                    if sc.comment is not None and sc.comment != "":
                        line += "  "
                        if not sc.comment.startswith(cmnt_ch):
                            line += cmnt_ch
                        line += sc.comment
                    outfile.write(line + "\n")

    def _main(self, entry=None, keys=None, values=None, flags=None, comment=None):
        if entry is not None:
            if self._load_content is not None:
                self.content.append(self._load_content)
                self._load_content = None
            content = Content()
            content.name = entry
            if keys is not None and values is not None:
                for k, v in zip(keys, values):
                    content.attr[k] = v
            if flags is not None:
                content.flags.extend(flags)
            if comment is not None:
                content.comment = comment
            self._load_content = content
            return
        if keys is not None and values is not None:
            for k, v in zip(keys, values):
                self.attr[k] = v
                if comment is not None:
                    self.attr["_" + k] = comment
        if flags is not None:
            self.flags.extend(flags)

    def _sub(self, entry=None, keys=None, values=None, flags=None, comment=None):
        if self._load_content is None:
            self._main(entry, keys, values, flags, comment)
            return
        if entry is not None:
            sc = Subcontent()
            sc.value = entry
            if keys is not None and values is not None:
                for k, v in zip(keys, values):
                    sc.attr[k] = v
                    if comment is not None:
                        sc.attr[hidn_ch + k] = comment
            if flags is not None:
                sc.flags = flags
            if comment is not None:
                sc.comment = comment
            self._load_content.subcontent.append(sc)
        else:
            if keys is not None and values is not None:
                for k, v in zip(keys, values):
                    self._load_content.attr[k] = v
                    if comment is not None:
                        self._load_content.attr[hidn_ch + k] = comment
            if flags is not None:
                self._load_content.flags.extend(flags)

    def _comment(self, comment):
        pass

    def _endoffile(self):
        if self._load_content is not None:
            self.content.append(self._load_content)
            self._load_content = None

class Reference(_SetupFile):

    def __init__(self):
        _SetupFile.__init__(self)
        self.focus = None
        self.ref = None

    def load(self, filename):
        if filename is not None:
            self._load(filename)
        for cn in self.content:
            if cn.name == "reference":
                self.ref = cn
            if cn.name == "focus":
                self.focus = cn

    def save(self, filename):
        if filename is not None:
            self._write(filename)

class Details(_SetupFile):

    def __init__(self):
        _SetupFile.__init__(self)
        self.info = None
        self.comps = None
        self.flows = []

    def _test(self):
        self.header = ["Test of setup file output", "Multiline"]
        self.attr.clear()
        self.content = []
        self.flags = ["testfile"]

        info = Content()
        info.name = "info"
        info.attr["catalyst"] = "NotACat"
        info.attr["catdate"] = 20181129
        info.attr["surfarea"] = 0.0
        info.attr["_surfarea"] = "m^2/g"
        info.attr["loading"] = 0.0
        info.attr["_loading"] = "g"
        info.attr["agilent"] = ""
        info.attr["_agilent"] = "filename.ext"
        info.attr["shimadzu"] = ""
        info.attr["_shimadzu"] = "basename (assumes -000.txt)"
        info.attr["folderout"] = "Results"
        info.flags.append("test flag")

        comp1 = Subcontent()
        comp1.value = "nitrogen"
        comp1.attr["rt"] = 0.00
        comp1.flags.append("inert")

        comp2 = Subcontent()
        comp2.value = "methane"
        comp2.attr["rt"] = 7.0
        comp2.flags.append("standard")

        comp3 = Subcontent()
        comp3.value = "ethanol"
        comp3.attr["rt"] = 27.0
        comp3.flags.append("reactant")

        comp4 = Subcontent()
        comp4.value = "acetaldehyde"
        comp4.attr["rt"] = 19.0

        comp5 = Subcontent()
        comp5.value = "diethyl ether"
        comp5.attr["rt"] = 33.0

        components = Content()
        components.name = "components"
        components.subcontent.append(comp1)
        components.subcontent.append(comp2)
        components.subcontent.append(comp3)
        components.subcontent.append(comp4)
        components.subcontent.append(comp5)

        eth1 = Subcontent()
        eth1.value = "ethanol"
        eth1.attr["vol"] = 0.018
        eth1.attr["density"] = 0.789

        eth2 = Subcontent()
        eth2.value = "ethanol"
        eth2.attr["vol"] = 0.006
        eth2.attr["density"] = 0.789

        nit1 = Subcontent()
        nit1.value = "nitrogen"
        nit1.attr["vol"] = 100.10

        nit2 = Subcontent()
        nit2.value = "nitrogen"
        nit2.attr["vol"] = 33.40

        meth = Subcontent()
        meth.value = "methane"
        meth.attr["vol"] = 5.00
        meth.flags.append("after")

        ignr = Subcontent()
        ignr.value = "1-butene"
        ignr.flags.append("ignore")

        flow1 = Content()
        flow1.name = "flow"
        flow1.attr["start"] = "11-29-2018 7:34PM"
        flow1.attr["stop"] = "11-29-2018 7:41PM"
        flow1.attr["pres"] = 4.7
        flow1.attr["_pres"] = "psig"
        flow1.attr["temp"] = 360
        flow1.attr["_temp"] = "Celsius"
        flow1.attr["equil"] = 0
        flow1.subcontent.append(eth1)
        flow1.subcontent.append(nit1)
        flow1.subcontent.append(meth)

        flow2 = Content()
        flow2.name = "flow"
        flow2.attr["start"] = "11-30-2018 1:00PM"
        flow2.attr["stop"] = "11-30-2018 3:00PM"
        flow2.subcontent.append(eth2)
        flow2.subcontent.append(nit2)
        flow2.subcontent.append(meth)
        flow2.subcontent.append(ignr)

        self.content.append(info)
        self.content.append(components)
        self.content.append(flow1)
        self.content.append(flow2)

    def load(self, filename):
        if filename is None:
            self._test()
        else:
            self._load(filename)
        for cn in self.content:
            if cn.name == "info":
                self.info = cn
            if cn.name == "components":
                self.comps = cn
            if cn.name == "flow":
                self.flows.append(cn)

    def save(self, filename):
        if filename is None:
            self._test()
        else:
            self._write(filename)

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

def test_config():
    test = Details()
    test._test()
    test._write(".test_reactor_conditions_output.txt")
    test2 = Details()
    test2.flags.append("main test")
    test2._load(".test_reactor_conditions_output.txt")
    test2._write(".test_reactor_conditions_input.txt")

    temp = Details()
    temp._load("rc_template.txt")
    temp._write("rc_template-test.txt")

class Setup():
    def __init__(self, config):
        self.info = Details()
        self.info.load(config)
        if TEST_CONFIG:
            self.info.attr["orig_file"] = config
            self.info.save("TEST_" + config)

    def load_data(self):
        return None

    def has_checkpoint(self):
        return None

class Reaction():
    pass

class BasicPeaks(Peaks):
    def init(self, rxn_data):
        pass

class BasicOutput(Output):
    def save_checkpoint(self, setup, rxn_data, peaks):
        pass

    def read_checkpoint(self):
        return None, None

    def save_analysis(self, setup, rxn_data, peaks):
        pass

def analyze(config):
    setup = Setup(config)
    peak_handler = BasicPeaks()
    output_manager = BasicOutput()
    if not setup.has_checkpoint():
        rxn_data = setup.load_data()
        peak_handler.init(rxn_data)
        output_manager.save_checkpoint(setup, rxn_data, peak_handler)
    else:
        rxn_data, peak_handler = output_manager.read_checkpoint(setup)

    output_manager.save_analysis(setup, rxn_data, peak_handler)

if __name__ == '__main__':
    ref_path = os.path.join(PATH, def_file)
    definitions = Reference()
    if os.path.exists(ref_path):
        definitions.load(ref_path)
    if TEST_CONFIG:
        definitions.attr["orig_path"] = repr(ref_path)
        definitions.save(os.path.join(PATH, "TEST_" + def_file))
        os.chdir(PATH)
        test_config()
    for cf in config_files:
        folder = os.path.join(PATH, cf[0], cf[1])
        os.chdir(folder)
        if os.path.exists(cf[2]):
            analyze(cf[2])
        else :
            print("Unable to find " + cf[2] + " in " + folder)


















