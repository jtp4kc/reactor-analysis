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
    ("", "20190205_5MgOonSiO2-IWI", "setup.txt"),
    ]

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

class Config():
    def __init__(self, filename):
        self.filename = filename
        self.prefix = "_"
        self.root = Config.Subentry()

    def save(self):
        base = os.path.basename(self.filename)
        filename = os.path.join(os.path.dirname(self.filename),
                                self.prefix + base)
        with open(filename) as outfile:
            outfile.write("#Reactor Analysis Beta - Configuration File\n\n")
            self._recursive_write("", outfile, self.root)
            outfile.close()

    def read(self):
        pass

    def _recursive_write(self, pre, out, entry):
        for comment in entry.cmnt:
            out.write(pre + "#" + comment + "\n")
        for flag in entry.flag:
            out.write(pre + flag + "\n")
        for key in entry.attr:
            if not key.startswith("_"):
                val = entry.attr[key]
                out.write(pre + key + " = " + val)
                if "_" + key in entry.attr:
                    cmnt = entry.attr["_" + key]
                    out.write(" #" + cmnt + "\n")
                else:
                    out.write("\n")
        for item in entry.item:
            out.write(pre + item.name + "\n")
            self._recursive_write(pre + "\t", out, item)

    class Subentry():
        def __init__(self):
            self.attr = dict()
            self.item = list()
            self.flag = list()
            self.cmnt = list()
            self.name = "."

class Common():
    def __init__(self, file):
        self.config = Config(file)
        if os.path.exists(file):
            self.config.read()

        self.data = self.config.root()
        found = False
        for item in self.data.item:
            if "focus" == item.name:
                found = True
                break
        if not found:
            item = Config.Subentry()
            item.name = "focus"
            self.data.item.append(item)

class Setup():
    def __init__(self, config):
        pass

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
    definitions = Common(os.path.join(PATH, def_file))
    for cf in config_files:
        folder = os.path.join(PATH, cf[0], cf[1])
        os.chdir(folder)
        if os.path.exists(cf[2]):
            analyze(cf[2])
        else :
            print("Unable to find " + cf[2] + " in " + folder)


















