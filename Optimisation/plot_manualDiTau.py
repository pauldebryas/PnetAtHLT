import pandas as pd
import json
import numpy as np

def plot(xaxis, yaxis, parameters, x_title, y_title, min_x, max_x, min_y, max_y, save_path,
        data=None, params_to_mark=[], plot_text=False):
    import matplotlib
    matplotlib.use("PDF")
    from matplotlib import pyplot as plt
    plt.figure()
    ax = plt.subplot()
    #if min_x and max_x:
    #    ax.set_xlim(min_x, max_x)
    #if min_y and max_y:
    #    ax.set_ylim(min_y, max_y)
    # data_final = data[data["params"].astype(str) == "[26, 0.64, 0.46]"]
    try:
        ax = data.plot(x=xaxis, y=yaxis, kind="scatter", ax=ax)
        if min_x and max_x:
           ax.set_xlim(min_x, max_x)
        if min_y and max_y:
           ax.set_ylim(min_y, max_y)
    except:
        plt.scatter(xaxis, yaxis, marker="o")
    for (x, y, label) in zip(data[xaxis], data[yaxis], parameters):
        if label in params_to_mark:
            plt.scatter(x, y, marker="o", color="r")
        if not plot_text and label not in params_to_mark:
            continue
        plt.annotate(label, # this is the text
             (x, y), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(0, 10), # distance from text to points (x,y)
             ha='center', # horizontal alignment can be left, right or center
             size=5)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    x_text=0.05
    y_text=0.9
    plt.text(x_text, 1.02, "CMS", fontsize='large', fontweight='bold',
        transform=ax.transAxes)
    upper_text = "private work"
    plt.text(x_text + 0.1, 1.02, upper_text, transform=ax.transAxes)
    # text = [self.dataset.process.label.latex, self.category.label.latex]
    # for t in text:
        # plt.text(x_text, y_text, t, transform=ax.transAxes)
        # y_text -= 0.05

    plt.savefig(save_path)
    plt.close('all')

if __name__ == '__main__':

    # with open("results_tri_optimised.json") as f:
        # d = json.load(f)


    # import glob
    # read_files = glob.glob("/eos/home-j/jleonhol/www/PNetAtHLT/PNetAtHLT/Optimisation/result/*.json")
    # d = []                                                                                                                                                                                                    
    # for f in read_files:                                                                                                                                                                                                
        # with open(f, "rb") as infile:                                                                                                                                                                                     
           # d.append(json.load(infile))

    df = pd.read_pickle("/afs/cern.ch/work/s/skeshri/TauHLT/Braden/Forked/TauTriggerDev/Optimisation/results_mutau.pickle")   

    print(df)

    # df = pd.DataFrame(d)
    # df.convert_dtypes()

    # with open("results.json") as f:
        # d_noditaujet = json.load(f)

    # # # # with open("results_deeptau.json") as f:
        # # # # dd = json.load(f)

    # # # # results = {
        # # # # "method": [
            # # # # "DeepTau",
            # # # # "DeepTau (no DiTauJet)",
            # # # # "PNet (no DiTauJet)",
        # # # # ],
        # # # # "rate": [
            # # # # dd["rate"],
            # # # # dd["noditaujet_rate"],
            # # # # df["rate_noditaujet"][0],
        # # # # ],
        # # # # "efficiency": [
            # # # # dd["efficiency"],
            # # # # dd["noditaujet_eff"],
            # # # # df["eff_noditaujet"][0],
        # # # # ]
    # # # # }
    # # # # results = pd.DataFrame(results)
    # # # # print(results)

    # plot("rate", "efficiency", df["params"], "Rate (Hz)", "Efficiency", 17.80, 18.20, 0.84, 0.86, "plot.pdf", data=df)
    # plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 60, 61, 0.675, 0.68, "plot_tri_pt_optimised_cut.pdf", data=df)
    # plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 60, 61, 0.679, 0.68, "plot_optimised_cut.pdf", data=df)
    plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 0, 100, 0., 1., "plot_optimised_ditau_new.pdf", data=df, params_to_mark=[["deeptau"],[0.48,0.4],[0.52,0.42],[0.56,0.47]])
    plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 48, 50, 0.7975, 0.81, "plot_optimised_ditau_medium_new.pdf", data=df, params_to_mark=[[0.56, 0.47]], plot_text=True)
    plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 39, 40, 0.77, 0.79, "plot_optimised_ditau_tight_new.pdf", data=df, params_to_mark=[[0.60, 0.50]], plot_text=True)
    # plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 0., 1., 0., 1., "plot_tri_pt_optimised_cut.pdf", data=df)
    #plot([15], [0.95], [[1,2]], "Rate (Hz)", "Efficiency", 10, 20, 0.9, 1., "plot.pdf")
