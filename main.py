"""
ESPnet experiment browser

TODO:
- auto reload when expdir/results/log changed
- color label for attention
- search function for all/conf/attention
- user-defined epoch for acc dashboard
- loss/cer/wer dashboard
- user-defined epoch for attention
- json, plot, html download
- chart.js zoom https://github.com/chartjs/chartjs-plugin-zoom
"""

import argparse
import flask
from flask import Flask, render_template, jsonify
import matplotlib as mpl
mpl.use('Agg')

app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--exp_root', required=True)
args = parser.parse_args()


class Results:
    def __init__(self, label, dir, theme):
        import os
        from glob import glob
        import json
        self.label = label
        self.color = str2color(dir, theme)
        c = mpl.colors.hex2color(self.color)
        self.rgba = "rgba({}, {}, {}, 0.05)".format(int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))
        self.color_s = self.color[1:]
        self.dir = os.path.abspath(dir)
        self.log = json.load(open(dir + "/log", "r"))
        self.idim, self.odim, self.config = json.load(open(dir + "/model.json", "r"))
        self.config["dir"] = self.dir[:-len("/results")]
        for log in self.log:
            if "validation/main/acc" in log:
                self.config["valid_acc"] = log["validation/main/acc"]
                self.config["train_acc"] = log["main/acc"]
                self.config["epoch"] = log["epoch"]
                self.config["elapsed_time"] = log["elapsed_time"]
                self.config["hour"] = log["elapsed_time"] / 3600
        self.atts = glob(dir + "/att_ws/*.png")
        self.train_label = self.label + "/train/acc"
        self.valid_label = self.label + "/valid/acc"

    def accumulate(self, key):
        ret = []
        for log in self.log:
            if key in log:
                ret.append(log[key])
        return ret

    def range(self, key):
        return list(range(len(self.accumulate(key))))

    def chart(self):
        train_acc = []
        valid_acc = []
        epoch_label = []
        for log in self.log:
            if "validation/main/acc" in log:
                train_acc.append(log["main/acc"])
                valid_acc.append(log["validation/main/acc"])
                epoch_label.append(log["epoch"])
        return {
            "labels": epoch_label,
            "datasets": [
                {
                    "label": self.train_label,
                    "data": train_acc,
                    "lineTension": 0,
                    "backgroundColor": 'transparent',
                    "borderColor": self.color,
                    "borderWidth": 2,
                    "borderDash": [2, 2],  # for dash line: - - -
                    "pointBackgroundColor": self.color,
                    "tableBackgroundColor": self.rgba,
                    "isTrain": True,
                    "hidden": True
                },
                {
                    "label": self.valid_label,
                    "data": valid_acc,
                    "lineTension": 0,
                    "backgroundColor": 'transparent',
                    "borderColor": self.color,
                    "borderWidth": 2,
                    "pointBackgroundColor": self.color,
                    "tableBackgroundColor": self.rgba,
                    "isTrain": False,
                    "hidden": False
                }
            ]}

    def conf(self, keys):
        ret = f"""<tr id="{Global.conf_table_row_prefix}{self.label}" style="background-color:{self.rgba}"
                      onclick="chartToggleTable('{self.label}', ['{self.train_label}', '{self.valid_label}'])">"""
        ret += f"""<td id="{Global.conf_table_val_prefix}{self.label}" style="background-color:{self.color}"></td>"""
        for k in keys:
            ret += "<td>{}</td>".format(self.config[k])
        return ret + "</tr>"

    def att(self, i=0):
        from os.path import basename
        from base64 import b64encode
        d = dict()
        if "epoch" not in self.config:
            app.logger.warning("no epoch at " + self.label)
            return d
        for a in sorted(self.atts):
            if "ep.%d." % self.config["epoch"] in a and "src_attn" in a:
                b = b64encode(open(a, "rb").read()).decode('utf-8')
                f = basename(a)
                d[f] = b
        return d


# TODO put these globals in any other place
class Global:
    color_themes = ["gist_stern", "rainbow", "gnuplot2", "viridis"]

    conf_table_id = "conf_table"
    conf_table_row_prefix = "conf_tr_"
    conf_table_val_prefix = "conf_td_"

    attention_accordion_id = "att_accordion"
    attention_accordion_prefix = "att_accordion_"


def str2color(s, theme):
    """
    :param str s:
    :param str theme: https://matplotlib.org/examples/color/colormaps_reference.html
    """
    import hashlib
    import matplotlib.pyplot as plt
    hash = int(hashlib.sha512(s.encode('utf-8')).hexdigest(), 16)
    f = hash / 2 ** 512
    rgb = plt.cm.get_cmap(theme)(f)
    return mpl.colors.rgb2hex(rgb)


def build_conf_table(results_list, conf_keys):
    conf_table = "<thead><tr>"
    conf_table += "<th>id</th>"
    for k in conf_keys:
        conf_table += "<th>{}</th>".format(k)
    conf_table += "</tr></thead>"
    conf_table += "<tbody>"
    for r in results_list:
        try:
            conf_table += r.conf(conf_keys)
        except KeyError as e:
            app.logger.warning(r.label + " has KeyError {}".format(e))
    conf_table += "</tbody>"
    return conf_table


@app.route('/')
def index():
    return top("viridis")


@app.route('/color/<theme>')
def color(theme):
    return top(theme)


def top(theme):
    from glob import glob
    # {% endfor %}
    results_list = []
    data = []
    epochs = []
    # TODO user defined dirs from browser
    for dir in glob(args.exp_root + "/**/results"):
        label = dir.split("/")[-2].strip()
        try:
            r = Results(label, dir, theme)
            if "valid_acc" not in r.config:
                continue
        except FileNotFoundError:
            app.logger.warning("file not found (maybe incompleted 1 epoch):" + dir)
            continue
        c = r.chart()
        results_list.append(r)
        data += c["datasets"]
        if len(c["labels"]) > len(epochs):
            epochs = c["labels"]
    results_list = sorted(results_list, key=lambda x: -x.config["valid_acc"])
    # TODO user defined keys from browser
    conf_key = ["hour", "epoch", "train_acc", "valid_acc", "opt", "lr_init", "ninit",
                "batch_size", "adim", "elayers", "eunits", "dlayers", "dunits", "dir"]
    return render_template('top.html', title='top',
                           results_list=results_list,
                           chart={"labels": epochs, "datasets": data},
                           conf_table=build_conf_table(results_list, conf_key),
                           Global=Global)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
