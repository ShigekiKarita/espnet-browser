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
    def __init__(self, label, dir):
        import os
        from glob import glob
        import json
        self.label = label
        self.color = str2color(dir)
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
        self.atts = glob(dir + "/att_ws/*.png")

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
                    "label": self.label + "/train/acc",
                    "data": train_acc,
                    "lineTension": 0,
                    "backgroundColor": 'transparent',
                    "borderColor": self.color,
                    "borderWidth": 4,
                    "borderDash": [4, 4],  # for dash line: - - -
                    "pointBackgroundColor": self.color # '#007bff'
                },
                {
                    "label": self.label + "/valid/acc",
                    "data": valid_acc,
                    "lineTension": 0,
                    "backgroundColor": 'transparent',
                    "borderColor": self.color,
                    "borderWidth": 4,
                    "pointBackgroundColor": self.color
                }
            ]}

    def conf(self, keys):
        c = mpl.colors.hex2color(self.color)
        rgba = "rgba({}, {}, {}, 0.05)".format(int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))
        ret = "<tr>"
        ret += "<td style=\"background-color:{}\"></td>".format(self.color)
        for k in keys:
            ret += "<td style=\"background-color:{}\">{}</td>".format(rgba, self.config[k])
        return ret + "</tr>"

    def att(self, i=0):
        from os.path import basename
        from base64 import b64encode
        d = dict()
        for a in sorted(self.atts):
            if "ep.%d." % self.config["epoch"] in a and "src_attn" in a:
                b = b64encode(open(a, "rb").read()).decode('utf-8')
                f = basename(a)
                d[f] = b
        return d


themes = ["gist_stern", "rainbow", "gnuplot2"]

def str2color(s, theme=themes[-1]):
    """
    :param str s:
    :param str theme: https://matplotlib.org/examples/color/colormaps_reference.html
    """
    import hashlib
    import matplotlib.pyplot as plt
    hash = int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
    f = hash / 2 ** 128
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
        conf_table += r.conf(conf_keys)
    conf_table += "</tbody>"
    return conf_table


@app.route('/')
def top():
    from glob import glob
    # {% endfor %}
    results_list = []
    data = []
    epochs = []
    # TODO user defined dirs from browser
    for dir in glob(args.exp_root + "/**/results"):
        label = dir.split("/")[-2].strip()
        r = Results(label, dir)
        c = r.chart()
        results_list.append(r)
        data += c["datasets"]
        if len(c["labels"]) > len(epochs):
            epochs = c["labels"]

    # TODO user defined keys from browser
    conf_key = ["epoch", "train_acc", "valid_acc", "opt", "lr_init",
                "batch_size", "adim", "elayers", "eunits", "dlayers", "dunits", "dir"]
    return render_template('top.html', title='top',
                           results_list=results_list,
                           chart={"labels": epochs, "datasets": data},
                           conf_table=build_conf_table(results_list, conf_key),
                           att=results_list[0].att())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
