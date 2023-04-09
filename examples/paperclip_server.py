from flask import Flask, jsonify

app = Flask(__name__)

ASSETS = {}


@app.route("/make/<item>")
def make_item(item):
    ASSETS.setdefault(item, 0)
    ASSETS[item] += 1
    return jsonify(ASSETS)


@app.route("/assets")
def assets():
    return jsonify(ASSETS)


@app.route("/reset")
def reset():
    ASSETS.clear()
    return jsonify(ASSETS)


if __name__ == "__main__":
    app.run(debug=False, port=8800)
