from typing import List, Tuple

# FIXME: Is this a good list of colors to use?
# FIXME: Right now alpha is always 1.0 ... should that be different?
colorlist: List[str] = ["#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
             "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43",
             "#8FB0FF", "#997D87", "#5A0007", "#809693", "#FEFFE6", "#1B4400",
             "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900",
             "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
             "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8",
             "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED", "#A079BF",
             "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062",
             "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
             "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648",
             "#0086ED", "#886F4C", ]

def colorHexToFloat(h: str) -> Tuple[float, float, float, float]:
    h = h.lstrip('#')
    r, g, b = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)

class ColorSequence:
    remain: List[str] = []

    def next(self) -> Tuple[float, float, float, float]:
        if len(self.remain) == 0:
            self.remain = colorlist.copy()

        v = self.remain.pop(0)
        return colorHexToFloat(v)
