import sys


def progress_bar(value, endvalue, title, bar_length=40):
    percent = float(value) / endvalue
    arrow = '=' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0}: [{1}] {2}%".format(title, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
