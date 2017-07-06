'''
Plot iteration information.

python examples/fashionpose/plot_iteration.py solver.log

'''

import argparse
import re
import time
import matplotlib.pyplot
import numpy as np

TIMESTAMP_FORMAT = r"%m%d %H:%M:%S.%f"

SOLVER_BEGIN = re.compile(
    r"^I(?P<timestamp>\d+ \d+:\d+:\d+\.\d+) +(?P<pid>\d+).*"
    r"Initializing solver from parameters:\s*$")
SOLVER_PARAM = re.compile(r"^(?P<name>[a-z_]+): (?P<value>[0-9e+\-.]+)$")
TRAINNET_FILE = re.compile(
    r"^I(?P<timestamp>\d+ \d+:\d+:\d+\.\d+) +(?P<pid>\d+).*"
    r"Creating training net from net file: (?P<file>\S+)$")

TRAIN_BEGIN = re.compile(
    r"^I(?P<timestamp>\d+ \d+:\d+:\d+\.\d+) +(?P<pid>\d+).*"
    r"Iteration (?P<iteration>\d+), loss = (?P<loss>[0-9e+\-.]+)$")
TRAIN_ITEM = re.compile(
    r"^I(?P<timestamp>\d+ \d+:\d+:\d+\.\d+) +(?P<pid>\d+).*"
    r"Train net output #\d+: (?P<key>\S+) = (?P<value>[0-9e+\-.]+) "
    r"\(\* \d+ = (?P<scaled_value>[0-9e+\-.]+) loss\)$")
TRAIN_END = re.compile(
    r"^I(?P<timestamp>\d+ \d+:\d+:\d+\.\d+) +(?P<pid>\d+).*"
    r"Iteration (?P<iteration>\d+), lr = (?P<lr>[0-9e+.\-]+)$")
TEST_BEGIN = re.compile(
    r"^I(?P<timestamp>\d+ \d+:\d+:\d+\.\d+) +(?P<pid>\d+).*"
    r"Iteration (?P<iteration>\d+), Testing net \(#\d+\)$")
TEST_ITEM = re.compile(
    r"^I(?P<timestamp>\d+ \d+:\d+:\d+\.\d+) +(?P<pid>\d+).*"
    r"Test net output #\d+: (?P<key>\S+) = (?P<value>[0-9e+\-.]+).*$")


def _parse(logfile):
    # Parse the log file.
    train_data = []
    test_data = []
    with open(logfile, 'r') as f:
        line = f.readline()
        while line:
            solver_match = SOLVER_BEGIN.match(line)
            train_match = TRAIN_BEGIN.match(line)
            test_match = TEST_BEGIN.match(line)
            if solver_match:
                # Initialize the line series.
                train_data.append({
                    'timestamp': [],
                    'iteration': [],
                    'loss': [],
                    'output': {},
                    'output_scaled': {},
                    'lr': [],
                    'count': 0
                    })
                test_data.append({
                    'timestamp': [],
                    'iteration': [],
                    'output': {},
                    'count': 0
                    })
                line = f.readline()
                while line:
                    item_match = SOLVER_PARAM.match(line)
                    file_match = TRAINNET_FILE.match(line)
                    if item_match:
                        print("{0}: {1}".format(
                            item_match.group('name'),
                            item_match.group('value')))
                    elif file_match:
                        print("net: {0}".format(file_match.group('file')))
                        train_data[-1]['net'] = file_match.group('file').replace(
                            'models/','').replace('/train_val.prototxt', '')
                        break
                    line = f.readline()
            elif train_match:
                t = time.strptime(train_match.group('timestamp'),
                                  TIMESTAMP_FORMAT)
                train_data[-1]['timestamp'].append(t)
                train_data[-1]['iteration'].append(
                    int(train_match.group('iteration')))
                train_data[-1]['loss'].append(float(train_match.group('loss')))
                train_data[-1]['count'] += 1
                line = f.readline()
                while line:
                    item_match = TRAIN_ITEM.match(line)
                    end_match = TRAIN_END.match(line)
                    if item_match:
                        key = item_match.group('key')
                        value = item_match.group('value')
                        if not key in train_data[-1]['output']:
                            train_data[-1]['output'][key] = []
                        train_data[-1]['output'][key].append(float(value))
                        value = item_match.group('scaled_value')
                        if not key in train_data[-1]['output_scaled']:
                            train_data[-1]['output_scaled'][key] = []
                        train_data[-1]['output_scaled'][key].append(
                            float(value))
                    elif end_match:
                        train_data[-1]['lr'].append(
                            float(end_match.group('lr')))
                        line = f.readline()
                        break
                    line = f.readline()
                if train_data[-1]['count'] % 1000 == 0:
                    print("Parsed {0} train entries".format(
                        train_data[-1]['count']))
            elif test_match:
                t = time.strptime(test_match.group('timestamp'),
                                  TIMESTAMP_FORMAT)
                test_data[-1]['timestamp'].append(t)
                test_data[-1]['iteration'].append(
                    int(test_match.group('iteration')))
                test_data[-1]['count'] += 1
                line = f.readline()
                while line:
                    item_match = TEST_ITEM.match(line)
                    if item_match:
                        key = item_match.group('key')
                        value = item_match.group('value')
                        if not key in test_data[-1]['output']:
                            test_data[-1]['output'][key] = []
                        test_data[-1]['output'][key].append(float(value))
                    else:
                        if TRAIN_BEGIN.match(line):
                            break
                    line = f.readline()
                if test_data[-1]['count'] % 1000 == 0:
                    print("Parsed {0} test entries".format(
                        test_data[-1]['count']))
            else:
                line = f.readline()
        print("Parsed {0} train entries with {1}".format(
            train_data[-1]['count'], ", ".join(
                train_data[-1]['output'].keys())))
        print("Parsed {0} test entries with {1}".format(
            test_data[-1]['count'], ", ".join(
                test_data[-1]['output'].keys())))
    return train_data, test_data


def plot_data(data, start_iter, fields=None, **kwargs):
    for j in xrange(len(data)):
        if fields is None:
            fields = sorted(data[j]['output'].keys())
        x = data[j]['iteration']
        if len(np.flatnonzero(np.array(x) >= start_iter))==0:
            continue
        start = np.flatnonzero(np.array(x) >= start_iter)[0]
        for field in fields:
            y = data[j]['output'][field]
            size = min(len(x), len(y))
            matplotlib.pyplot.plot(x[start:size], y[start:size], **kwargs)


def plot_loss(train_data, start_iter, **kwargs):
    all_fields = []
    for i in xrange(len(train_data)):
        x = train_data[i]['iteration']
        y = train_data[i]['loss']
        if len(np.flatnonzero(np.array(x) >= start_iter))==0:
            continue
        start = np.flatnonzero(np.array(x) >= start_iter)[0]
        size = min(len(x), len(y))
        matplotlib.pyplot.plot(x[start:size], y[start:size], zorder=10, **kwargs)
        fields = sorted(train_data[i]['output_scaled'].keys())
        for field in fields:
            y = train_data[i]['output_scaled'][field]
            size = min(len(x), len(y))
            matplotlib.pyplot.plot(x[start:size], y[start:size], **kwargs)
        all_fields += ['loss'] + fields
    matplotlib.pyplot.legend(all_fields, prop={'size': 10})


def main(options):
    train_data, test_data = _parse(options.logfile)
    fields = set()
    for i in xrange(len(train_data)):
        fields |= set(train_data[i]['output'].keys() +
                      test_data[i]['output'].keys())
    fields = list(fields)
    matplotlib.pyplot.subplot(1 + len(fields), 1, 1)
    matplotlib.pyplot.title(train_data[0]['net'])
    plot_loss(train_data, options.start)
    matplotlib.pyplot.grid()
    for i in xrange(len(fields)):
        all_fields = []
        matplotlib.pyplot.subplot(1 + len(fields), 1, 2 + i)
        matplotlib.pyplot.title(fields[i])
        plot_data(train_data, options.start, fields=[fields[i]])
        all_fields += ['train' for j in xrange(len(train_data))]
        
        #if test_data[0]['count']:
        plot_data(test_data, options.start, fields=[fields[i]])
        all_fields += ['val' for j in xrange(len(test_data))]

        matplotlib.pyplot.grid()
        matplotlib.pyplot.legend(all_fields, prop={'size': 10})
    matplotlib.pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot iteration information from a log.')
    parser.add_argument('logfile', type=str,
                        help='Training log file.')
    parser.add_argument('--start', type=int, default=0,
                        help='Start iteration.')
    main(parser.parse_args())
