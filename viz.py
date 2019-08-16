import matplotlib.pyplot as plt


with open('/sdcard/Download/train_base_model.log', 'r') as fp:
    data = fp.read().split('\n')[:-1]

steps = []
train_metrics = {}
test_metrics = {}

for line in data:
    step, total, trains, tests, eta = line.split('\t')
    steps.append(int(step))
    
    for item in trains.split('  '):
        key = item[:item.rindex(' ')].strip()
        value = item[item.rindex(' '):].strip()
        
        if key not in train_metrics:
            train_metrics[key] = []
        train_metrics[key].append(float(value))
    
    for item in tests.split('  '):
        key = item[:item.rindex(' ')].strip()
        value = item[item.rindex(' '):].strip()
        
        if key not in test_metrics:
            test_metrics[key] = []
        test_metrics[key].append(float(value))


plt.subplot(211)
plt.plot(steps, train_metrics['loss'])
plt.plot(steps, test_metrics['loss'])

plt.subplot(212)
plt.plot(steps, train_metrics['top-1-accuracy'])
plt.plot(steps, test_metrics['top-1-accuracy'])

plt.show()