import matplotlib.pyplot as plt
import numpy as np

def read_train_test(dirlink):
	with open(dirlink, "rb") as fn:
		reader = fn.readlines()
		val_accs = []
		train_accs = []
		for line in reader:
			if "Epoch" in line:
				val_acc = line.split("Validation Acc:")[1].split(",")[0]
				train_acc = line.split("Train Acc:")[1]
				if "\\" in val_acc:
					val_acc = val_acc.replace("\\", "")
				if "\\" in train_acc:
					train_acc = train_acc.replace("\\", "")
				val_accs.append(float(val_acc))
				train_accs.append(float(train_acc))
		return val_accs, train_accs

"""
x = np.arange(10)

val_acc_baseline, train_acc_baseline = read_train_test("unigram_accuracy.txt")
sgd_acc_baseline_val, sgd_acc_baseline_train = read_train_test("sgd_optimizer.txt")
plt.gca().set_color_cycle(['grey', 'red', 'green', 'blue', 'yellow'])
x = [ i for i in range(0, 61)]
plt.plot(x, [0]+ val_acc_baseline)
plt.plot(x, [0] + sgd_acc_baseline_val)
#plt.plot(x, [0.04,1.64,1.54, 1.96, 3.98, 6.18, 10.98, 16.74, 15.1,30.74, 14.94, 18.18, 26.34, 21.72, 28.54, 23.1, 33.02, 27.58, 35.04, 31.22, 30.14, 31.6, 45.94, 25.52, 29.66, 42.36, 23.12, 25.4, 31.24, 35.44, 32.8, 34.36, 44.16, 38.9, 39.96, 31.96, 32.4, 52.32, 40.26, 43.38, 44.28, 35.72, 32.74, 36.5, 39.82, 34.26, 40.02, 33.46, 41.6, 45.04, 40.42, 40.26, 46.68, 44.32, 47.44, 41.88, 56.6, 38.76, 44.74, 42.74])

plt.legend(['validation accuracy of baseline', 'validation accuracy of SGD optimizer model'], loc='upper left')
plt.title("SGD Optimizer Model")
plt.ylabel("Accuracy of Model With Given Paramter")
plt.xlabel("Steps (Batch size 32)")
plt.ylim(0,100)
plt.xlim(0,)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.show()
"""
def make_graph(dirlinks, legend, title):
	# this is when you want to overlay
	plt.title(title)
	x = [ i for i in range(0, 61)]

	plt.gca().set_color_cycle(['grey', 'red', 'green', 'blue', 'yellow', 'black'])
	for dirlink in dirlinks:
		val_acc_baseline, train_acc_baseline = read_train_test(dirlink)
		plt.plot(x, [0]+ val_acc_baseline)
	#plt.plot(x, [0.04,1.64,1.54, 1.96, 3.98, 6.18, 10.98, 16.74, 15.1,30.74, 14.94, 18.18, 26.34, 21.72, 28.54, 23.1, 33.02, 27.58, 35.04, 31.22, 30.14, 31.6, 45.94, 25.52, 29.66, 42.36, 23.12, 25.4, 31.24, 35.44, 32.8, 34.36, 44.16, 38.9, 39.96, 31.96, 32.4, 52.32, 40.26, 43.38, 44.28, 35.72, 32.74, 36.5, 39.82, 34.26, 40.02, 33.46, 41.6, 45.04, 40.42, 40.26, 46.68, 44.32, 47.44, 41.88, 56.6, 38.76, 44.74, 42.74])

	plt.legend(legend, loc='upper left')
	plt.ylabel("Accuracy of Model With Given Parameter")
	plt.xlabel("Steps (Batch size 32)")
	plt.ylim(0,100)
	plt.xlim(0, 60)
	plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	plt.show()

"""
val_acc_baseline, train_acc_baseline = read_train_test("unigram_accuracy.txt")
val_acc200 = read_train_test("lr005.txt")
val_acc500 = read_train_test("lr01.txt")
val_acc1000 = read_train_test("lr05.txt")
val_acc5000 = read_train_test("lr1.txt")
f, axarr = plt.subplots(2, 2)
x = [ i for i in range(0, 61)]
axarr[0,1].plot(x, [0]+ val_acc_baseline)
axarr[0, 0].plot(x, [0]+ val_acc200[0])
axarr[0, 0].plot(x, [0]+ val_acc200[1])
axarr[0,0].legend(["Baseline validation accuracy (Learning Rate=0.01)", "Learning Rate = 0.05 validation accuracy", "Learning Rate = 0.05 train accuracy"], loc='lower left', fontsize=8)
axarr[0, 0].set_title('Accuracy Curve with Learning Rate = 0.05')
axarr[0, 0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axarr[0, 0].set_ylim(0,100)
axarr[0, 0].set_xlim(0, 60)
axarr[0, 0].set_ylabel("Accuracy of Model With Given Parameter", fontsize=8)
axarr[0, 0].set_xlabel("Steps (Batch size 32)", fontsize=8)

axarr[0,1].plot(x, [0]+ val_acc_baseline)
axarr[0, 1].plot(x, [0]+ val_acc500[0])
axarr[0, 1].plot(x, [0]+ val_acc500[1])
axarr[0,1].legend(["Baseline validation accuracy (Learning Rate = 0.01)", "Learning Rate = 0.1 validation accuracy", "Learning Rate =0.1 train accuracy"], loc='lower left', fontsize=8)
axarr[0, 1].set_title('Accuracy Curve with Learning Rate = 0.1')
axarr[0, 1].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axarr[0, 1].set_ylim(0,100)
axarr[0, 1].set_xlim(0, 60)
axarr[0, 1].set_ylabel("Accuracy of Model With Given Parameter", fontsize=8)
axarr[0, 1].set_xlabel("Steps (Batch size 32)", fontsize=8)


axarr[1, 0].plot(x, [0]+ val_acc_baseline)
axarr[1, 0].plot(x, [0]+ val_acc1000[0])
axarr[1, 0].plot(x, [0]+ val_acc1000[1])
axarr[1,0].legend(["Baseline validation accuracy (Learning Rate = 0.01)", "Learning Rate = 0.5 validation accuracy", "Learning Rate = 0.5 train accuracy"], loc='lower left', fontsize=8)
axarr[1,0].set_title('Accuracy Curve with Learning Rate = 0.5')
axarr[1, 0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axarr[1,0].set_ylim(0,100)
axarr[1,0].set_xlim(0, 60)
axarr[1, 0].set_ylabel("Accuracy of Model With Given Parameter", fontsize=8)
axarr[1, 0].set_xlabel("Steps (Batch size 32)", fontsize=8)


axarr[1, 1].plot(x, [0]+ val_acc_baseline)
axarr[1,1].plot(x, [0]+ val_acc5000[0])
axarr[1,1].plot(x, [0]+ val_acc5000[1])
axarr[1,1].legend(["Baseline validation accuracy  (Learning Rate = 0.01)", "Learning Rate = 1.0 validation accuracy", "Learning Rate = 1.0 train accuracy"], loc='lower left', fontsize=8)
axarr[1,1].set_title('Accuracy Curve with Learning Rate = 1.0')
axarr[1,1].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axarr[1, 1].set_ylim(0,100)
axarr[1, 1].set_xlim(0, 60)
axarr[1,1].set_ylabel("Accuracy of Model With Given Parameter", fontsize=8)
axarr[1,1].set_xlabel("Steps (Batch size 32)", fontsize=8)


plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.show()
"""
make_graph(["unigram_accuracy.txt", "reducedlr.txt"], ["Baseline (No Annealing)", "With Annealing"], "Effect of Annealing on Model")
