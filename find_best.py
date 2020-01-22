import json
import numpy as np

def read_json(file_name):
	with open(file_name, 'r', encoding='utf8') as f:
		json_dict = json.load(f)
	return json_dict

def calculate_f1(confusion_mats):
	f1s = []
	for i in range(confusion_mats.shape[0]):
		tn, fp, fn, tp = confusion_mats[i].ravel()
		precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
		recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
		f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
		f1s.append(f1)
	return np.mean(f1s)

def print_results_single(results):
	best_acc = 0
	for setting in results.keys():
			acc = np.mean(results[setting]['predictor_acc'])
			if acc > best_acc:
				best_acc = acc
				neg_confusion_matrix = np.array(results[setting]['neg_confusion_mat'])
				pos_confusion_matrix = np.array(results[setting]['pos_confusion_mat'])
				f1 = calculate_f1(neg_confusion_matrix + pos_confusion_matrix)
				neg_fpr = np.mean(results[setting]['neg_fpr'])
				neg_fnr = np.mean(results[setting]['neg_fnr'])
				pos_fpr = np.mean(results[setting]['pos_fpr'])
				pos_fnr = np.mean(results[setting]['pos_fnr'])

	print('Setting {} -> Neg FPR: {:.5f}, Pos FPR: {:.5f}, Neg FNR: {:.5f}, Pos FNR: {:.5f}, pred acc: {:.5f}, pred F1: {:.5f}'.format(setting, neg_fpr, pos_fpr, neg_fnr, pos_fnr, best_acc, f1))


def print_results_double(results1, results2, tolerance=0.05):
	best_acc = 0
	for setting in results1.keys():
		neg_fpr = np.mean(results1[setting]['neg_fpr'] + results2[setting]['neg_fpr'])
		neg_fnr = np.mean(results1[setting]['neg_fnr'] + results2[setting]['neg_fnr'])
		pos_fpr = np.mean(results1[setting]['pos_fpr'] + results2[setting]['pos_fpr'])
		pos_fnr = np.mean(results1[setting]['pos_fnr'] + results2[setting]['pos_fnr'])
		if np.abs(neg_fnr - pos_fnr) < tolerance and np.abs(neg_fpr - pos_fpr) < tolerance:
			predictor_acc = np.mean(results1[setting]['predictor_acc'] + results2[setting]['predictor_acc'])
			neg_confusion_matrix = np.array(results1[setting]['neg_confusion_mat'])
			pos_confusion_matrix = np.array(results1[setting]['pos_confusion_mat'])
			f1_1 = calculate_f1(neg_confusion_matrix + pos_confusion_matrix)
			neg_confusion_matrix = np.array(results2[setting]['neg_confusion_mat'])
			pos_confusion_matrix = np.array(results2[setting]['pos_confusion_mat'])
			f1_2 = calculate_f1(neg_confusion_matrix + pos_confusion_matrix)
			f1 = np.mean([f1_1, f1_2])
			if predictor_acc > best_acc:
				best_acc = predictor_acc
				best_setting = setting
			print('Setting {} -> Neg FPR: {:.5f}, Pos FPR: {:.5f}, Neg FNR: {:.5f}, Pos FNR: {:.5f}, pred acc: {:.5f}, pred F1: {:.5f}'.format(setting, neg_fpr, pos_fpr, neg_fnr, pos_fnr, predictor_acc, f1))
	print('Best accuracy found: {} for setting {}'.format(best_acc, setting))


if __name__ == '__main__':

	file_name = '../data_biased.json'
	results = read_json(file_name)

	print('Without debias: ')
	print_results_single(results)

	file_name1 = '../data_debias_first.json'
	file_name2 = '../data_debias_second.json'
	results1 = read_json(file_name1)
	results2 = read_json(file_name2)

	print('With debias:')
	print_results_double(results1, results2, tolerance=0.05)

