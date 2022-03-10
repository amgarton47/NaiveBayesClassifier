package ml.data;

import ml.classifiers.NBClassifier;

public class Experimenter {
	public static void main(String[] args) {
		DataSet data = new DataSet("../assign7b-starter/data/wines.train", DataSet.TEXTFILE);
		NBClassifier nb = new NBClassifier();

		CrossValidationSet cvs = new CrossValidationSet(data, 10);

		double lambda = 10;
		for (int j = 0; j < 10; j++) {

			double correct = 0, total = 0;
			for (int i = 0; i < 10; i++) {
				DataSetSplit d = cvs.getValidationSet(i);

				nb.setUseOnlyPositiveFeatures(true);
				nb.setLambda(lambda);
				nb.train(d.getTrain());

				double c = 0, t = 0;
				for (Example e : d.getTest().getData()) {
					if (nb.classify(e) == e.getLabel()) {
						c++;
					}
					t++;
				}
				correct += c;
				total += t;
			}

			System.out.println("Average accuracy of all splits, with lambda=" + lambda + ": " + correct / total);
//			System.out.println("(" + lambda + ", " + correct / total + ")");
			lambda /= 10;
//			lambda += 0.00000000000001;
		}

		// check that probabilities are the same as calculated from assign 7A

		// for pos labels
//		for (int i : data.getAllFeatureIndices()) {
//			System.out.println(nb.getFeatureProb(i, 1) + " " + data.getFeatureMap().get(i));
//		}
//
//		System.out.println();
//
//		// for neg labels
//		for (int i : data.getAllFeatureIndices()) {
//			System.out.println(nb.getFeatureProb(i, -1) + " " + data.getFeatureMap().get(i));
//		}
	}
}
