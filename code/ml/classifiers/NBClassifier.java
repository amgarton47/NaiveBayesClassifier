package ml.classifiers;

import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

public class NBClassifier implements Classifier {

	private double lambda;
	private boolean usePositiveFeaturesOnly;
	private double countPos, countNeg;

	private Set<Integer> allFeatureIndices;

	// map of features to # of appearances in examples
	private HashMapCounter<Integer> countsPosLabel;
	private HashMapCounter<Integer> countsNegLabel;

	public NBClassifier() {
		lambda = 0.01;
		usePositiveFeaturesOnly = false;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public void setUseOnlyPositiveFeatures(boolean usePositiveFeaturesOnly) {
		this.usePositiveFeaturesOnly = usePositiveFeaturesOnly;
	}

	public double getLogProb(Example ex, double label) {
		double probOfLabel, countOfLabel;
		HashMapCounter<Integer> counts;
		if (label == 1) {
			probOfLabel = countPos / (countPos + countNeg);
			countOfLabel = countPos;
			counts = countsPosLabel;
		} else {
			probOfLabel = countNeg / (countPos + countNeg);
			countOfLabel = countNeg;
			counts = countsNegLabel;
		}
		double pred = Math.log10(probOfLabel);

		if (usePositiveFeaturesOnly) {
			for (int feature : ex.getFeatureSet()) {
				if (ex.getFeature(feature) > 0) {
					pred += Math.log10((counts.get(feature) + lambda) / (countOfLabel + lambda * 2));
				}
			}
		} else {
			for (int feature : allFeatureIndices) {
				if (ex.getFeature(feature) > 0) {
					pred += Math.log10((counts.get(feature) + lambda) / (countOfLabel + lambda * 2));
				} else {
					pred += Math.log10(1 - (counts.get(feature) + lambda) / (countOfLabel + lambda * 2));
				}
			}
		}

		return pred;
	}

	/**
	 * Calculates p(featureIndex | label) for a provided feature index and label
	 * 
	 * @param featureIndex
	 * @param label
	 * @return p(featureIndex | label)
	 */
	public double getFeatureProb(int featureIndex, double label) {
		HashMapCounter<Integer> featureCounts = label == 1 ? countsPosLabel : countsNegLabel;
		double labelTotal = label == 1 ? countPos : countNeg;

		return featureCounts.get(featureIndex) / labelTotal;
	}

	@Override
	public void train(DataSet data) {

		// clear the model
		countsPosLabel = new HashMapCounter<Integer>();
		countsNegLabel = new HashMapCounter<Integer>();

		countNeg = 0;
		countPos = 0;

		// save ALL features for when not using only positive features
		allFeatureIndices = data.getAllFeatureIndices();

		for (Example e : data.getData()) {
			if (e.getLabel() == 1) {
				countPos++;
				for (int feature : e.getFeatureSet()) {
					if (e.getFeature(feature) != 0) {
						if (countsPosLabel.containsKey(feature)) {
							countsPosLabel.increment(feature);
						} else {
							countsPosLabel.put(feature, 1);
						}
					}
				}
			} else {
				countNeg++;
				for (int feature : e.getFeatureSet()) {
					if (e.getFeature(feature) > 0) {
						if (countsNegLabel.containsKey(feature)) {
							countsNegLabel.increment(feature);
						} else {
							countsNegLabel.put(feature, 1);
						}
					}
				}
			}
		}
	}

	@Override
	public double classify(Example example) {
		return getLogProb(example, 1) >= getLogProb(example, -1) ? 1 : -1;
	}

	@Override
	public double confidence(Example example) {
		return Math.max(getLogProb(example, 1), getLogProb(example, -1));
	}

	public static void main(String[] args) {
//		DataSet data = new DataSet("../assign7b-starter/data/simple.data", DataSet.TEXTFILE);
		DataSet data = new DataSet("../assign7b-starter/data/titanic-train.csv", DataSet.CSVFILE);
		NBClassifier nb = new NBClassifier();

		nb.setUseOnlyPositiveFeatures(false);
		nb.setLambda(6);
		nb.train(data);

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
//			System.out.println(nb.getFeatureProb(i, -11) + " " + data.getFeatureMap().get(i));
//		}
//
//		System.out.println();

		// get accuracy of model
		double c = 0, t = 0;
		for (Example e : data.getData()) {
			if (nb.classify(e) == e.getLabel()) {
				c++;
			}
			t++;
		}

		System.out.println(c / t);
	}
}
