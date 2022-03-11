package ml.classifiers;

import java.util.HashMap;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

/**
 * An implementation of the Naive Bayes classifier.
 * 
 * @author Aidan Garton
 *
 */
public class NBClassifier implements Classifier {
	private double lambda = 0.01; // smoothing parameter
	private boolean usePositiveFeaturesOnly = false;
	private double totalCount = 0;

	// we store these values globally; they're initialized during training
	private Set<Double> allLabelIndices; // all possible labels in the data
	private Set<Integer> allFeatureIndices; // all possible features in the data

	// list of lists, one per class in the data set. Each index of the lists is the
	// number of occurrences of that feature at that index from the data set
	private HashMap<Double, HashMapCounter<Integer>> featureCounts;

	// maps different class labels to their number of occurrence in the data set
	private HashMap<Double, Integer> labelCounts;

	public NBClassifier() {
	}

	/**
	 * Setter method for the lambda hyper-parameter
	 * 
	 * @param lambda the value to set lambda to
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Setter method for denoting if the model should look at all features or only
	 * positive ones
	 * 
	 * @param usePositiveFeaturesOnly
	 */
	public void setUseOnlyPositiveFeatures(boolean usePositiveFeaturesOnly) {
		this.usePositiveFeaturesOnly = usePositiveFeaturesOnly;
	}

	/**
	 * Calculates the log10 conditional probability p(ex.features | label)
	 * 
	 * @param ex    the example to get a probability for
	 * @param label the conditional part of the probability
	 * @return the log10 probability that 'ex' is of class 'label'
	 */
	public double getLogProb(Example ex, double label) {
		// get number of occurrences of label
		double countOfLabel = labelCounts.get(label);
		double logProb = Math.log10(countOfLabel / totalCount);

		// iterate over all features or only positive depending on setting
		Set<Integer> featureSet = usePositiveFeaturesOnly ? ex.getFeatureSet() : allFeatureIndices;

		// iterate over features, calculating the conditional probability for each one
		for (int feature : featureSet) {
			double conditionalProb = getFeatureProb(feature, label);
			if (ex.getFeature(feature) > 0) {
				logProb += Math.log10(conditionalProb);
			} else if (!usePositiveFeaturesOnly) {
				logProb += Math.log10(1 - conditionalProb);
			}
		}

		return logProb;
	}

	/**
	 * Calculates p(featureIndex | label) for a provided feature index and label
	 * 
	 * @param featureIndex
	 * @param label
	 * @return p(featureIndex | label)
	 */
	public double getFeatureProb(int featureIndex, double label) {
		double numFeatureOccurrences = (double) (featureCounts.get(label).get(featureIndex)) + lambda;
		double numLabelOccurrences = labelCounts.get(label) + lambda * 2;

		return numFeatureOccurrences / numLabelOccurrences;
	}

	@Override
	public void train(DataSet data) {
		allLabelIndices = data.getLabels();

		// clear the model
		featureCounts = new HashMap<Double, HashMapCounter<Integer>>();
		labelCounts = new HashMap<Double, Integer>();

		for (double label : data.getLabels()) {
			featureCounts.put(label, new HashMapCounter<Integer>());
			labelCounts.put(label, 0);
		}

		// save ALL features for when not using only positive features
		allFeatureIndices = data.getAllFeatureIndices();

		for (Example e : data.getData()) {
			totalCount += 1;
			labelCounts.put(e.getLabel(), labelCounts.get(e.getLabel()) + 1);

			// the list corresponding to this example's class
			HashMapCounter<Integer> hm = featureCounts.get(e.getLabel());

			// for each feature of the current example, increment the count for it in the
			// list corresponding to its label
			for (int feature : e.getFeatureSet()) {
				if (e.getFeature(feature) != 0) {
					if (hm.containsKey(feature)) {
						hm.increment(feature);
					} else {
						hm.put(feature, 1);
					}
				}
			}
		}
	}

	/**
	 * Helper function to calculate the maximum probability label for a given
	 * example
	 * 
	 * @param example
	 * @return the largest conditional probability AND its associated label
	 */
	private double[] getMaxPredLabel(Example example) {
		double maxPred = Double.NEGATIVE_INFINITY, maxLabel = -1;

		for (double label : allLabelIndices) {
			double pred = getLogProb(example, label);

			if (pred > maxPred) {
				maxLabel = label;
				maxPred = pred;
			}
		}

		return new double[] { maxPred, maxLabel };
	}

	@Override
	public double classify(Example example) {
		return getMaxPredLabel(example)[1];
	}

	@Override
	public double confidence(Example example) {
		return getMaxPredLabel(example)[0];
	}

	public static void main(String[] args) {
		DataSet data = new DataSet("../assign7b-starter/data/simple.data", DataSet.TEXTFILE);
//		DataSet data = new DataSet("../assign7b-starter/data/titanic-train.csv", DataSet.CSVFILE);
//		DataSet data = new DataSet("../assign7b-starter/data/wines.train", DataSet.TEXTFILE);
		NBClassifier nb = new NBClassifier();

		nb.setUseOnlyPositiveFeatures(false);
		nb.setLambda(0.01);
		nb.train(data);

		// for pos labels
		for (int i : data.getAllFeatureIndices()) {
			System.out.println(nb.getFeatureProb(i, 1) + " " + data.getFeatureMap().get(i));
		}

		System.out.println();

		// for neg labels
		for (int i : data.getAllFeatureIndices()) {
			System.out.println(nb.getFeatureProb(i, -1) + " " + data.getFeatureMap().get(i));
		}
	}
}
