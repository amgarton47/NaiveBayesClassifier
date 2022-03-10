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
		double countOfLabel = labelCounts.get(ex.getLabel());
		double logProb = Math.log10(countOfLabel / totalCount);

		Set<Integer> featureSet = usePositiveFeaturesOnly ? ex.getFeatureSet() : allFeatureIndices;

		for (int feature : featureSet) {
			double numFeatureOccurrences = (double) (featureCounts.get(label).get(feature)) + lambda;
			double numLabelOccurrences = countOfLabel + lambda * allLabelIndices.size();

			double conditionalProb = numFeatureOccurrences / numLabelOccurrences;

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
		double num = (double) featureCounts.get(label).get(featureIndex);
		double den = (double) labelCounts.get(label);
		return num / den;
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

	@Override
	public double classify(Example example) {
		double maxPred = Double.NEGATIVE_INFINITY, maxLabel = -1;

		for (double label : allLabelIndices) {
			double pred = getLogProb(example, label);

			if (pred > maxPred) {
				maxLabel = label;
				maxPred = pred;
			}
		}

		return maxLabel;
	}

	@Override
	public double confidence(Example example) {
		double maxPred = Double.NEGATIVE_INFINITY;

		for (double label : allLabelIndices) {
			double pred = getLogProb(example, label);

			if (pred > maxPred) {
				maxPred = pred;
			}
		}

		return maxPred;
	}

	public static void main(String[] args) {
//		DataSet data = new DataSet("../assign7b-starter/data/simple.data", DataSet.TEXTFILE);
//		DataSet data = new DataSet("../assign7b-starter/data/titanic-train.csv", DataSet.CSVFILE);
		DataSet data = new DataSet("../assign7b-starter/data/wines.train", DataSet.TEXTFILE);
		NBClassifier nb = new NBClassifier();

		nb.setUseOnlyPositiveFeatures(true);
		nb.setLambda(0.7);
		nb.train(data);

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
