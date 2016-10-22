package weka.classifiers.evaluation;

import weka.core.Utils;
import weka.core.Instance;

import java.util.List;
import java.util.ArrayList;

public class LogRMSE extends AbstractEvaluationMetric 
											implements StandardEvaluationMetric {
											
  /** Sum of squared errors. */
   protected double m_SumSqrErr;

  /** The weight of all instances that had a class assigned to them. */
  protected double m_WithClass;
  
  /**
   * Returns the log root mean squared error.
   * 
   * @param sum the current sum of squared log differences
   * @param num the current number of instances
   * @return the log root mean squared error
   */
  public final double logRMSE(double sum,double num) {
  	return Math.sqrt(m_SumSqrErr / m_WithClass);
  }
    
  /**
   * a formatted string (suitable for displaying in the console or GUI output) that 
   * contains all the statistics that this metric computes
   * @return a formatted string
   */
  public String toSummaryString() {
  	 return "Log root mean square error         " + Utils.doubleToString(getStatistic("log rmse"), 12, 4) + "\n";

  }

  /**
   * Update stats for a nominal class. Does nothing because metrics are for regression only.
   * @param predictedDistribution the probabilities assigned to each class
   * @param instance the instance to be classified
   */
  public  void updateStatsForClassifier(double[] predictedDistribution, Instance instance) {
    // Do nothing
    }

  /**
   * Updates all the sum of sqr errs about a predictors performance for the current
   * test instance.
   * 
   * @param predictedValue the numeric value the classifier predicts
   * @param instance the instance to be classified
   */
  public void updateStatsForPredictor(double predictedValue, Instance instance) { 
    if (!instance.classIsMissing()) {
      if (!Utils.isMissingValue(predictedValue)) {
      	double diff = Math.log(predictedValue) - Math.log(instance.classValue());
      	m_SumSqrErr += diff*diff;
    	m_WithClass++;
      }
    }
  }
  
  /**
   * Return true if this evaluation metric can be computed when the class is
   * nominal
   * 
   * @return true if this evaluation metric can be computed when the class is
   *         nominal
   */
	@Override								
	public boolean appliesToNominalClass() { return false; }
	
  /**
   * Return true if this evaluation metric can be computed when the class is
   * numeric
   * 
   * @return true if this evaluation metric can be computed when the class is
   *         numeric
   */
	@Override								
	public boolean appliesToNumericClass() { return true; }

  /**
   * Get the name of this metric
   * 
   * @return the name of this metric
   */	
	@Override
	public String getMetricName() { return "Log RMSE"; }

  /**
   * Get a short description of this metric (algorithm, forumulas etc.).
   * 
   * @return a short description of this metric
   */	
	@Override
	public String getMetricDescription() {
		return "Root Mean Square Error of the log of predicted versus actual";
	}

  /**
   * Get a list of the names of the statistics that this metrics computes. E.g.
   * an information theoretic evaluation measure might compute total number of
   * bits as well as average bits/instance
   * 
   * @return the names of the statistics that this metric computes
   */	
	@Override 
	public List<String> getStatisticNames() {
		ArrayList<String> names = new ArrayList<String>();
    	names.add("log rmse");

    	return names;
	}

  /**
   * Get the value of the named statistic
   * 
   * @param statName the name of the statistic to compute the value for
   * @return the computed statistic or Utils.missingValue() if the statistic
   *         can't be computed for some reason
   */	
	@Override
	public double getStatistic(String statName) {
		return logRMSE(m_SumSqrErr,m_WithClass);
	}

}