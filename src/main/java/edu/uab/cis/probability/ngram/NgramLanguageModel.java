package edu.uab.cis.probability.ngram;

import static com.google.common.collect.Lists.charactersOf;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A probabilistic n-gram language model.
 * 
 * @param <T>
 *          The type of items in the sequences over which the language model
 *          estimates probabilities.
 */
public class NgramLanguageModel<T> {

  
  enum Smoothing {
    /**
     * Do not apply smoothing. An n-gram w<sub>1</sub>,...,w<sub>n</sub> will
     * have its joint probability P(w<sub>1</sub>,...,w<sub>n</sub>) estimated
     * as #(w<sub>1</sub>,...,w<sub>n</sub>) / N, where N indicates the total
     * number of all 1-grams observed during training.
     * 
     * Note that we have defined only the joint probability of an n-gram here.
     * Deriving the conditional probability from the definition above is left as
     * an exercise.
     */
    NONE,

    /**
     * Apply Laplace smoothing. An n-gram w<sub>1</sub>,...,w<sub>n</sub> will
     * have its conditional probability
     * P(w<sub>n</sub>|w<sub>1</sub>,...,w<sub>n-1</sub>) estimated as (1 +
     * #(w<sub>1</sub>,...,w<sub>n</sub>)) / (V +
     * #(w<sub>1</sub>,...,w<sub>n-1</sub>)), where # indicates the number of
     * times an n-gram was observed during training and V indicates the number
     * of <em>unique</em> 1-grams observed during training.
     * 
     * Note that Laplace smoothing defines only the conditional probability of
     * an n-gram, not the joint probability.
     */
    LAPLACE
  }

  enum Representation {
    /**
     * Calculate probabilities in the normal range, [0,1].
     */
    PROBABILITY,
    /**
     * Calculate log-probabilities instead of probabilities. In every case where
     * probabilities would have been multiplied, take advantage of the fact that
     * log(P(x)*P(y)) = log(P(x)) + log(P(y)) and add log-probabilities instead.
     * This will improve efficiency since addition is faster than
     * multiplication, and will avoid some numerical underflow problems that
     * occur when taking the product of many small probabilities close to zero.
     */
    LOG_PROBABILITY
  }
  Smoothing smoothing_type;
  Representation rep_type;
  int gram, N, V;
  Map<List<T>, Integer> prob_model; 

  /**
   * Creates an n-gram language model.
   * 
   * @param n
   *          The number of items in an n-gram.
   * @param representation
   *          The type of representation to use for probabilities.
   * @param smoothing
   *          The type of smoothing to apply when estimating probabilities.
   */
  public NgramLanguageModel(int n, Representation representation, Smoothing smoothing) {
    // TODO
      this.smoothing_type = smoothing;
      this.rep_type = representation;
      this.gram = n;
      prob_model = new HashMap<List<T>, Integer>();
      N = 0;
      V = 0;
  }

  /**
   * Trains the language model with the n-grams from a sequence of items.
   * 
   * This typically involves collecting counts of n-grams that occurred in the
   * sequence.
   * 
   * @param sequence
   *          The sequence on which the model should be trained.
   */
  public void train(List<T> sequence) {
    // TODO
      int step;
      N += sequence.size();
      Set<T> unique = new HashSet<T>(sequence);
      unique.addAll(sequence);
      V+= unique.size();
      //System.out.println("N : " + N + "\n V :" +V);
        
      //for each gram from 1 to n;
      for(int i = 1;i<=gram;i++)
      {
          //System.out.println("Gram : " + i);
          step = i;
          //for each element
          for(int j = i;j<=sequence.size();j++)
          {
              List<T> subLs = sequence.subList(j-step, j);
              //System.out.println("P("+ j + "|[" + (j-step) + "," + j + "]");
              //System.out.println(subLs);
              if(!prob_model.containsKey(subLs))
                  prob_model.put(subLs, 1);
              else
                  prob_model.replace(subLs, prob_model.get(subLs)+1);
          }
      }      
  }
  /**
   * Return the estimated n-gram probability of the sequence:
   * 
   * P(w<sub>0</sub>,...,w<sub>k</sub>) = ∏<sub>i=0,...,k</sub>
   * P(w<sub>i</sub>|w<sub>i-n+1</sub>, w<sub>i-n+2</sub>, ..., w<sub>i-1</sub>)
   * 
   * For example, a 3-gram language model would calculate the probability of the
   * sequence [A,B,B,C,A] as:
   * 
   * P(A,B,B,C,A) = P(A)*P(B|A)*P(B|A,B)*P(C|B,B)*P(A|B,C)
   * 
   * The exact calculation of the conditional probabilities in this expression
   * depends on the smoothing method. See {@link Smoothing}.
   * 
   * The result is in the range [0,1] with {@link Representation#PROBABILITY}
   * and in the range (-∞,0] with {@link Representation#LOG_PROBABILITY}.
   * 
   * @param sequence
   *          The sequence of items whose probability is to be estimated.
   * @return The estimated probability of the sequence.
   */
  public double probability(List<T> sequence) {
    // TODO
      
      //if training set is zero or testing set is zero return prob = 0;
      if(N==0 || sequence.size() == 0)
          return 0;
      
      int step = gram - 1;
      int firstIndex, lastIndex;
      List<T> num = new ArrayList<T>();
      List<T> den = new ArrayList<T>();
      
      double indProb = 0,prob = 0;
      if(rep_type.equals(Representation.PROBABILITY))
          prob = 1;
      else if(rep_type.equals(Representation.LOG_PROBABILITY))
          prob = 0;
      for(int i = 0;i<sequence.size();i++)
      {
          firstIndex = i - step;
          if(firstIndex<0) firstIndex = 0;
          lastIndex = i;
          den.addAll(sequence.subList(firstIndex, lastIndex));
          num.addAll(den);
          num.add(sequence.get(lastIndex));
          //System.out.println(num +"/" + den);
          indProb = getProbBasedOnSmooting(num, den); 
          //System.out.print(prob);
          //System.out.print("*");
          if(rep_type.equals(Representation.PROBABILITY))
              prob *= indProb;
          else if(rep_type.equals(Representation.LOG_PROBABILITY))
              prob += Math.log(indProb);
          
          num.clear();
          den.clear();
          
      }
      //System.out.println("\nProb : " + prob);
    return prob;
  }
  private double getProbBasedOnSmooting(List<T> num, List<T>den)
  {
      Integer numVal, denValue;
      if(!prob_model.containsKey(num))
          numVal = 0;
      else
        numVal = prob_model.get(num);
      
      if(!den.isEmpty())
      {
          if(!prob_model.containsKey(den))
              denValue = 0;
          else
              denValue = prob_model.get(den);
          if(smoothing_type.equals(Smoothing.NONE)){
            //System.out.print((numVal+1)+"/"+(denValue+V)+"*");
            return numVal/(double)(denValue);
            
          }
          else if(smoothing_type.equals(Smoothing.LAPLACE)){
            //System.out.print((numVal+1)+"/"+(denValue+V)+"*");
            return (numVal+1)/(double)(denValue+V);          }
          
          
      }
      else {
          if(smoothing_type.equals(Smoothing.NONE)){
            //System.out.print(numVal+"/"+N+"*");
            return (double)numVal/N; 
          }
            else if(smoothing_type.equals(Smoothing.LAPLACE)){
                //System.out.print((numVal+1)+"/"+(N+V)+"*");
                return (double)(numVal+1)/(N+V);
            }
          }
      
      return 0.0;
  }

  public static void main(String[] args)
  {
//      NgramLanguageModel<Character> model =
//        new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.LAPLACE);
//      model.train(charactersOf("babbaaaa"));
//      model.probability(charactersOf("aaaabb"));
       NgramLanguageModel<Integer> model =
        new NgramLanguageModel<>(4, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
       model.train(Arrays.asList(1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0));
       model.probability(Arrays.asList(1, 0, 0, 0, 0, 0, 1));
//    NgramLanguageModel<Character> model =
//        new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.NONE);
//    model.train(charactersOf(""));
//      System.out.println("N : " + model.N + "V: " + model.V);
//      System.out.println( model.probability(charactersOf("aaaabb")));
//  
  }
}
